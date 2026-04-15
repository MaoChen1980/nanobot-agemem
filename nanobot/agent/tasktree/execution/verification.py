"""LLMVerificationAgent: verifies task completion via LLM judgment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tasktree.models import NodeResult
from nanobot.agent.tasktree.scheduler import VerificationAgent as VerificationAgentProtocol, VerificationResult
from nanobot.providers.base import LLMProvider

if TYPE_CHECKING:
    pass


class LLMVerificationAgent:
    """VerificationAgent backed by an LLM judge.

    Evaluates whether the execution results satisfy the root goal.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model: str | None = None,
    ):
        self.provider = provider
        self._model = model or provider.get_default_model()

    async def verify(
        self,
        root_goal: str,
        results: dict[str, NodeResult],
    ) -> VerificationResult:
        """Verify that results satisfy the root goal.

        Builds a prompt with the root goal and all node results,
        asks the LLM to judge pass/fail and return failed node ids.
        """
        prompt = _build_verification_prompt(root_goal, results)

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.provider.chat(
                messages=messages,
                model=self._model,
                max_tokens=1024,
            )
            text = response.content if hasattr(response, "content") else str(response)
            return _parse_verification_response(text)
        except Exception as e:
            return VerificationResult(
                passed=False,
                reason=f"Verification error: {e}",
                evidence=[],
            )


def _build_verification_prompt(
    root_goal: str,
    results: dict[str, NodeResult],
) -> str:
    """Build the verification prompt."""
    result_lines = []
    for node_id, result in results.items():
        lines = [f"### Node {node_id}", f"Summary: {result.summary}"]
        if result.artifacts:
            lines.append("Artifacts:")
            for a in result.artifacts:
                lines.append(f"  - [{a.type}] {a.path or ''}: {a.description}")
        result_lines.append("\n".join(lines))

    results_text = "\n\n".join(result_lines) if result_lines else "(no results)"

    return f"""Evaluate whether the following task execution satisfies the root goal.

Root Goal: {root_goal}

Execution Results:
{results_text}

Respond with a JSON object with these fields:
- passed: boolean, true if the root goal is satisfied
- failed_nodes: list of node IDs that failed or contributed to failure (empty if passed)
- reason: string explaining the judgment
- evidence: list of specific evidence supporting the judgment

Respond ONLY with the JSON object, no explanation."""


def _parse_verification_response(text: str) -> VerificationResult:
    """Parse LLM verification response into VerificationResult."""
    import json
    import re

    text = text.strip()
    match = re.search(r'\{[^{}]*"passed"[^{}]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            return VerificationResult(
                passed=bool(data.get("passed", False)),
                failed_nodes=data.get("failed_nodes", []),
                reason=data.get("reason", ""),
                evidence=data.get("evidence", []),
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: assume failed on parse error
    return VerificationResult(
        passed=False,
        reason=f"Could not parse verification response: {text[:200]}",
        evidence=[],
    )
