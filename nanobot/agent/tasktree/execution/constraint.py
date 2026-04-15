"""DefaultConstraintAgent: generates ConstraintSet via LLM + memory-guided veto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.tasktree.models import ConstraintSet, NodeResult, TaskNode
from nanobot.agent.tasktree.scheduler import ConstraintAgent as ConstraintAgentProtocol

if TYPE_CHECKING:
    from nanobot.agent.agemem.retriever import MemoryRetriever
    from nanobot.providers.base import LLMProvider


@dataclass
class DefaultConstraintAgentConfig:
    """Configuration for DefaultConstraintAgent."""

    max_depth: int = 10
    forbidden_actions: list[str] | None = None
    failure_count_limit: int = 10  # N=10 as confirmed by user


class DefaultConstraintAgent:
    """ConstraintAgent backed by LLM call + MemoryRetriever-guided veto.

    For each node:
        1. Generate constraints via LLM (depth, forbidden actions, failure limit)
        2. Query MemoryRetriever for similar past failures (same root cause)
        3. If similar failures >= N → tighten failure_count_limit to enforce hard veto
    """

    def __init__(
        self,
        provider: LLMProvider,
        memory_retriever: MemoryRetriever | None = None,
        config: DefaultConstraintAgentConfig | None = None,
    ):
        self.provider = provider
        self.memory_retriever = memory_retriever
        self.config = config or DefaultConstraintAgentConfig()

    async def get_constraints(
        self,
        node: TaskNode,
        parent_result: NodeResult | None,
        root_goal: str,
    ) -> ConstraintSet:
        """Generate the constraint set for a node.

        Uses LLM to propose constraints, then augments with memory-guided veto
        if similar failures are found in past executions.
        """
        prompt = _build_constraint_prompt(node, parent_result, root_goal)

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.provider.chat(
                messages=messages,
                model=self.provider.get_default_model(),
                max_tokens=512,
            )
            text = response.content if hasattr(response, "content") else str(response)
            constraints = _parse_constraint_response(text, self.config)
        except Exception:
            logger.warning("ConstraintAgent LLM call failed, using conservative defaults")
            constraints = ConstraintSet(
                max_depth=self.config.max_depth,
                forbidden_actions=self.config.forbidden_actions or [],
                failure_count_limit=self.config.failure_count_limit,
            )

        # Memory-guided veto: check for similar past failures
        if self.memory_retriever is not None:
            constraints = self._apply_memory_veto(node, constraints)

        return constraints

    def _apply_memory_veto(self, node: TaskNode, constraints: ConstraintSet) -> ConstraintSet:
        """Query memory for similar failures and tighten constraints if found.

        If a similar failure is found, reduce failure_count_limit to enforce
        a harder veto — preventing the same mistake from repeating N times.
        """
        try:
            scored = self.memory_retriever.retrieve(
                query=f"TaskTree failure {node.goal}",
                top_k=10,
            )
            # Count entries tagged with "failure" and matching this goal
            similar_failures = [
                se for se in scored
                if se.entry.tags and "failure" in se.entry.tags
            ]
            count = len(similar_failures)

            if count > 0:
                logger.debug(
                    "ConstraintAgent: found {} similar past failures for node {}",
                    count,
                    node.id,
                )
                # Tighten failure_count_limit proportionally
                # If we've failed N times before, allow fewer retries
                tight_limit = max(1, self.config.failure_count_limit - count)
                # Also suggest adding to forbidden_actions if count is high
                forbidden = list(constraints.forbidden_actions)
                if count >= 3:
                    forbidden.append(f"repeat_failure_pattern_{node.id}")
                return ConstraintSet(
                    max_depth=constraints.max_depth,
                    forbidden_actions=forbidden,
                    failure_count_limit=tight_limit,
                )
        except Exception as e:
            logger.warning("MemoryRetriever query failed: {}", e)

        return constraints


def _build_constraint_prompt(
    node: TaskNode,
    parent_result: NodeResult | None,
    root_goal: str,
) -> str:
    """Build the prompt for constraint generation."""
    parent_info = ""
    if parent_result:
        parent_info = f"\nParent result: {parent_result.summary}"

    return f"""Given the following task node, propose execution constraints.

Root Goal: {root_goal}
Node: {node.id} (depth {node.depth})
Task: {node.goal}
{parent_info}

Respond with a JSON object with these fields:
- max_depth: maximum allowed depth from root (default 10)
- forbidden_actions: list of action names that should be blocked (e.g. ["rm_rf", "delete_file"])
- failure_count_limit: how many times to retry the same root cause before giving up (default 10)

Respond ONLY with the JSON object, no explanation."""


def _parse_constraint_response(text: str, config: DefaultConstraintAgentConfig) -> ConstraintSet:
    """Parse LLM response into a ConstraintSet."""
    import json
    import re

    text = text.strip()
    match = re.search(r'\{[^{}]*"[^"]+"\s*:[^}]+\}', text)
    if match:
        try:
            data = json.loads(match.group())
            return ConstraintSet(
                max_depth=int(data.get("max_depth", config.max_depth)),
                forbidden_actions=data.get("forbidden_actions", config.forbidden_actions or []),
                failure_count_limit=int(data.get("failure_count_limit", config.failure_count_limit)),
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback to defaults
    return ConstraintSet(
        max_depth=config.max_depth,
        forbidden_actions=config.forbidden_actions or [],
        failure_count_limit=config.failure_count_limit,
    )
