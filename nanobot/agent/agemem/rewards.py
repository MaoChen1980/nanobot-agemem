"""Reward functions for AgeMem RL training.

Implements the three reward components from the AgeMem paper:
- Rtask: Task completion quality (LLM-judge based)
- Rcontext: Context compression efficiency
- Rmemory: Memory storage quality

These rewards are used by MemoryHook to track memory policy performance.
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardResult:
    """Result of reward computation."""
    total: float
    breakdown: dict[str, float]
    details: dict[str, Any]


class RewardFunctions:
    """Collection of reward functions for AgeMem memory policy.

    These functions evaluate memory operations and return reward signals
    used for credit assignment in Phase 5 (GRPO).
    """

    def __init__(self, llm_judge: Any = None, model: str = "gpt-4o-mini"):
        self._llm_judge = llm_judge
        self._model = model

    def set_judge(self, llm_judge: Any, model: str | None = None) -> None:
        """Set the LLM judge for async Rtask evaluation."""
        self._llm_judge = llm_judge
        if model:
            self._model = model

    # ---------------------------------------------------------------------------
    # Rtask: Task completion reward
    # ---------------------------------------------------------------------------

    async def compute_rtask(
        self,
        final_response: str,
        task_description: str,
    ) -> float:
        """Rtask: Evaluate task completion quality using LLM-as-judge.

        Args:
            final_response: The agent's final response to the user
            task_description: The original task or question

        Returns:
            Score from 0.0 (completely failed) to 1.0 (perfectly completed)
        """
        if not self._llm_judge:
            return 0.5  # Neutral default

        prompt = f"""Evaluate how well the following response completes the task.

Task: {task_description}

Response: {final_response[:1000]}

Rate the response on a scale of 0.0 to 1.0:
- 1.0: Perfectly addresses the task, accurate and complete
- 0.7: Mostly addresses the task with minor gaps
- 0.5: Partially addresses the task, significant gaps
- 0.3: Minimally addresses the task, mostly incomplete
- 0.0: Completely failed or incorrect

Respond with only the numeric score."""

        try:
            response = await self._llm_judge.chat_with_retry(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
                max_tokens=10,
            )
            content = response.content or ""
            match = re.search(r"0?\.\d+", content)
            if match:
                return max(0.0, min(1.0, float(match.group())))
        except Exception:
            pass

        return 0.5

    def compute_rtask_sync(self, final_response: str, task_description: str) -> float:
        """Sync heuristic Rtask (when LLM judge is unavailable).

        Uses simple heuristics: response length, completeness indicators.
        """
        score = 0.5

        # Length heuristic
        if len(final_response) > 100:
            score += 0.1
        if len(final_response) > 500:
            score += 0.1

        # Completeness indicators
        if "?" in task_description and "?" in final_response:
            score += 0.1  # Answered a question
        if any(kw in final_response.lower() for kw in ["however", "therefore", "in summary"]):
            score += 0.05  # Shows reasoning

        # Error indicators
        if "i don't know" in final_response.lower() or "cannot" in final_response.lower():
            score -= 0.2

        return max(0.0, min(1.0, score))

    # ---------------------------------------------------------------------------
    # Rcontext: Context compression efficiency
    # ---------------------------------------------------------------------------

    def compute_rcontext(
        self,
        before_tokens: int,
        after_tokens: int,
        original_messages: list[dict[str, Any]],
        summary: str,
    ) -> float:
        """Rcontext: Evaluate context compression quality.

        Rewards efficient compression that preserves key information.

        Args:
            before_tokens: Token count before compression
            after_tokens: Token count after compression
            original_messages: Original message list
            summary: Generated summary text

        Returns:
            Score from 0.0 to 1.0 (compression efficiency + info preservation)
        """
        if before_tokens <= 0 or after_tokens <= 0:
            return 0.0

        compression_ratio = after_tokens / before_tokens

        # Optimal compression is around 20-50%
        efficiency_score = 0.0
        if 0.2 <= compression_ratio <= 0.5:
            efficiency_score = 0.5
        elif 0.1 <= compression_ratio < 0.2:
            efficiency_score = 0.4
        elif 0.5 < compression_ratio <= 0.7:
            efficiency_score = 0.3
        elif compression_ratio > 0.7:
            efficiency_score = 0.1  # Too little compression
        else:
            efficiency_score = 0.2  # Very aggressive compression

        # Information preservation heuristic
        info_score = 0.0
        summary_lower = summary.lower()

        # Check if key entities from original messages appear in summary
        key_indicators = ["user", "said", "asked", "told", "mentioned", "project", "deadline", "name"]
        preserved = sum(1 for kw in key_indicators if kw in summary_lower)
        info_score = preserved / len(key_indicators) * 0.5

        return max(0.0, min(1.0, efficiency_score + info_score))

    def compute_compression_ratio(self, before_tokens: int, after_tokens: int) -> float:
        """Simple compression ratio metric (no LLM needed)."""
        if before_tokens <= 0:
            return 0.0
        return min(1.0, after_tokens / before_tokens)

    # ---------------------------------------------------------------------------
    # Rmemory: Memory storage quality
    # ---------------------------------------------------------------------------

    def compute_rmemory(
        self,
        operation: str,
        memory_content: str,
        importance: float,
        context: str = "",
    ) -> float:
        """Rmemory: Evaluate memory storage quality.

        Rewards:
        - Adding high-quality, distinctive content
        - Appropriate importance scoring
        - Maintaining memory diversity

        Args:
            operation: "add", "update", "delete"
            memory_content: The content being stored
            importance: The assigned importance score (0.0-1.0)
            context: Optional context (task or conversation)

        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.5  # Base

        if operation == "add":
            # Check for distinctive content (not generic)
            if self._is_generic(memory_content):
                score -= 0.2
            else:
                score += 0.1

            # Importance appropriateness
            # High importance content should have high scores
            if importance >= 0.7 and len(memory_content) > 20:
                score += 0.15  # Substantial high-importance content
            elif importance <= 0.3 and len(memory_content) > 50:
                score -= 0.1  # Possibly too verbose for low importance

            # Content quality signals
            if any(kw in memory_content.lower() for kw in ["name", "preference", "always", "never"]):
                score += 0.1  # Specific, actionable content

        elif operation == "update":
            # Update is positive if improving relevance
            if importance > 0.7:
                score += 0.2  # Good to have high-importance memories updated

        elif operation == "delete":
            # Delete is positive if removing noise
            if importance < 0.3:
                score += 0.2  # Good to prune low-value memories
            elif importance >= 0.5:
                score -= 0.1  # Be careful deleting important content

        return max(0.0, min(1.0, score))

    def _is_generic(self, content: str) -> bool:
        """Check if content is generic/vague (low memory value)."""
        generic_phrases = [
            "i see", "okay", "thanks", "sure", "ok",
            "no problem", "you're welcome", "got it",
            "conversation", "discussed", "talked about",
        ]
        content_lower = content.lower()
        return any(phrase in content_lower for phrase in generic_phrases)

    # ---------------------------------------------------------------------------
    # Combined reward
    # ---------------------------------------------------------------------------

    def compute_combined(
        self,
        rtask: float | None = None,
        rcontext: float | None = None,
        rmemory: float | None = None,
        weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    ) -> float:
        """Compute weighted combined reward.

        Args:
            rtask: Task completion reward
            rcontext: Context compression reward
            rmemory: Memory storage reward
            weights: (w_task, w_context, w_memory) - must sum to 1.0
        """
        w_task, w_context, w_memory = weights

        total = 0.0
        total_weight = 0.0

        if rtask is not None:
            total += rtask * w_task
            total_weight += w_task
        if rcontext is not None:
            total += rcontext * w_context
            total_weight += w_context
        if rmemory is not None:
            total += rmemory * w_memory
            total_weight += w_memory

        if total_weight == 0:
            return 0.5

        return total / total_weight
