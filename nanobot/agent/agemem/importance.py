"""Importance scoring for AgeMem using LLM-as-judge.

Provides learned importance scores for memory entries based on:
- LLM-as-judge evaluation of content
- Access patterns (recency, frequency)
- Explicit user signals
- Reflection-based feedback
"""

import re
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from nanobot.agent.agemem.entry import MemoryEntry


class ImportanceScorer:
    """Computes importance scores for memory entries.

    Uses a combination of:
    - LLM-as-judge: direct evaluation of content importance
    - Access-based: frequency and recency of access
    - Signal-based: explicit/implicit importance signals
    """

    def __init__(
        self,
        provider: Any = None,  # LLMProvider - set before use
        model: str = "gpt-4o-mini",
    ):
        self._provider = provider
        self._model = model

    def set_provider(self, provider: Any, model: str | None = None) -> None:
        """Set the LLM provider for async scoring."""
        self._provider = provider
        if model:
            self._model = model

    async def score_async(self, content: str, context: str = "") -> float:
        """Async LLM-as-judge importance scoring (0.0-1.0).

        Args:
            content: The memory entry content to score
            context: Optional additional context (e.g., recent conversation)
        """
        if not self._provider:
            return 0.5  # Default

        prompt = f"""Rate the importance of the following memory on a scale of 0.0 to 1.0.

Memory: "{content}"
{('Context: ' + context) if context else ''}

Consider:
- 1.0: Critical personal facts (name, allergies, health conditions, financial info)
- 0.8: Important preferences and patterns (work style, communication preferences)
- 0.6: Useful knowledge (project context, skill-related info)
- 0.4: Minor preferences or one-time facts
- 0.2: Trivial conversational content
- 0.0: Not worth remembering

Respond with only the numeric score (e.g., "0.7")."""

        try:
            response = await self._provider.chat_with_retry(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
                max_tokens=10,
            )
            content_str = response.content or ""
            # Parse the score
            match = re.search(r"0?\.\d+", content_str)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))
        except Exception:
            logger.warning("LLM importance scoring failed, using default 0.5")

        return 0.5

    def score_sync(self, content: str, access_count: int = 0, created_at: str = "") -> float:
        """Sync heuristic scoring based on access patterns.

        This is a fast fallback when LLM is not available.
        Returns a score in 0.0-1.0.
        """
        base_score = 0.5

        # Boost for frequent access
        if access_count > 10:
            base_score += 0.2
        elif access_count > 5:
            base_score += 0.1
        elif access_count > 2:
            base_score += 0.05

        # Recency bonus (simplified - no actual datetime parsing)
        # In production, would parse created_at ISO string
        if created_at:
            try:
                created = datetime.fromisoformat(created_at)
                age_days = (datetime.now() - created).days
                if age_days < 7:
                    base_score += 0.1  # Recent memory bonus
                elif age_days > 90:
                    base_score -= 0.1  # Old memory penalty
            except (ValueError, OSError):
                pass

        return max(0.0, min(1.0, base_score))

    def extract_importance_signals(self, content: str) -> float | None:
        """Extract explicit importance signals from content.

        Looks for phrases like:
        - "IMPORTANT:" / "CRITICAL:" / "remember this forever"
        - "please always..." / "never forget"
        - "this is secret" / "confidential"

        Returns importance hint or None.
        """
        content_lower = content.lower()

        # Critical signals
        critical_signals = [
            "critical", "allergy", "allergic", "medical", "health",
            "life-threatening", "emergency contact", "blood type",
            "social security", "password", "secret key",
        ]
        for signal in critical_signals:
            if signal in content_lower:
                return 0.95

        # High importance signals
        high_signals = [
            "important", "remember this", "never forget",
            "always do", "preference", "don't forget",
            "must remember", "please remember",
        ]
        for signal in high_signals:
            if signal in content_lower:
                return 0.7

        # Medium importance signals
        medium_signals = [
            "noted", "keep in mind", "mention",
            "i like", "i prefer", "i hate",
        ]
        for signal in medium_signals:
            if signal in content_lower:
                return 0.5

        # Low/minor signals
        low_signals = [
            "btw", "by the way", "fyi",
            "maybe", "probably", "might",
        ]
        for signal in low_signals:
            if signal in content_lower:
                return 0.2

        return None  # No signal found
