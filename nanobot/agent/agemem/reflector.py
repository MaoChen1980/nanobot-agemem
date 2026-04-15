"""Reflector: the conscious memory core for gap detection and reflection loops.

Implements the "有意识记忆" (conscious memory) cycle:
- Detects when memories are missing (gaps)
- Records gap events for learning
- Generates policy updates from reflection
"""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class GapEvent:
    """Records a single memory retrieval gap."""
    timestamp: str
    query: str
    retrieved_count: int
    was_empty: bool
    source: str  # "ltm" or "history"
    resolved: bool = False  # whether this gap was later filled


class Reflector:
    """Conscious memory: detects retrieval gaps and closes the learning loop.

    The reflection cycle:
    1. Gap detected -> recorded in gaps.jsonl
    2. Dream analysis -> calls reflect() to generate policy updates
    3. Policy updated -> agent now proactively remembers similar content
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.gaps_file = workspace / "memory" / "gaps.jsonl"
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.gaps_file.exists():
            self.gaps_file.parent.mkdir(parents=True, exist_ok=True)
            self.gaps_file.write_text("", encoding="utf-8")

    def on_retrieve(self, query: str, retrieved_count: int, was_empty: bool) -> None:
        """Called after retrieve_memories tool execution.

        Records the retrieval event. If was_empty=True, this is a gap event.
        """
        event = GapEvent(
            timestamp=datetime.now().isoformat(),
            query=query,
            retrieved_count=retrieved_count,
            was_empty=was_empty,
            source="ltm",
        )
        self._append_gap(event)
        if was_empty:
            logger.debug("Memory gap detected: query='{}'", query)

    def on_history_retrieval(self, query: str, history_entries: list[dict[str, Any]]) -> None:
        """Called when agent falls back to history because LTM was empty.

        This is a significant learning signal: the agent needed something
        it hadn't proactively stored.
        """
        event = GapEvent(
            timestamp=datetime.now().isoformat(),
            query=query,
            retrieved_count=len(history_entries),
            was_empty=len(history_entries) == 0,
            source="history",
        )
        self._append_gap(event)
        logger.info(
            "Memory gap from history: query='{}', found={} entries",
            query, len(history_entries),
        )

    def _append_gap(self, event: GapEvent) -> None:
        """Append a gap event to the JSONL log."""
        with open(self.gaps_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def get_gaps(self, unresolved_only: bool = True) -> list[GapEvent]:
        """Load all gap events from the log."""
        if not self.gaps_file.exists():
            return []
        events = []
        try:
            with open(self.gaps_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        event = GapEvent(**d)
                        if not unresolved_only or not event.resolved:
                            events.append(event)
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            pass
        return events

    def mark_resolved(self, query: str) -> None:
        """Mark gaps matching the query as resolved."""
        events = self.get_gaps(unresolved_only=True)
        if not events:
            return
        # Rewrite file with updated resolved flag
        all_events: list[GapEvent] = []
        try:
            with open(self.gaps_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        event = GapEvent(**d)
                        if event.query == query and not event.resolved:
                            event.resolved = True
                        all_events.append(event)
                    except (json.JSONDecodeError, TypeError):
                        continue
            with open(self.gaps_file, "w", encoding="utf-8") as f:
                for event in all_events:
                    f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def reflect(self, store: Any | None = None) -> list[dict[str, Any]]:
        """Analyze gap history and return recommended policy updates.

        This is the core reflection function that generates learning
        from memory gaps. In Phase 2 it uses simple pattern extraction.
        Phase 3 would use LLM for deeper analysis.

        Returns a list of recommended actions:
        - {"action": "add_rule", "pattern": "...", "importance": 0.8, "reason": "..."}
        - {"action": "increase_importance", "existing_id": "...", "delta": 0.2, "reason": "..."}
        """
        gaps = self.get_gaps(unresolved_only=True)
        if not gaps:
            return []

        recommendations = []

        # Group gaps by query similarity
        query_groups = self._group_similar_queries([g.query for g in gaps])

        for group_queries, count in query_groups.items():
            if count < 1:  # skip single occurrences initially
                continue

            # Extract key terms from the query group
            terms = self._extract_key_terms(group_queries)
            if not terms:
                continue

            # Determine recommended importance based on frequency
            # More gaps = higher importance needed
            importance = min(0.5 + (count * 0.1), 1.0)

            recommendations.append({
                "action": "add_rule",
                "pattern": " OR ".join(terms[:3]),  # top 3 terms
                "importance": round(importance, 2),
                "reason": f"Gap detected {count} time(s): '{group_queries[:50]}'",
                "gap_count": count,
                "queries": list(query_groups.keys())[:5],  # sample
            })

        # Sort by gap count descending
        recommendations.sort(key=lambda r: r.get("gap_count", 0), reverse=True)
        return recommendations[:10]  # top 10

    def _group_similar_queries(self, queries: list[str]) -> dict[str, int]:
        """Group queries by similarity (simple keyword overlap)."""
        groups: dict[str, int] = {}
        for q in queries:
            # Normalize query
            normalized = self._normalize_query(q)
            if not normalized:
                continue
            # Find existing group with high overlap
            best_group = None
            best_score = 0
            for group_key in groups:
                score = self._query_overlap(normalized, group_key)
                if score > best_score and score > 0.3:
                    best_score = score
                    best_group = group_key
            if best_group:
                groups[best_group] += 1
            else:
                groups[normalized] = 1
        return groups

    def _normalize_query(self, query: str) -> str:
        """Normalize a query for comparison."""
        return " ".join(re.findall(r"[a-z0-9]{2,}", query.lower()))

    def _query_overlap(self, q1: str, q2: str) -> float:
        """Compute keyword overlap between two normalized queries."""
        words1 = set(q1.split())
        words2 = set(q2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key distinctive terms from text.

        Filters out common stopwords and returns significant terms.
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "or", "if", "because", "until", "while", "about",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "it", "its", "they", "them", "their", "we", "us", "our",
            "you", "your", "he", "him", "his", "she", "her",
            "i", "me", "my", "myself", "any", "both", "either", "neither",
        }
        terms = re.findall(r"[a-z0-9]{3,}", text.lower())
        return [t for t in terms if t not in stopwords][:5]
