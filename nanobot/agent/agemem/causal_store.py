"""CausalStore: storage and retrieval for causally-linked TimestampedFacts.

Provides causal graph operations:
- Store facts with causal links (causes → fact → effects)
- Query by time range
- Query by causal chain (what caused X? what did X cause?)
- Query by content similarity (embedding-based relevance)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.agemem.fact import TimestampedFact
from nanobot.utils.helpers import ensure_dir


class CausalStore:
    """Causal fact storage with JSONL persistence.

    Persistence: JSONL file at {workspace}/memory/causal.jsonl
    Each line is a JSON-encoded TimestampedFact with causal links.
    """

    def __init__(self, workspace: Path, max_facts: int = 5000):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.causal_file = self.memory_dir / "causal.jsonl"
        self.max_facts = max_facts
        self._facts: dict[str, TimestampedFact] = {}
        self._load()

    def _load(self) -> None:
        """Load facts from JSONL file on startup."""
        self._facts.clear()
        if not self.causal_file.exists():
            return
        try:
            with open(self.causal_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        fact = TimestampedFact.from_dict(d)
                        self._facts[fact.id] = fact
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed causal entry: {} — {}", line[:100], e)
        except OSError as e:
            logger.warning("Could not load causal store: {}", e)

    def _save(self) -> None:
        """Persist all facts to JSONL."""
        with open(self.causal_file, "w", encoding="utf-8") as f:
            for fact in self._facts.values():
                f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
        self._compact()

    def _compact(self) -> None:
        """Trim oldest/lowest importance facts if over max_facts."""
        if len(self._facts) <= self.max_facts:
            return
        sorted_facts = sorted(
            self._facts.values(),
            key=lambda f: (f.importance, f.timestamp),
            reverse=True,
        )
        keep = sorted_facts[: self.max_facts]
        self._facts = {f.id: f for f in keep}
        logger.info("Causal compact: kept {} of {} facts", len(keep), len(sorted_facts))

    # -- CRUD ----------------------------------------------------------------

    def add(self, fact: TimestampedFact) -> TimestampedFact:
        """Add a fact and persist."""
        self._facts[fact.id] = fact
        self._save()
        logger.debug("CausalStore add: id={}, type={}", fact.id, fact.type)
        return fact

    def add_fact(
        self,
        content: dict[str, Any],
        fact_type: str,
        importance: float = 0.5,
        timestamp: str | None = None,
        tags: list[str] | None = None,
    ) -> TimestampedFact:
        """Convenience method to create and add a fact."""
        fact = TimestampedFact(
            id=str(uuid.uuid4()),
            timestamp=timestamp or datetime.now().isoformat(),
            type=fact_type,
            content=content,
            importance=importance,
            tags=tags or [],
        )
        return self.add(fact)

    def get(self, id: str) -> TimestampedFact | None:
        """Get a fact by ID."""
        return self._facts.get(id)

    def get_all(self) -> list[TimestampedFact]:
        """Return all facts sorted by timestamp desc."""
        return sorted(self._facts.values(), key=lambda f: f.timestamp, reverse=True)

    def link_causal(self, cause_id: str, effect_id: str, confidence: float = 1.0) -> bool:
        """Create a causal link: cause_id → effect_id.

        Returns False if either fact doesn't exist.
        """
        cause = self._facts.get(cause_id)
        effect = self._facts.get(effect_id)
        if not cause or not effect:
            return False

        cause.add_effect(effect_id)
        effect.add_cause(cause_id)
        cause.confidence = min(cause.confidence, confidence)
        effect.confidence = min(effect.confidence, confidence)
        self._save()
        logger.debug("Causal link: {} → {} (conf={})", cause_id, effect_id, confidence)
        return True

    def get_causes(self, fact_id: str) -> list[TimestampedFact]:
        """Get all facts that are direct causes of this fact."""
        fact = self._facts.get(fact_id)
        if not fact:
            return []
        return [self._facts[cid] for cid in fact.causes if cid in self._facts]

    def get_effects(self, fact_id: str) -> list[TimestampedFact]:
        """Get all facts that are direct effects of this fact."""
        fact = self._facts.get(fact_id)
        if not fact:
            return []
        return [self._facts[eid] for eid in fact.effects if eid in self._facts]

    def get_causal_chain(
        self, fact_id: str, depth: int = 3, direction: str = "both"
    ) -> list[TimestampedFact]:
        """Get causal chain (causes or effects) up to depth levels.

        direction: "causes" | "effects" | "both"
        """
        visited: set[str] = {fact_id}
        result: list[TimestampedFact] = []
        frontier = [fact_id]

        for _ in range(depth):
            if not frontier:
                break
            next_frontier: list[str] = []
            for fid in frontier:
                fact = self._facts.get(fid)
                if not fact:
                    continue

                if direction in ("causes", "both"):
                    for cid in fact.causes:
                        if cid not in visited:
                            visited.add(cid)
                            next_frontier.append(cid)
                            cf = self._facts.get(cid)
                            if cf:
                                result.append(cf)

                if direction in ("effects", "both"):
                    for eid in fact.effects:
                        if eid not in visited:
                            visited.add(eid)
                            next_frontier.append(eid)
                            ef = self._facts.get(eid)
                            if ef:
                                result.append(ef)
            frontier = next_frontier

        return result

    def query_by_time_range(
        self, start: str, end: str, limit: int | None = None
    ) -> list[TimestampedFact]:
        """Query facts within a time range (ISO format timestamps)."""
        results = [
            f for f in self._facts.values()
            if start <= f.timestamp <= end
        ]
        results.sort(key=lambda f: f.timestamp, reverse=True)
        if limit:
            results = results[:limit]
        return results

    def query_by_content(
        self, content_pattern: dict[str, Any], limit: int | None = None
    ) -> list[TimestampedFact]:
        """Query facts by content field match (shallow check)."""
        results = []
        for fact in self._facts.values():
            match = True
            for key, value in content_pattern.items():
                if fact.content.get(key) != value:
                    match = False
                    break
            if match:
                results.append(fact)
        results.sort(key=lambda f: (f.importance, f.timestamp), reverse=True)
        if limit:
            results = results[:limit]
        return results
