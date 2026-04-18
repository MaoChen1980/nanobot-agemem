"""TimestampedFact: atomic unit of causally-relevant information.

A Fact is the basic building block for causal memory. Each fact is:
- Timestamped (when it happened)
- Structured (content is a dict, not raw text)
- Typed (what kind of fact: action, state, event, etc.)
- Potentially causal (can be linked to other facts via causes/effects)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TimestampedFact:
    """A timestamped atomic fact for causal reasoning.

    Content structure (example):
        {"action": "write_file", "path": "x.txt", "result": "success"}
        {"state": "user_preference", "key": "theme", "value": "dark"}
        {"event": "task_completed", "task_id": "123", "duration": 120}
    """

    id: str
    timestamp: str  # ISO format datetime
    type: str  # "action" | "state" | "event" | "relationship" | ...
    content: dict[str, Any]  # structured fact data
    importance: float = 0.5  # 0.0-1.0

    # Causal links (causes = what led to this fact, effects = what this fact caused)
    causes: list[str] = field(default_factory=list)  # fact IDs that caused this
    effects: list[str] = field(default_factory=list)  # fact IDs this caused

    # Metadata
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0  # certainty of the causal link (0.0-1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type,
            "content": self.content,
            "importance": self.importance,
            "causes": self.causes,
            "effects": self.effects,
            "tags": self.tags,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimestampedFact":
        return cls(
            id=d["id"],
            timestamp=d.get("timestamp", datetime.now().isoformat()),
            type=d.get("type", "unknown"),
            content=d["content"],
            importance=d.get("importance", 0.5),
            causes=d.get("causes", []),
            effects=d.get("effects", []),
            tags=d.get("tags", []),
            confidence=d.get("confidence", 1.0),
        )

    def add_cause(self, cause_id: str) -> None:
        """Link a cause fact ID to this fact."""
        if cause_id not in self.causes:
            self.causes.append(cause_id)

    def add_effect(self, effect_id: str) -> None:
        """Link an effect fact ID to this fact."""
        if effect_id not in self.effects:
            self.effects.append(effect_id)
