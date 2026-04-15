"""Memory entry data structures for AgeMem."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryEntry:
    """Structured long-term memory entry with AgeMem fields.

    Replaces the flat MEMORY.md text with typed, addressable entries.
    """

    id: str
    content: str
    importance: float  # 0.0-1.0
    tags: list[str] = field(default_factory=list)
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    deleted: bool = False  # soft delete flag

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "importance": self.importance,
            "tags": self.tags,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=d["id"],
            content=d["content"],
            importance=d.get("importance", 0.5),
            tags=d.get("tags", []),
            access_count=d.get("access_count", 0),
            created_at=d.get("created_at", datetime.now().isoformat()),
            updated_at=d.get("updated_at", datetime.now().isoformat()),
            deleted=d.get("deleted", False),
        )

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.updated_at = datetime.now().isoformat()

    def soft_delete(self) -> None:
        """Mark this entry as deleted without removing it."""
        self.deleted = True
        self.updated_at = datetime.now().isoformat()
