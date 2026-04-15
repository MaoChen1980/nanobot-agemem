"""Structured long-term memory store (AgeMem LTM Layer).

Provides typed, addressable memory entries with JSONL persistence.
Replaces flat MEMORY.md with structured storage.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.agemem.entry import MemoryEntry
from nanobot.utils.helpers import ensure_dir


class MemoryStoreV2:
    """Structured LTM store with MemoryEntry CRUD operations.

    Persistence: JSONL file at {workspace}/memory/ltm.jsonl
    Each line is a JSON-encoded MemoryEntry.
    """

    def __init__(self, workspace: Path, max_entries: int = 1000):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.ltm_file = self.memory_dir / "ltm.jsonl"
        self.max_entries = max_entries
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load entries from JSONL file on startup."""
        self._entries.clear()
        if not self.ltm_file.exists():
            return
        try:
            with open(self.ltm_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        entry = MemoryEntry.from_dict(d)
                        if not entry.deleted:
                            self._entries[entry.id] = entry
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("Skipping malformed LTM entry: {}", line[:100])
        except OSError as e:
            logger.warning("Could not load LTM store: {}", e)

    def _save(self) -> None:
        """Persist all entries (including soft-deleted) to JSONL."""
        with open(self.ltm_file, "w", encoding="utf-8") as f:
            for entry in self._entries.values():
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._compact()

    def _compact(self) -> None:
        """Trim oldest entries if over max_entries (keeping highest importance)."""
        if len(self._entries) <= self.max_entries:
            return
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.access_count),
            reverse=True,
        )
        keep = sorted_entries[: self.max_entries]
        self._entries = {e.id: e for e in keep}
        logger.info("LTM compact: kept {} of {} entries", len(keep), len(sorted_entries))

    # -- CRUD ----------------------------------------------------------------

    def add(self, content: str, importance: float = 0.5, tags: list[str] | None = None) -> MemoryEntry:
        """Add a new memory entry and persist."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            importance=importance,
            tags=tags or [],
        )
        self._entries[entry.id] = entry
        self._save()
        logger.debug("LTM add: id={}, importance={}", entry.id, importance)
        return entry

    def update(
        self,
        id: str,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        """Update an existing entry. Returns None if not found."""
        entry = self._entries.get(id)
        if entry is None:
            return None
        if content is not None:
            entry.content = content
        if importance is not None:
            entry.importance = importance
        if tags is not None:
            entry.tags = tags
        entry.updated_at = datetime.now().isoformat()
        self._save()
        logger.debug("LTM update: id={}", id)
        return entry

    def delete(self, id: str) -> bool:
        """Soft-delete an entry. Returns False if not found."""
        entry = self._entries.get(id)
        if entry is None:
            return False
        entry.soft_delete()
        del self._entries[id]
        self._save()
        logger.debug("LTM delete: id={}", id)
        return True

    def get(self, id: str) -> MemoryEntry | None:
        """Get an entry by ID (no access count update)."""
        return self._entries.get(id)

    def get_all(self) -> list[MemoryEntry]:
        """Return all active entries sorted by updated_at desc."""
        return sorted(self._entries.values(), key=lambda e: e.updated_at, reverse=True)

    def query(
        self,
        tags: list[str] | None = None,
        min_importance: float | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Query entries by tags and/or minimum importance."""
        results = list(self._entries.values())
        if tags:
            results = [e for e in results if any(t in e.tags for t in tags)]
        if min_importance is not None:
            results = [e for e in results if e.importance >= min_importance]
        results.sort(key=lambda e: (e.importance, e.access_count), reverse=True)
        if limit:
            results = results[:limit]
        return results

    def record_access(self, id: str) -> None:
        """Increment access count for an entry."""
        entry = self._entries.get(id)
        if entry:
            entry.touch()
            self._save()

    # -- import from legacy MEMORY.md -----------------------------------------

    @staticmethod
    def migrate_from_text(workspace: Path, text: str) -> list[MemoryEntry]:
        """Parse legacy MEMORY.md text into MemoryEntry objects.

        Splits on double newlines or markdown headers.
        """
        entries = []
        current_content: list[str] = []
        current_tags: list[str] = []

        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#") and current_content:
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content="\n".join(current_content).strip(),
                    importance=0.5,
                    tags=current_tags,
                )
                entries.append(entry)
                current_content = []
                current_tags = []
            current_content.append(line)
        if current_content:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content="\n".join(current_content).strip(),
                importance=0.5,
                tags=current_tags,
            )
            entries.append(entry)
        return entries
