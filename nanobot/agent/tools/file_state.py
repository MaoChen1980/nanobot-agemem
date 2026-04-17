"""Track file-read state for read-before-edit warnings and read deduplication."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReadState:
    mtime: float
    offset: int
    limit: int | None
    content_hash: str | None
    can_dedup: bool


_state: dict[str, ReadState] = {}


def _hash_file(p: str) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def record_read(path: str | Path, offset: int = 1, limit: int | None = None) -> None:
    """Record that a file was read (called after successful read)."""
    p = str(Path(path).resolve())
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return
    _state[p] = ReadState(
        mtime=mtime,
        offset=offset,
        limit=limit,
        content_hash=_hash_file(p),
        can_dedup=True,
    )


def record_write(path: str | Path) -> None:
    """Record that a file was written (updates mtime in state)."""
    p = str(Path(path).resolve())
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        _state.pop(p, None)
        return
    _state[p] = ReadState(
        mtime=mtime,
        offset=1,
        limit=None,
        content_hash=_hash_file(p),
        can_dedup=False,
    )


def check_read(path: str | Path) -> str | None:
    """Check if a file has been read and is fresh.

    Returns None if OK, or a warning string.
    When mtime changed but file content is identical (e.g. touch, editor save),
    the check passes to avoid false-positive staleness warnings.
    """
    p = str(Path(path).resolve())
    entry = _state.get(p)
    if entry is None:
        return "Warning: file has not been read yet. Read it first to verify content before editing."
    try:
        current_mtime = os.path.getmtime(p)
    except OSError:
        return None
    if current_mtime != entry.mtime:
        if entry.content_hash and _hash_file(p) == entry.content_hash:
            entry.mtime = current_mtime
            return None
        return "Warning: file has been modified since last read. Re-read to verify content before editing."
    return None


def is_unchanged(path: str | Path, offset: int = 1, limit: int | None = None) -> bool:
    """Return True if file was previously read with same params and content is unchanged.

    Uses mtime as the primary check. When mtime matches, also verifies content hash
    to handle Windows filesystem timestamp resolution issues where rapid writes
    within the same second may not update mtime.
    """
    p = str(Path(path).resolve())
    entry = _state.get(p)
    if entry is None:
        return False
    if not entry.can_dedup:
        return False
    if entry.offset != offset or entry.limit != limit:
        return False
    try:
        current_mtime = os.path.getmtime(p)
    except OSError:
        return False
    if current_mtime != entry.mtime:
        return False
    # mtime matched — verify content hash to handle Windows timestamp resolution
    if entry.content_hash is not None:
        current_hash = _hash_file(p)
        if current_hash != entry.content_hash:
            return False
    return True


def clear() -> None:
    """Clear all tracked state (useful for testing)."""
    _state.clear()
