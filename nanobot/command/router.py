"""Minimal command routing table for slash commands."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

if TYPE_CHECKING:
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from nanobot.session.manager import Session

Handler = Callable[["CommandContext"], Awaitable["OutboundMessage | None"]]


def _normalize_gitbash_path(text: str) -> str:
    """Strip Git Bash expanded path prefix.

    Git Bash translates Unix-style paths like /taskinfo to
    C:/Program Files/Git/taskinfo on Windows. This reverses that
    translation so commands like /taskinfo are recognized correctly.
    """
    # Commands that need Git Bash path normalization
    normalized_commands = ("taskplan", "taskinfo", "taskstatus", "taskcancel")

    try:
        git_base = os.environ.get("MINGW_PREFIX", "").replace("\\", "/").strip("/").lower()
        if git_base:
            prefixes = [f"c:/{git_base}/", f"C:/{git_base}/"]
            for prefix in prefixes:
                if text.lower().startswith(prefix.lower()):
                    rest = text[len(prefix):]
                    for cmd in normalized_commands:
                        if rest.lower().startswith(cmd):
                            return "/" + rest
                    return rest
    except Exception:
        pass
    # Fallback: known Git Bash path patterns
    for prefix in ("c:/program files/git/", "C:/Program Files/Git/"):
        if text.lower().startswith(prefix.lower()):
            rest = text[len(prefix):]
            for cmd in normalized_commands:
                if rest.lower().startswith(cmd):
                    return "/" + rest
    return text


@dataclass
class CommandContext:
    """Everything a command handler needs to produce a response."""

    msg: InboundMessage
    session: Session | None
    key: str
    raw: str
    args: str = ""
    loop: Any = None


class CommandRouter:
    """Pure dict-based command dispatch.

    Three tiers checked in order:
      1. *priority* — exact-match commands handled before the dispatch lock
         (e.g. /stop, /restart).
      2. *exact* — exact-match commands handled inside the dispatch lock.
      3. *prefix* — longest-prefix-first match (e.g. "/team ").
      4. *interceptors* — fallback predicates (e.g. team-mode active check).
    """

    def __init__(self) -> None:
        self._priority: dict[str, Handler] = {}
        self._priority_prefix: list[tuple[str, Handler]] = []  # (prefix, handler), longest first
        self._exact: dict[str, Handler] = {}
        self._prefix: list[tuple[str, Handler]] = []
        self._interceptors: list[Handler] = []

    def priority(self, cmd: str, handler: Handler) -> None:
        self._priority[cmd] = handler

    def priority_prefix(self, pfx: str, handler: Handler) -> None:
        """Register a prefix command that is checked as priority (before lock, before routing).

        Unlike normal prefix commands, priority_prefix commands are handled before
        the dispatch lock is acquired — useful for commands like /taskplan that
        should be universal and lightweight.
        """
        self._priority_prefix.append((pfx, handler))
        self._priority_prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def exact(self, cmd: str, handler: Handler) -> None:
        self._exact[cmd] = handler

    def prefix(self, pfx: str, handler: Handler) -> None:
        self._prefix.append((pfx, handler))
        self._prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def intercept(self, handler: Handler) -> None:
        self._interceptors.append(handler)

    def is_priority(self, text: str) -> bool:
        """Check if text matches a priority command (exact or prefix)."""
        text_lower = text.strip().lower()
        # Normalize Git Bash path expansion: C:/Program Files/Git/taskinfo -> /taskinfo
        text_normalized = _normalize_gitbash_path(text_lower)
        if text_normalized != text_lower:
            logger.debug("is_priority: normalized Git Bash path from {!r} to {!r}", text_lower, text_normalized)
            text_lower = text_normalized
        if text_lower in self._priority:
            logger.debug("is_priority {!r}: exact match in _priority", text_lower)
            return True
        # Also check priority prefix commands
        for pfx, _ in self._priority_prefix:
            if text_lower.startswith(pfx):
                logger.debug("is_priority {!r}: prefix match {!r}", text_lower, pfx)
                return True
        logger.debug("is_priority {!r}: no match (priority={}, prefix={})", text_lower, list(self._priority.keys()), [p for p, _ in self._priority_prefix])
        return False

    async def dispatch_priority(self, ctx: CommandContext) -> OutboundMessage | None:
        """Dispatch a priority command (exact or prefix). Called from run() without the lock."""
        raw_lower = ctx.raw.strip().lower()
        # Normalize Git Bash path expansion: C:/Program Files/Git/taskinfo -> /taskinfo
        raw_normalized = _normalize_gitbash_path(raw_lower)
        if raw_normalized != raw_lower:
            logger.debug("dispatch_priority: normalized Git Bash path from {!r} to {!r}", raw_lower, raw_normalized)
            raw_lower = raw_normalized
        # Try exact priority first
        handler = self._priority.get(raw_lower)
        if handler:
            return await handler(ctx)
        # Try priority prefix commands (longest match first)
        for pfx, handler in self._priority_prefix:
            if raw_lower.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]  # Use raw (with original casing) for args
                return await handler(ctx)
        return None

    async def dispatch(self, ctx: CommandContext) -> OutboundMessage | None:
        """Try exact, prefix, then interceptors. Returns None if unhandled."""
        cmd = ctx.raw.lower()

        if handler := self._exact.get(cmd):
            return await handler(ctx)

        for pfx, handler in self._prefix:
            if cmd.startswith(pfx):
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)

        for interceptor in self._interceptors:
            result = await interceptor(ctx)
            if result is not None:
                return result

        return None
