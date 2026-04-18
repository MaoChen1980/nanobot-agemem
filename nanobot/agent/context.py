"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from nanobot.agent.agemem.retriever import MemoryRetriever
from nanobot.agent.agemem.store import MemoryStoreV2
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import build_assistant_message, current_time_str, detect_image_mime
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.config.schema import MemoryConfig
    from collections.abc import Callable


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(self, workspace: Path, timezone: str | None = None, disabled_skills: list[str] | None = None, memory_config: "MemoryConfig | None" = None, tasktree_status_fn: "Callable[[str], str | None] | None" = None):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)
        self._memory_config = memory_config
        self._retriever: "MemoryRetriever | None" = None
        self._tasktree_status_fn = tasktree_status_fn  # callable(chat_id) -> str | None

    def _get_retriever(self) -> "MemoryRetriever":
        """Lazily create the BM25 memory retriever."""
        if self._retriever is None:
            from nanobot.agent.agemem.retriever import MemoryRetriever
            from nanobot.agent.agemem.store import MemoryStoreV2
            self._retriever = MemoryRetriever(MemoryStoreV2(self.workspace))
        return self._retriever

    def get_memory_context_for_query(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant memories for a query using BM25 scoring.

        Returns a formatted memory context string for injection into the system prompt.
        This enables query-specific memory retrieval rather than full MEMORY.md injection.
        Only active when memory_config.enabled != False.
        """
        # Skip if AgeMem memory system is disabled
        if self._memory_config is not None and self._memory_config.enabled is False:
            return ""
        retriever = self._get_retriever()
        scored = retriever.retrieve(query, top_k=top_k)
        if not scored:
            return ""

        lines = []
        for se in scored:
            e = se.entry
            ts = e.created_at[:16] if e.created_at else ""
            src = "来源: nanobot记忆"
            tags = f"[tags={', '.join(e.tags)}]" if e.tags else ""
            freshness = f" {se.freshness_label}" if se.freshness_label else ""
            lines.append(f"- {src} {ts}{freshness} {tags} {e.content[:150]}{'...' if len(e.content) > 150 else ''}")

        return (
            "[记忆参考]\n"
            + "\n".join(lines)
            + "\n[/记忆参考]"
        )

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        memory_context: str | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills.

        Args:
            skill_names: skill names to include.
            channel: chat channel for identity.
            memory_context: optional pre-retrieved memory context (BM25 selected).
                           If None, falls back to legacy MEMORY.md injection.
        """
        parts = [self._get_identity(channel=channel)]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        if memory_context:
            parts.append(f"# 记忆参考\n\n{memory_context}")
        else:
            # Legacy: full MEMORY.md injection (backward compat)
            memory_content = self.memory.read_memory()
            if memory_content and not self._is_template_content(memory_content, "memory/MEMORY.md"):
                # Extract most recent [YYYY-MM-DD] or [YYYY-MM-DD HH:MM] from content
                latest_ts = self._extract_latest_timestamp(memory_content)
                if latest_ts:
                    mtime_ts = f"\n\n> MEMORY.md 最新记录: {latest_ts}"
                else:
                    mtime_ts = ""
                parts.append(f"# 记忆参考{mtime_ts}\n\n{memory_content}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        entries = self.memory.read_unprocessed_history(since_cursor=self.memory.get_last_dream_cursor())
        if entries:
            capped = entries[-self._MAX_RECENT_HISTORY:]
            parts.append("# 历史记录参考\n\n" + "\n".join(
                f"- [{e['timestamp']}] {e['content']}" for e in capped
            ))

        return "\n\n---\n\n".join(parts)

    def _get_identity(self, channel: str | None = None) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    def _build_runtime_context(
        self,
        channel: str | None, chat_id: str | None, timezone: str | None = None,
        session_summary: str | None = None,
    ) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        # TaskTree status from the optional status function (set by gateway)
        if self._tasktree_status_fn and chat_id:
            tt_status = self._tasktree_status_fn(chat_id)
            if tt_status:
                lines += ["", "[TaskTree Status]", tt_status]
        if session_summary:
            lines += ["", "[Resumed Session]", session_summary]
        return self._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + self._RUNTIME_CONTEXT_END

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _extract_latest_timestamp(content: str) -> str | None:
        """Extract the most recent date from MEMORY.md content.

        Supports both bracketed entries [YYYY-MM-DD HH:MM] and markdown headings
        #### YYYY-MM-DD so the LLM knows how stale the content is.
        """
        import re
        # Match [YYYY-MM-DD HH:MM] or #### YYYY-MM-DD
        timestamps = re.findall(r"(?:\[|#### )(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)", content)
        if not timestamps:
            return None
        timestamps.sort(reverse=True)
        return timestamps[0]

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        try:
            tpl = pkg_files("nanobot") / "templates" / template_path
            if tpl.is_file():
                return content.strip() == tpl.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return False

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        session_summary: str | None = None,
        memory_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call.

        Args:
            memory_context: optional BM25-retrieved memory context. If provided,
                           overrides the legacy MEMORY.md injection in system prompt.
        """
        runtime_ctx = self._build_runtime_context(channel, chat_id, self.timezone, session_summary=session_summary)
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content
        messages = [
            {"role": "system", "content": self.build_system_prompt(skill_names, channel=channel, memory_context=memory_context)},
            *history,
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: Any,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
