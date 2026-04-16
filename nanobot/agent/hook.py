"""Shared lifecycle hook primitives for agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMResponse, ToolCallRequest


@dataclass(slots=True)
class AgentHookContext:
    """Mutable per-iteration state exposed to runner hooks."""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None


class AgentHook:
    """Minimal lifecycle surface for shared runner customization."""

    def __init__(self, reraise: bool = False) -> None:
        self._reraise = reraise

    def wants_streaming(self) -> bool:
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        pass

    def on_iteration_end(self, context: AgentHookContext) -> str | None:
        """Called at the end of each iteration, before the next iteration begins.

        Return a string to inject it as a user message in the next iteration.
        Override to inject user input into the agent loop.
        """
        return None

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return content


_FACTUAL_QUESTION_KEYWORDS = (
    # 问句模式（什么/是谁/在哪里/有没有 — 可能是文件里能查到的事实）
    "是什么", "在哪里", "是谁", "哪个", "有什么", "有没有",
    "是不是", "是否", "是否存在", "什么时候", "为什么",
    "写的吗", "真的吗", "来自", "来源",
    # 代码/技术实现类
    "代码", "函数", "逻辑", "实现", "原理", "源代码",
    "file:", "line ", "source code",
    # 文件/目录查询
    "文件", "目录", "内容", "查看", "这个文件", "哪个文件",
    ".md", ".json", ".yaml", ".toml", ".py", ".js",
    "memory/", "nanobot/", "skills/", "config/", ".md 是",
    # 归属/生成/创建
    "生成", "创建", "是谁生成", "谁创建", "来自哪里",
)


class SourceTracingHook(AgentHook):
    """Enforces tool use for verifiable factual questions.

    Any question that could be answered by reading a file (code, config,
    docs, memory) should be verified with tools — not answered from memory.
    If no tools were called and no source citations exist, trigger a retry
    or rewrite the response to express uncertainty.
    """

    def __init__(self, require_citations: bool = True) -> None:
        super().__init__()
        self._require_citations = require_citations
        self._triggered: bool = False

    def _is_factual_question(self, text: str) -> bool:
        if not text:
            return False
        return any(kw in text for kw in _FACTUAL_QUESTION_KEYWORDS)

    def _has_citation(self, text: str | None) -> bool:
        if not text:
            return False
        return "file:" in text or "line " in text

    def _should_retry(self, context: AgentHookContext) -> bool:
        if context.final_content is None:
            return False
        last_user_msg = ""
        for msg in reversed(context.messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if not self._is_factual_question(last_user_msg):
            return False
        return not bool(context.tool_calls) and not self._has_citation(context.final_content)

    def on_iteration_end(self, context: AgentHookContext) -> str | None:
        if self._triggered:
            return None
        if not self._should_retry(context):
            return None
        self._triggered = True
        return (
            "[答案溯源检查] 你回答了一个代码相关问题，但没有调用任何工具，也没有引用任何来源。\n"
            "请重新回答：必须先用 Read/grep 等工具查阅实际代码，"
            "答案必须包含具体的 file:line 引用。不要凭记忆回答，不要编造代码内容。"
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        if content is None:
            return None
        if self._triggered:
            self._triggered = False
            return content
        if not self._should_retry(context):
            return content
        self._triggered = False
        return (
            "[答案溯源检查] 我的回答包含未经证实的声明。\n"
            "请重新提问涉及代码的问题，我会先查源码再回答。"
        )


class CompositeHook(AgentHook):
    """Fan-out hook that delegates to an ordered list of hooks.

    Error isolation: async methods catch and log per-hook exceptions
    so a faulty custom hook cannot crash the agent loop.
    ``finalize_content`` is a pipeline (no isolation — bugs should surface).
    """

    __slots__ = ("_hooks",)

    def __init__(self, hooks: list[AgentHook]) -> None:
        super().__init__()
        self._hooks = list(hooks)

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self._hooks)

    async def _for_each_hook_safe(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            if getattr(h, "_reraise", False):
                await getattr(h, method_name)(*args, **kwargs)
                continue

            try:
                await getattr(h, method_name)(*args, **kwargs)
            except Exception:
                logger.exception("AgentHook.{} error in {}", method_name, type(h).__name__)

    async def before_iteration(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_iteration", context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._for_each_hook_safe("on_stream", context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._for_each_hook_safe("on_stream_end", context, resuming=resuming)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_execute_tools", context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._for_each_hook_safe("after_iteration", context)

    def on_iteration_end(self, context: AgentHookContext) -> str | None:
        """Return user input from the first hook that provides it."""
        for h in self._hooks:
            result = h.on_iteration_end(context)
            if result is not None:
                return result
        return None

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        for h in self._hooks:
            content = h.finalize_content(context, content)
        return content
