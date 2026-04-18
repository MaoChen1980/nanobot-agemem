"""High-level programmatic interface to nanobot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tasktree.service import TaskTreeService
from nanobot.bus.router import RouterBus


@dataclass(slots=True)
class RunResult:
    """Result of a single agent run."""

    content: str
    tools_used: list[str]
    messages: list[dict[str, Any]]


class Nanobot:
    """Programmatic facade for running the nanobot agent.

    Usage::

        bot = Nanobot.from_config()
        result = await bot.run("Summarize this repo", hooks=[MyHook()])
        print(result.content)
    """

    def __init__(self, loop: AgentLoop, router: RouterBus, tasktree_service: TaskTreeService | None = None, tasktree_queue: asyncio.Queue | None = None) -> None:
        self._loop = loop
        self._router = router
        self._tasktree = tasktree_service
        self._tasktree_queue = tasktree_queue
        self._tasktree_consumer_task: asyncio.Task | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> Nanobot:
        """Create a Nanobot instance from a config file.

        Uses RouterBus for message routing between AgentLoop and TaskTreeService.
        """
        from nanobot.agent.agemem.retriever import MemoryRetriever
        from nanobot.agent.agemem.store import MemoryStoreV2
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.tasktree.execution import (
            DefaultConstraintAgent,
            DefaultExecutionAgent,
            LLMSubgoalParser,
            LLMVerificationAgent,
        )
        from nanobot.config.loader import load_config, resolve_config_env_vars
        from nanobot.config.schema import Config

        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = resolve_config_env_vars(load_config(resolved))
        if workspace is not None:
            config.agents.defaults.workspace = str(
                Path(workspace).expanduser().resolve()
            )

        provider = _make_provider(config)
        router = RouterBus()
        defaults = config.agents.defaults
        workspace_path = config.workspace_path

        # Build shared components needed by both AgentLoop and TaskTreeService
        memory_store = MemoryStoreV2(workspace_path)
        memory_retriever = MemoryRetriever(memory_store)
        from nanobot.agent.agemem.causal_store import CausalStore
        causal_store = CausalStore(workspace_path)
        context_builder = ContextBuilder(
            workspace_path,
            timezone=defaults.timezone,
            disabled_skills=defaults.disabled_skills,
            memory_config=defaults.memory,
        )
        from nanobot.session.manager import SessionManager
        session_manager = SessionManager(workspace_path)

        # Build TaskTreeService
        from nanobot.agent.tools.registry import ToolRegistry
        tools = ToolRegistry()

        tasktree_service = TaskTreeService(
            bus=router,
            provider=provider,
            tools=tools,
            context_builder=context_builder,
            session_manager=session_manager,
            memory_store=memory_store,
            memory_retriever=memory_retriever,
            model=defaults.model,
            causal_store=causal_store,
        )

        # Register TaskTreeService route on RouterBus
        def _tasktree_predicate(msg: Any) -> bool:
            content = getattr(msg, "content", "") or ""
            metadata = getattr(msg, "metadata", {}) or {}
            return (
                content.strip().startswith("/taskplan")
                or bool(metadata.get("_tasktree_task"))
            )

        tasktree_queue = router.register_route("tasktree", _tasktree_predicate)

        # Start TaskTreeService consumer loop — it reads from the tasktree queue
        # and calls submit() for each matched message.
        async def _tasktree_consumer() -> None:
            while True:
                try:
                    msg = await tasktree_queue.get()
                    await tasktree_service.submit(msg)
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("tasktree consumer error")


        # AgentLoop receives everything else (non-TaskTree messages)
        loop = AgentLoop(
            bus=router,
            provider=provider,
            workspace=workspace_path,
            model=defaults.model,
            max_iterations=defaults.max_tool_iterations,
            context_window_tokens=defaults.context_window_tokens,
            context_block_limit=defaults.context_block_limit,
            max_tool_result_chars=defaults.max_tool_result_chars,
            provider_retry_mode=defaults.provider_retry_mode,
            web_config=config.tools.web,
            exec_config=config.tools.exec,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            timezone=defaults.timezone,
            unified_session=defaults.unified_session,
            disabled_skills=defaults.disabled_skills,
            session_ttl_minutes=defaults.session_ttl_minutes,
            memory_config=defaults.memory,
            session_manager=session_manager,
            tasktree_service=tasktree_service,
        )
        return cls(loop, router, tasktree_service, tasktree_queue)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        hooks: list[AgentHook] | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            hooks: Optional lifecycle hooks for this run.
        """
        # Start the TaskTree consumer task once (idempotent — no-op if already started)
        if self._tasktree_consumer_task is None:
            await self._router.start_router()

            async def _tasktree_consumer() -> None:
                while True:
                    try:
                        msg = await self._tasktree_queue.get()
                        await self._tasktree.submit(msg)
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        logger.exception("tasktree consumer error")

            self._tasktree_consumer_task = asyncio.create_task(_tasktree_consumer())

        prev = self._loop._extra_hooks
        if hooks is not None:
            self._loop._extra_hooks = list(hooks)
        try:
            response = await self._loop.process_direct(
                message, session_key=session_key,
            )
        finally:
            self._loop._extra_hooks = prev

        content = (response.content if response else None) or ""
        return RunResult(content=content, tools_used=[], messages=[])

    # -------------------------------------------------------------------------
    # TaskTree API (SDK-level access to TaskTreeService)
    # -------------------------------------------------------------------------

    async def run_tasktree(
        self,
        message: str,
        session_key: str = "sdk:default",
    ) -> None:
        """Submit a TaskTree task for background execution.

        Does not wait for completion. Use get_task_status() to check progress.
        """
        from nanobot.bus.events import InboundMessage
        inbound = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id=session_key,
            content=message,
            media=[],
        )
        await self._router.start_router()
        await self._tasktree.start()
        await self._tasktree.submit(inbound)

    async def get_task_status(self, session_key: str = "sdk:default") -> str:
        """Return current TaskTree progress as a string."""
        return await self._tasktree.get_status(session_key)

    async def cancel_task(self, session_key: str = "sdk:default") -> bool:
        """Cancel the running TaskTree task for session_key. Returns True if cancelled."""
        return await self._tasktree.cancel(session_key)


def _make_provider(config: Any) -> Any:
    """Create the LLM provider from config (extracted from CLI)."""
    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            raise ValueError("Azure OpenAI requires api_key and api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "github_copilot":
        from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key, api_base=p.api_base, default_model=model
        )
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider
