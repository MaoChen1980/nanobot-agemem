"""TaskTreeService: independent service that runs a Scheduler and publishes results via bus."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tasktree import Scheduler
from nanobot.agent.tasktree.callbacks import SessionPersistenceCallback
from nanobot.agent.tasktree.execution import (
    DefaultConstraintAgent,
    DefaultExecutionAgent,
    LLMSubgoalParser,
    LLMVerificationAgent,
)
from nanobot.agent.tasktree.memory_callback import MemoryCallback
from nanobot.agent.tasktree.models import FailureReport, NodeResult, TaskNode
from nanobot.agent.tasktree.scheduler import SchedulerCallbacks
from nanobot.agent.tasktree.tree import TaskTree
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider

if TYPE_CHECKING:
    from nanobot.agent.agemem.retriever import MemoryRetriever
    from nanobot.agent.agemem.store import MemoryStoreV2
    from nanobot.agent.context import ContextBuilder
    from nanobot.session.manager import SessionManager


class TaskTreeService:
    """Independent service that runs TaskTree Scheduler.

    Supports:
    - Background task execution with bus progress notifications
    - /taskstatus: query current progress
    - /taskcancel: cancel a running task
    - User input requests: TaskTree can pause and ask user a question
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        tools: Any,  # ToolRegistry
        context_builder: ContextBuilder,
        session_manager: SessionManager,
        memory_store: MemoryStoreV2,
        memory_retriever: MemoryRetriever,
        model: str | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.context_builder = context_builder
        self.session_manager = session_manager
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        self.model = model or provider.get_default_model()

        # chat_id → running task
        self._tasks: dict[str, asyncio.Task] = {}
        # chat_id → Scheduler (for status queries)
        self._schedulers: dict[str, Scheduler] = {}
        # chat_id → asyncio.Event (for user input requests)
        self._input_events: dict[str, asyncio.Event] = {}
        # chat_id → user input result
        self._input_results: dict[str, str] = {}
        # chat_id → asyncio.Event (for pending task confirmations)
        self._pending_confirms: dict[str, asyncio.Event] = {}

    async def start(self) -> None:
        """Start the service."""
        logger.info("TaskTreeService started")

    async def stop(self) -> None:
        """Stop all running tasks."""
        for task in list(self._tasks.values()):
            task.cancel()
        logger.info("TaskTreeService stopped")

    # -------------------------------------------------------------------------
    # Public API (called by AgentLoop / command handlers)
    # -------------------------------------------------------------------------

    async def submit(self, inbound: InboundMessage) -> None:
        """Submit a task for execution. Non-blocking — runs in background.

        Checks for a saved tree checkpoint and resumes if found.
        """
        chat_id = inbound.chat_id
        # Cancel any existing task for this chat_id
        if chat_id in self._tasks:
            await self.cancel(chat_id)

        session_key = f"tasktree:{chat_id}"
        session_cb = SessionPersistenceCallback(
            session_manager=self.session_manager,
            session_key=session_key,
        )

        # Try to resume from checkpoint
        saved_tree = session_cb.load_checkpoint()
        if saved_tree is not None:
            logger.info("TaskTree: resuming from checkpoint for chat_id={}", chat_id)
        else:
            logger.info("TaskTree: starting fresh for chat_id={}", chat_id)

        async def _run_and_publish() -> None:
            """Wrapper: runs _run and publishes the response to the bus."""
            result = await self._run(inbound, session_cb, saved_tree)
            if result is not None:
                await self.bus.publish_outbound(result)

        task = asyncio.create_task(_run_and_publish())
        self._tasks[chat_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(chat_id, None))

    async def cancel(self, chat_id: str) -> bool:
        """Cancel the running task for chat_id. Returns True if a task was cancelled."""
        task = self._tasks.get(chat_id)
        if task is None:
            return False
        task.cancel()
        self._tasks.pop(chat_id, None)
        self._schedulers.pop(chat_id, None)
        # Clean up any pending input/confirm events
        self._input_events.pop(chat_id, None)
        self._pending_confirms.pop(chat_id, None)
        # Notify cancellation
        await self.bus.publish_outbound(OutboundMessage(
            channel="cli",
            chat_id=chat_id,
            content="🛑 TaskTree 任务已取消",
            metadata={"_tasktree_cancelled": True},
        ))
        logger.info("TaskTree cancelled for chat_id={}", chat_id)
        return True

    async def get_status(self, chat_id: str) -> str:
        """Return a human-readable status string for /taskstatus."""
        if chat_id in self._tasks and not self._tasks[chat_id].done():
            scheduler = self._schedulers.get(chat_id)
            if scheduler is not None:
                tree = scheduler.get_tree()
                status = self._format_tree_status(tree)
                return status
            return f"🔄 TaskTree 任务运行中 (chat_id={chat_id})"

        session_key = f"tasktree:{chat_id}"
        try:
            session = self.session_manager.get_or_create(session_key)
        except Exception:
            return f"没有找到 TaskTree 任务 (chat_id={chat_id})"

        tt_msgs = [
            m for m in session.messages
            if isinstance(m, dict) and "[TaskTree" in m.get("content", "")
        ]
        if not tt_msgs:
            return f"没有找到 TaskTree 任务 (chat_id={chat_id})"

        lines = ["📋 TaskTree 进度:"]
        for m in tt_msgs[-10:]:
            content = m.get("content", "")
            if isinstance(content, str):
                lines.append(f"  {content}")
        return "\n".join(lines)

    async def confirm_task(self, chat_id: str, raw_goal: str, channel: str, auto_confirm: bool = False) -> str | None:
        """Ask the user to confirm/enrich a task goal via LLM paraphrase.

        Returns the confirmed goal string, or None if the user cancelled.

        Flow:
          1. Ask LLM to paraphrase the goal into a structured description
          2. Publish the paraphrase to the user via bus
          3. Wait for user confirmation via _pending_confirms[chat_id]
          4. Return confirmed goal or None if user said cancel
        """
        # Step 1: LLM paraphrase
        paraphrase_prompt = f"""The user wants to accomplish this task:

{raw_goal}

Your job is to:
1. Re-read and understand the task
2. Rephrase it clearly and precisely in a structured format

Output your paraphrased version in this format:
[Task Summary]
<one sentence summary of the goal>

[Detailed Description]
<detailed description of what success looks like, including any constraints or requirements mentioned or implied by the goal>

Do NOT ask questions. Do NOT add new requirements. Only rephrase and clarify what was already stated.
If the goal is already clear and specific, simply summarize it concisely."""

        try:
            messages = [{"role": "user", "content": paraphrase_prompt}]
            response = await self.provider.chat(
                messages=messages,
                model=self.model,
                max_tokens=512,
            )
            paraphrase = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning("Failed to paraphrase goal: {}", e)
            paraphrase = f"[Task Summary]\n{raw_goal}\n\n[Detailed Description]\n(LLM paraphrase unavailable)"

        # Auto-confirm mode (non-interactive): skip confirmation prompt
        if auto_confirm:
            logger.info("TaskTree auto-confirm enabled, skipping user confirmation")
            return paraphrase

        # Step 2: Publish paraphrase and ask for confirmation
        confirm_msg = (
            "📋 任务确认\n\n"
            f"{paraphrase}\n\n"
            "请回复：\n"
            "  - 直接回车确认任务\n"
            "  - 或输入补充信息/修改\n"
            "  - 或输入 /cancel 取消任务"
        )
        event = asyncio.Event()
        self._pending_confirms[chat_id] = event
        try:
            await self.bus.publish_outbound(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=confirm_msg,
                metadata={"_tasktree_needs_input": True, "_task_id": chat_id},
            ))

            # Step 3: Wait for user response
            await event.wait()
            user_response = self._input_results.pop(chat_id, "").strip()
        finally:
            self._pending_confirms.pop(chat_id, None)

        # Step 4: Handle response
        if user_response.lower() in ("/cancel", "cancel", "取消"):
            return None
        if not user_response or user_response == "/confirm":
            # User confirmed — use original paraphrase as the goal
            return paraphrase
        # User modified — use their version
        return user_response

    def submit_user_input(self, chat_id: str, response: str) -> None:
        """Called by AgentLoop when user responds to a pending input request."""
        self._input_results[chat_id] = response
        # Check _input_events (for request_user_input) and _pending_confirms (for confirm_task)
        event = self._input_events.pop(chat_id, None)
        if event is None:
            event = self._pending_confirms.pop(chat_id, None)
        if event is not None:
            event.set()

    async def request_user_input(self, chat_id: str, question: str) -> asyncio.Event:
        """Called by ExecutionAgent when it needs user input.

        Returns an Event that will be set when the user responds.
        The caller should await event.wait() after calling this.
        """
        event = asyncio.Event()
        self._input_events[chat_id] = event
        # Publish the question to the user
        await self.bus.publish_outbound(OutboundMessage(
            channel="cli",
            chat_id=chat_id,
            content="🤔 需要你的介入：" + question,
            metadata={"_tasktree_needs_input": True, "_task_id": chat_id},
        ))
        return event

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    async def _run(
        self,
        inbound: InboundMessage,
        session_cb: SessionPersistenceCallback,
        saved_tree: TaskTree | None,
    ) -> OutboundMessage | None:
        """Run a single task to completion and return the response.

        Does NOT publish to bus — caller is responsible for handling the response.
        """
        task_goal = inbound.content
        session_key = f"tasktree:{inbound.chat_id}"
        logger.info("TaskTreeService: starting task for chat_id={}", inbound.chat_id)

        # Pre-execution confirmation (for new tasks, not resume)
        if saved_tree is None:
            confirmed_goal = await self.confirm_task(
                inbound.chat_id, task_goal, inbound.channel,
                auto_confirm=inbound.metadata.get("_tasktree_auto_confirm", False),
            )
            if confirmed_goal is None:
                # User cancelled
                return OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content="任务已取消",
                    metadata=inbound.metadata,
                )
            task_goal = confirmed_goal

        scheduler = self._build_scheduler(session_key, inbound, session_cb)
        self._schedulers[inbound.chat_id] = scheduler

        try:
            root_result = await scheduler.run(task_goal, resume_tree=saved_tree)
            return self._build_response(root_result, inbound)
        except asyncio.CancelledError:
            logger.info("TaskTree cancelled mid-execution for chat_id={}", inbound.chat_id)
            return None
        except Exception as e:
            logger.exception("TaskTreeService error for chat_id={}", inbound.chat_id)
            return OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=f"TaskTree 执行错误: {e}",
                metadata=inbound.metadata,
            )
        finally:
            self._schedulers.pop(inbound.chat_id, None)

        logger.info("TaskTreeService: task complete for chat_id={}", inbound.chat_id)

    def _build_scheduler(
        self,
        session_key: str,
        inbound: InboundMessage,
        session_cb: SessionPersistenceCallback,
    ) -> Scheduler:
        """Build a Scheduler with all dependencies."""
        execution_agent = DefaultExecutionAgent(
            provider=self.provider,
            tools=self.tools,
            context_builder=self.context_builder,
            session_manager=self.session_manager,
        )
        constraint_agent = DefaultConstraintAgent(
            provider=self.provider,
            memory_retriever=self.memory_retriever,
        )
        verification_agent = LLMVerificationAgent(
            provider=self.provider,
            model=self.model,
        )
        memory_cb = MemoryCallback(memory_store=self.memory_store)

        bus = self.bus
        chat_id = inbound.chat_id
        channel = inbound.channel
        metadata = inbound.metadata
        input_events = self._input_events
        input_results = self._input_results
        service = self  # for request_user_input

        class BusNotifierCallback(SchedulerCallbacks):
            """Publishes progress events to bus + handles user input requests."""

            def _notify(self, emoji: str, node_id: str, text: str) -> None:
                asyncio.create_task(bus.publish_outbound(OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=f"{emoji} [{node_id}] {text}",
                    metadata={**(metadata or {}), "_tasktree_progress": True},
                )))

            def on_node_start(self, node: TaskNode) -> None:
                session_cb.on_node_start(node)
                self._notify("🚀", node.id, f"开始: {node.goal[:60]}")

            def on_node_done(self, node: TaskNode, result: NodeResult) -> None:
                session_cb.on_node_done(node, result)
                memory_cb.on_node_done(node, result)
                summary = (result.summary[:80] or "完成")
                self._notify("✅", node.id, summary)

            def on_node_failed(self, node: TaskNode, failure: FailureReport) -> None:
                session_cb.on_node_failed(node, failure)
                memory_cb.on_node_failed(node, failure)
                self._notify("❌", node.id, f"失败: {failure.summary[:60]}")

            def on_node_blocked(self, node: TaskNode, failure: FailureReport) -> None:
                session_cb.on_node_blocked(node, failure)
                memory_cb.on_node_blocked(node, failure)
                self._notify("🚫", node.id, f"被阻止: {failure.summary[:60]}")

        return Scheduler(
            execution_agent=execution_agent,
            constraint_agent=constraint_agent,
            subgoal_parser=LLMSubgoalParser(),
            callbacks=BusNotifierCallback(),
            verification_agent=verification_agent,
        )

    def _format_tree_status(self, tree: TaskTree) -> str:
        """Format the current tree state for /taskstatus."""
        if tree.root_id is None:
            return "TaskTree 未初始化"

        lines = ["📋 TaskTree 当前状态:"]
        for node_id, node in tree.nodes.items():
            depth = "  " * node.depth
            if node.status.value == "done":
                icon = "✅"
            elif node.status.value == "failed":
                icon = "❌"
            elif node.status.value == "blocked":
                icon = "🚫"
            elif node.status.value == "running":
                icon = "🔄"
            else:
                icon = "⏳"
            lines.append(f"  {icon} {depth}{node.id}: {node.goal[:50]}")
        return "\n".join(lines)

    def _build_response(
        self,
        root_result: NodeResult,
        inbound: InboundMessage,
    ) -> OutboundMessage:
        """Build an OutboundMessage from the root result."""
        content = root_result.summary or "Task completed."
        if root_result.artifacts:
            lines = ["[Results]"]
            for a in root_result.artifacts:
                lines.append(f"- [{a.type}] {a.path or ''}: {a.description}")
            content += "\n\n" + "\n".join(lines)
        return OutboundMessage(
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            content=content,
            metadata={**(inbound.metadata or {}), "_tasktree": True},
        )
