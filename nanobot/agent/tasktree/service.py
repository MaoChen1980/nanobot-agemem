"""TaskTreeService: independent service that runs a Scheduler and publishes results via bus."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tasktree import Scheduler
from nanobot.agent.tasktree.callbacks import SessionPersistenceCallback
from nanobot.agent.tasktree.execution import (
    DefaultConstraintAgent,
    DefaultConstraintAgentConfig,
    DefaultExecutionAgent,
    LLMSubgoalParser,
    LLMVerificationAgent,
)
from nanobot.agent.tasktree.memory_callback import MemoryCallback
from nanobot.agent.tasktree.models import FailureReport, NodeResult, TaskNode
from nanobot.agent.tasktree.scheduler import SchedulerCallbacks, SchedulerConfig
from nanobot.agent.tasktree.tree import TaskTree
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain

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
        # chat_id → channel (for routing outbound messages to correct channel)
        self._channels: dict[str, str] = {}

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
        Rejects new submissions if a task is already running.
        """
        chat_id = inbound.chat_id

        # Reject only if a task is truly RUNNING (not done) for this chat_id.
        # A completed task allows new submissions — the old checkpoint will be
        # overwritten naturally by the new task.
        running_task = self._tasks.get(chat_id)
        if running_task is not None and not running_task.done():
            scheduler = self._schedulers.get(chat_id)
            busy_goal = ""
            if scheduler is not None:
                tree = scheduler.get_tree()
                if tree and tree.root_id:
                    busy_goal = tree.nodes[tree.root_id].goal
                    if len(busy_goal) > 60:
                        busy_goal = busy_goal[:57] + "..."
            busy_msg = (
                f"⚠️ TaskTree 正忙，无法接收新任务。\n"
                f"当前任务：{busy_goal or '(未知)'}\n"
                f"请等待当前任务结束，或发送 /taskcancel 取消后再试。"
            )
            await self.bus.publish_outbound(OutboundMessage(
                channel=inbound.channel,
                chat_id=chat_id,
                content=busy_msg,
                metadata={"_tasktree_busy": True},
            ))
            logger.info("TaskTree: rejected new submit, chat_id={} busy with {}", chat_id, busy_goal[:40])
            return

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
        # Get channel BEFORE cleaning up _channels dict
        channel = self._channels.get(chat_id, "cli")
        task.cancel()
        self._tasks.pop(chat_id, None)
        self._schedulers.pop(chat_id, None)
        self._channels.pop(chat_id, None)
        # Clear checkpoint and session messages so the next submit starts fresh
        session_key = f"tasktree:{chat_id}"
        try:
            session_cb = SessionPersistenceCallback(
                session_manager=self.session_manager,
                session_key=session_key,
            )
            session_cb.clear_session()
        except Exception:
            pass
        # Notify cancellation on the correct channel
        await self.bus.publish_outbound(OutboundMessage(
            channel=channel,
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

        # No active task — check the saved tree checkpoint
        session_key = f"tasktree:{chat_id}"
        session_cb = SessionPersistenceCallback(
            session_manager=self.session_manager,
            session_key=session_key,
        )
        saved_tree = session_cb.load_checkpoint()
        if saved_tree is not None:
            return self._format_tree_status(saved_tree)

        return f"没有找到 TaskTree 任务 (chat_id={chat_id})"

    def get_status_str(self, chat_id: str) -> str | None:
        """Return a one-line status string for ContextBuilder injection, or None if idle.

        This is a sync method safe to call from the agent context hot path.
        """
        task = self._tasks.get(chat_id)
        if task is None:
            return None
        if task.done():
            return None

        scheduler = self._schedulers.get(chat_id)
        if scheduler is None:
            return None
        tree = scheduler.get_tree()
        if tree is None or tree.root_id is None:
            return None
        root = tree.nodes.get(tree.root_id)
        if root is None:
            return None
        goal = root.goal.replace("\n", " ").strip()
        if len(goal) > 50:
            goal = goal[:47] + "..."
        return f"🔄 TaskTree 进行中: {goal}"

    def submit_user_input(self, chat_id: str, response: str) -> None:
        """No-op. Kept for CLI compatibility."""
        pass

    async def request_user_input(self, chat_id: str, question: str, channel: str) -> asyncio.Event:
        """No-op. User input goes through //taskinfo command."""
        return asyncio.Event()

    def find_wait_info_node(self, chat_id: str) -> tuple[str, str] | None:
        """Find the WAIT_INFO node for chat_id.

        Returns (node_id, question) if found, None otherwise.
        """
        scheduler = self._schedulers.get(chat_id)
        if scheduler is None:
            return None
        tree = scheduler.get_tree()
        if tree is None:
            return None
        for nid, node in tree.nodes.items():
            if node.status.value == "wait_info":
                question = ""
                if node.result and node.result.user_input_question:
                    question = node.result.user_input_question
                return (nid, question)
        return None

    async def _submit_user_input_and_resume(self, chat_id: str, user_input: str) -> OutboundMessage | None:
        """Submit user input for a WAIT_INFO node and resume the task.

        Called by //taskinfo command handler.
        Returns None if no WAIT_INFO node found.
        """
        wait_info = self.find_wait_info_node(chat_id)
        if wait_info is None:
            return None

        node_id, question = wait_info
        scheduler = self._schedulers.get(chat_id)
        if scheduler is None:
            return None

        tree = scheduler.get_tree()
        node = tree.nodes.get(node_id)
        if node is None:
            return None

        # Fill in the user input
        if node.result is None:
            from nanobot.agent.tasktree.models import NodeResult, TaskStatus
            node.result = NodeResult(node_id=node_id, status=TaskStatus.WAIT_INFO)
        node.result.user_input_answer = user_input
        # Clear user_input_question so re-execution doesn't trigger WAIT_INFO again
        node.result.user_input_question = None
        # Reset status to PENDING so scheduler executes it (not skips as WAIT_INFO)
        node.status = node.status.__class__.PENDING

        # Clear waiting question
        self._waiting_questions.pop(chat_id, None)

        logger.info("TaskTreeService: resuming with user input for chat_id={}, node={}", chat_id, node_id)

        # Resume the scheduler
        try:
            root_result = await scheduler.run(scheduler._root_goal, resume_tree=tree)
            tree = scheduler.get_tree()

            # If still WAIT_INFO, don't complete
            if root_result is None:
                logger.info("TaskTreeService: still WAIT_INFO after resume, chat_id={}", chat_id)
                return None

            return self._build_response(root_result, tree, None)
        except asyncio.CancelledError:
            logger.info("TaskTreeService: cancelled during resume for chat_id={}", chat_id)
            return None
        except Exception as e:
            logger.exception("TaskTreeService error during resume for chat_id={}", chat_id)
            return OutboundMessage(
                channel=self._channels.get(chat_id, "cli"),
                chat_id=chat_id,
                content=f"TaskTree 执行错误: {e}",
                metadata={},
            )
        finally:
            # Clean up if task is done
            root_node = None
            if self._schedulers.get(chat_id) is not None:
                t = self._schedulers[chat_id].get_tree()
                if t and t.root_id:
                    root_node = t.nodes.get(t.root_id)
            is_wait_info = root_node is not None and root_node.status.value == "wait_info"
            if not is_wait_info:
                self._schedulers.pop(chat_id, None)
                self._channels.pop(chat_id, None)

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

        Returns None when:
        - Task was cancelled
        - WAIT_INFO reached (task remains in _schedulers for //taskinfo to resume)
        """
        task_goal = inbound.content
        session_key = f"tasktree:{inbound.chat_id}"
        self._channels[inbound.chat_id] = inbound.channel
        logger.info("TaskTreeService: starting task for chat_id={}", inbound.chat_id)

        scheduler = self._build_scheduler(session_key, inbound, session_cb)
        self._schedulers[inbound.chat_id] = scheduler

        try:
            root_result = await scheduler.run(task_goal, resume_tree=saved_tree)
            tree = scheduler.get_tree()

            # None result means WAIT_INFO — don't complete the task yet
            # Scheduler remains in _schedulers for //taskinfo to resume
            if root_result is None:
                logger.info("TaskTreeService: WAIT_INFO, awaiting //taskinfo for chat_id={}", inbound.chat_id)
                # Find the WAIT_INFO node and notify the user
                wait_info = self.find_wait_info_node(inbound.chat_id)
                if wait_info:
                    node_id, question = wait_info
                    if question:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=inbound.channel,
                            chat_id=inbound.chat_id,
                            content=f"⏸️ [{node_id}] 需要信息:\n{question}\n\n请回复 //taskinfo <你的回答>",
                            metadata={"_tasktree_wait_info": True},
                        ))
                return None

            return self._build_response(root_result, tree, inbound)
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
            # Only clean up if task is truly done (not WAIT_INFO)
            # WAIT_INFO leaves scheduler in _schedulers for //taskinfo
            root_node = None
            if self._schedulers.get(inbound.chat_id) is not None:
                tree = self._schedulers[inbound.chat_id].get_tree()
                if tree and tree.root_id:
                    root_node = tree.nodes.get(tree.root_id)
            is_wait_info = root_node is not None and root_node.status.value == "wait_info"
            if not is_wait_info:
                self._schedulers.pop(inbound.chat_id, None)
                self._channels.pop(inbound.chat_id, None)

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
            tasktree_service=self,
        )
        constraint_agent = DefaultConstraintAgent(
            provider=self.provider,
            memory_retriever=self.memory_retriever,
            config=DefaultConstraintAgentConfig(max_depth=5),  # max tree depth: 5
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
        class BusNotifierCallback(SchedulerCallbacks):
            """Publishes progress events to session and memory only.

            No direct bus messages during normal execution — the final result
            is sent via _build_response when the scheduler finishes.
            Only WAIT_INFO nodes send a bus notification to prompt the user.
            """

            def _notify(self, emoji: str, node_id: str, text: str) -> None:
                asyncio.create_task(bus.publish_outbound(OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=f"{emoji} [{node_id}] {text}",
                    metadata={**(metadata or {}), "_tasktree_progress": True},
                )))

            def on_node_start(self, node: TaskNode) -> None:
                session_cb.on_node_start(node)

            def on_node_done(self, node: TaskNode, result: NodeResult) -> None:
                session_cb.on_node_done(node, result)
                memory_cb.on_node_done(node, result)

            def on_node_failed(self, node: TaskNode, failure: FailureReport) -> None:
                session_cb.on_node_failed(node, failure)
                memory_cb.on_node_failed(node, failure)

            def on_node_blocked(self, node: TaskNode, failure: FailureReport) -> None:
                session_cb.on_node_blocked(node, failure)
                memory_cb.on_node_blocked(node, failure)

        return Scheduler(
            config=SchedulerConfig(max_planning_children=5, max_depth=5),  # cap at 5 planning children, depth 5
            execution_agent=execution_agent,
            constraint_agent=constraint_agent,
            subgoal_parser=LLMSubgoalParser(),
            callbacks=BusNotifierCallback(),
            verification_agent=verification_agent,
            provider=self.provider,
        )

    def _format_tree_status(self, tree: TaskTree) -> str:
        """Format a concise tree state for /taskstatus — shows root + depth-1 nodes only."""
        if tree.root_id is None:
            return "📋 TaskTree 当前状态:\n  (空树)"

        root = tree.nodes[tree.root_id]

        def node_icon(node) -> str:
            if node.status.value == "done":
                return "✅"
            elif node.status.value == "failed":
                return "❌"
            elif node.status.value == "blocked":
                return "🚫"
            elif node.status.value == "running":
                return "🔄"
            elif node.status.value == "wait_info":
                return "⏸️"
            elif node.status.value == "tried":
                return "🔁"
            return "⏳"

        def summary(node) -> str:
            return node.goal.replace("\n", " ").strip()

        root_icon = node_icon(root)
        lines = [f"📋 TaskTree 当前状态:\n└── {root_icon} {summary(root)}"]

        # Show depth-1 children only (not full recursive tree)
        for child_id in root.children:
            child = tree.nodes[child_id]
            icon = node_icon(child)
            lines.append(f"    ├── {icon} {summary(child)}")

        # Count pending vs done
        total = len(tree.nodes)
        done = sum(1 for n in tree.nodes.values() if n.status.value == "done")
        pending = sum(1 for n in tree.nodes.values() if n.status.value in ("pending", "running"))
        lines.append(f"\n共 {total} 步 | ✅完成 {done} | ⏳进行中 {pending}")

        return "\n".join(lines)

    def _render_tree_diagram(
        self,
        tree: TaskTree,
        title: str = "📋 TaskTree 执行结果:",
    ) -> str:
        """Render the full tree as an ASCII diagram with status per node.

        Structure:
        📋 TaskTree 执行结果:
        └── ✅ root: 开发电商小程序 demo
            ├── ✅ node_1: 设计数据库模型
            │   └── ✅ node_1_1: 设计 User/Product/Order 模型
            └── ✅ node_2: 实现后端 API
                ├── ✅ node_2_1: 实现用户注册接口
                └── ❌ node_2_2: 实现商品列表接口 (失败原因...)
        """
        if tree.root_id is None:
            return title + "\n  (空树)"

        lines = [title]

        def node_icon(node) -> str:
            if node.status.value == "done":
                return "✅"
            elif node.status.value == "failed":
                return "❌"
            elif node.status.value == "blocked":
                return "🚫"
            elif node.status.value == "running":
                return "🔄"
            elif node.status.value == "wait_info":
                return "⏸️"
            elif node.status.value == "tried":
                return "🔁"
            return "⏳"

        def render_node(node_id: str, prefix: str, is_last: bool) -> None:
            node = tree.nodes[node_id]
            icon = node_icon(node)
            # Connector: "└──" for last child, "├──" for others
            connector = "└── " if is_last else "├── "
            # Blank prefix continuation for parent connector lines
            blank = prefix.replace("└── ", "    ").replace("├── ", "│   ")

            goal_text = node.goal.replace("\n", " ").strip()

            lines.append(f"{prefix}{connector}{icon} {node.id}: {goal_text}")

            # Failure reason
            if node.status.value == "failed" and node.failure:
                reason = node.failure.summary.replace("\n", " ").strip()
                if len(reason) > 60:
                    reason = reason[:57] + "..."
                lines.append(f"{blank}    └─ ❗ {reason}")

            # Children
            child_count = len(node.children)
            for i, child_id in enumerate(node.children):
                is_last_child = (i == child_count - 1)
                child_prefix = blank + ("    " if is_last_child else "│   ")
                render_node(child_id, child_prefix, is_last_child)

        root = tree.nodes[tree.root_id]
        root_goal_text = root.goal.replace("\n", " ").strip()
        lines.append(f"└── {node_icon(root)} {root.id}: {root_goal_text}")

        # Render root's children recursively with 4-space indent
        for i, child_id in enumerate(root.children):
            is_last = (i == len(root.children) - 1)
            render_node(child_id, "    ", is_last)

        return "\n".join(lines)

    def _build_response(
        self,
        root_result: NodeResult,
        tree: TaskTree,
        inbound: InboundMessage,
    ) -> OutboundMessage:
        """Build an OutboundMessage from the root result."""
        # Tree diagram — use concise format for all channels
        tree_diagram = self._format_tree_status(tree)

        # Summary + artifacts
        content = root_result.summary or "Task completed."
        if root_result.artifacts:
            lines = ["[Results]"]
            for a in root_result.artifacts:
                lines.append(f"- [{a.type}] {a.path or ''}: {a.description}")
            content += "\n\n" + "\n".join(lines)

        # Combine: tree diagram first, then summary
        full_content = f"{tree_diagram}\n\n{content}"

        return OutboundMessage(
            channel=inbound.channel,
            chat_id=inbound.chat_id,
            content=full_content,
            metadata={**(inbound.metadata or {}), "_tasktree": True},
        )
