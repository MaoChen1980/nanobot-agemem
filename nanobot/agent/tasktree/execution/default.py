"""DefaultExecutionAgent: wraps AgentRunner for single-node execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook
from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.tasktree.context import build_node_context, build_result_from_agent_response
from nanobot.agent.tasktree.models import (
    ConstraintSet,
    FailureReport,
    NodeResult,
    RootCause,
    TaskNode,
    TaskStatus,
    WorkspaceState,
)
from nanobot.agent.tasktree.scheduler import ExecutionAgent
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import SessionManager

if TYPE_CHECKING:
    from nanobot.agent.tasktree.tree import TaskTree


@dataclass
class DefaultExecutionAgentConfig:
    """Configuration for DefaultExecutionAgent."""

    model: str | None = None
    max_iterations: int = 20
    max_tool_result_chars: int = 4000


class DefaultExecutionAgent:
    """ExecutionAgent backed by the existing AgentRunner.

    This wraps the core nanobot ReAct loop into a single-node execution.
    Each call to execute() runs one TaskNode and returns a NodeResult or FailureReport.
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: Any,  # ToolRegistry
        context_builder: ContextBuilder,
        session_manager: SessionManager,
        config: DefaultExecutionAgentConfig | None = None,
        tasktree_service: Any = None,  # TaskTreeService, for user input during execution
    ):
        self.provider = provider
        self.tools = tools
        self.context_builder = context_builder
        self.session_manager = session_manager
        self.runner = AgentRunner(provider)
        self.config = config or DefaultExecutionAgentConfig()
        self._model = self.config.model or provider.get_default_model()
        self._tasktree_service = tasktree_service

    async def execute(
        self,
        node: TaskNode,
        constraints: ConstraintSet,
        parent_result: NodeResult | None,
        root_goal: str,
        root_goal_context: str,
        tree: TaskTree | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> NodeResult | FailureReport:
        """Execute a single TaskNode via AgentRunner.

        Builds the node's context using build_node_context(), runs it through
        AgentRunner, and returns a NodeResult with the response content.
        """
        # Build messages for this node
        if tree is None:
            # If no tree provided, construct a minimal path context
            messages = self._build_standalone_messages(
                node=node,
                constraints=constraints,
                channel=channel,
                chat_id=chat_id,
            )
        else:
            messages = build_node_context(
                context_builder=self.context_builder,
                tree=tree,
                node=node,
                parent_result=parent_result,
                constraints=constraints,
                history=None,
                channel=channel,
                chat_id=chat_id,
            )

        spec = AgentRunSpec(
            initial_messages=messages,
            tools=self.tools,
            model=self._model,
            max_iterations=self.config.max_iterations,
            max_tool_result_chars=self.config.max_tool_result_chars,
        )

        try:
            result = await self.runner.run(spec)
            token_spent = (
                result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)
            )
            tools_used = result.tools_used or []

            # Non-root nodes must use tools to perform actual work.
            # If no tools were called, the LLM just said "done" without acting —
            # treat this as a failure so the parent can REPLAN with the context.
            if node.parent_id is not None and not tools_used:
                logger.warning(
                    "ExecutionAgent: node {} ({}) used no tools. "
                    "Failing so parent can REPLAN.",
                    node.id,
                    node.goal[:50],
                )
                return FailureReport(
                    node_id=node.id,
                    status=TaskStatus.FAILED,
                    root_cause=RootCause.NO_REMAINING_OPTIONS,
                    summary="No tools were used. The agent must use tools (e.g., write_file, exec) to complete the task.",
                    constraint_veto=False,
                    workspace_state=WorkspaceState.CLEAN,
                )

            # Root node: if it claims no children (##[TASKS][]) and also used
            # no tools, the LLM just gave up without doing anything — fail it.
            # This is distinct from "empty tasks = trivial done" which would still
            # use tools (create the file directly).
            if node.parent_id is None and not tools_used:
                logger.warning(
                    "ExecutionAgent: root node {} used no tools and produced no children. "
                    "Failing so a concrete plan is generated.",
                    node.id,
                )
                return FailureReport(
                    node_id=node.id,
                    status=TaskStatus.FAILED,
                    root_cause=RootCause.NO_REMAINING_OPTIONS,
                    summary="Root node produced no children and used no tools. The agent must either decompose the goal into subtasks or execute it directly using tools.",
                    constraint_veto=False,
                    workspace_state=WorkspaceState.CLEAN,
                )

            # Extract artifacts and detect workspace state
            artifacts, ws_state = self._extract_artifacts(result.messages)

            # Parse structured agent response: detect {..., "user_input_request": "..."}
            # so the agent can explicitly ask the user a specific question.
            # Also accepts "summary" override in JSON.
            user_input_question: str | None = None
            agent_summary: str | None = None
            raw_content = result.final_content or ""
            if raw_content.strip().startswith("{"):
                try:
                    import json
                    parsed = json.loads(raw_content)
                    user_input_question = parsed.get("user_input_request") or None
                    agent_summary = parsed.get("summary")
                except Exception:
                    pass

            # If no user_input_request in JSON, fall back to detecting
            # a [USER_INPUT_REQUEST] marker in plain text.
            if not user_input_question and raw_content:
                marker = "[USER_INPUT_REQUEST]"
                if marker in raw_content:
                    idx = raw_content.index(marker)
                    end = raw_content.index("]", idx)
                    user_input_question = raw_content[idx + len(marker):end].strip()
                    raw_content = raw_content[:idx].strip()

            # Also update final_content to clean text (without marker)
            if not agent_summary and not user_input_question:
                agent_summary = raw_content

            return build_result_from_agent_response(
                node_id=node.id,
                agent_content=agent_summary or raw_content or "",
                artifacts=artifacts,
                token_spent=token_spent,
                workspace_state=ws_state,
                user_input_question=user_input_question,
            )
        except Exception as e:
            logger.exception("ExecutionAgent error for node {}", node.id)
            return build_failure_from_error(
                node_id=node.id,
                error_message=str(e),
                root_cause="unknown",
            )

    def _build_standalone_messages(
        self,
        node: TaskNode,
        constraints: ConstraintSet,
        channel: str,
        chat_id: str,
    ) -> list[dict[str, Any]]:
        """Build messages when no TaskTree is available."""
        system_prompt = self.context_builder.build_system_prompt(channel=channel)
        runtime_ctx = self.context_builder._build_runtime_context(channel, chat_id, self.context_builder.timezone)

        task_block = (
            f"[Root Goal]\n{node.goal}\n[/Root Goal]\n\n"
            f"[Constraints]\n"
            f"- max_depth: {constraints.max_depth}\n"
            f"- forbidden_actions: {', '.join(constraints.forbidden_actions)}\n"
            f"[/Constraints]\n\n"
            f"[Your Task]\n{node.goal}\n[/Your Task]"
        )

        user_content = f"{runtime_ctx}\n\n{task_block}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _extract_artifacts(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        """Extract artifacts and detect workspace state from tool calls in the message history.

        Returns:
            Tuple of (artifacts list, workspace_state string: "clean" | "partial" | "dirty")
        """
        artifacts: list[dict[str, Any]] = []
        workspace_state = "clean"

        # Tools that indicate file modifications
        WRITING_TOOLS = {"write_file", "edit_file", "create_file", "make_file"}
        # Tools that modify the environment more broadly
        DIRTY_TOOLS = {
            "bash", "shell", "run_command", "exec", "subprocess",
            "git", "npm", "pip", "brew", "apt", "yum",  # package managers
            "docker", "kubectl",  # container tools
        }

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tool_call in msg.get("tool_calls", []):
                fn = tool_call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        import json
                        args = json.loads(args)
                    except Exception:
                        args = {}

                if name in WRITING_TOOLS:
                    path = args.get("file_path") or args.get("path") or args.get("file")
                    if path:
                        artifacts.append({
                            "type": "file_written" if name in ("write_file", "create_file", "make_file") else "file_modified",
                            "path": path,
                            "description": f"Tool: {name}",
                        })
                        workspace_state = "partial"

                if name in DIRTY_TOOLS:
                    workspace_state = "dirty"

        return artifacts, workspace_state


class _UserInputInjectHook(AgentHook):
    """Hook that injects user input from TaskTreeService into the agent loop.

    When the scheduler calls request_user_input(), the user's response is stored
    in service._input_results[chat_id]. This hook checks for it at the end of
    each iteration and injects it as the next user message.
    """

    __slots__ = ("_service", "_chat_id")

    def __init__(self, service: Any, chat_id: str):
        self._service = service
        self._chat_id = chat_id

    def on_iteration_end(self, context: AgentHookContext) -> str | None:
        # Always check for user input — it may have arrived via submit_user_input
        return self._service._input_results.pop(self._chat_id, None)

    def wants_streaming(self) -> bool:
        return False


class _NoOpHook(AgentHook):
    """A no-op hook used when no external hooks are needed."""

    def wants_streaming(self) -> bool:
        return False
