"""TaskTree tools for LLM-driven hierarchical task management."""

from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema


@tool_parameters(
    tool_parameters_schema(
        goal=StringSchema("The goal or task description for TaskTree to plan and execute"),
        required=["goal"],
    )
)
class TaskplanTool(Tool):
    """Submit a goal to TaskTree for background hierarchical execution."""

    def __init__(self, tasktree_service: Any):
        self._service = tasktree_service

    @property
    def name(self) -> str:
        return "taskplan"

    @property
    def description(self) -> str:
        return (
            "Submit a goal to TaskTree for background hierarchical planning and execution. "
            "Use this for complex multi-step tasks that benefit from automatic decomposition. "
            "The task runs in background and reports progress. "
            "Check status with taskstatus(), cancel with taskcancel()."
        )

    async def execute(self, goal: str, **kwargs: Any) -> str:
        """Submit a TaskTree goal."""
        from nanobot.bus.events import InboundMessage

        # Use the current session's chat_id so the command handler and LLM tool
        # operate on the same TaskTree session.
        session_key = kwargs.get("session_key", "sdk:direct")
        # Extract channel and chat_id from session_key (e.g. "feishu:ou_xxx" -> channel="feishu", chat_id="ou_xxx")
        parts = session_key.split(":", 1)
        channel = parts[0] if len(parts) > 1 else "cli"
        chat_id = parts[-1]
        inbound = InboundMessage(
            channel=channel,
            sender_id="llm",
            chat_id=chat_id,
            content=goal,
            media=[],
        )
        # Check if service has the task running already
        running = self._service._tasks.get(chat_id)
        if running and not running.done():
            return "A TaskTree task is already running for this session. Use taskstatus() to check progress or taskcancel() to cancel it."

        await self._service.submit(inbound)
        return f"TaskTree started for goal: {goal[:80]}{'...' if len(goal) > 80 else ''}. Use taskstatus() to check progress."


@tool_parameters(tool_parameters_schema())
class TaskStatusTool(Tool):
    """Get the current status of the running TaskTree task."""

    def __init__(self, tasktree_service: Any):
        self._service = tasktree_service

    @property
    def name(self) -> str:
        return "taskstatus"

    @property
    def description(self) -> str:
        return (
            "Check the current status of the running TaskTree task. "
            "Returns progress, completed steps, and any pending user questions. "
            "Use this to monitor background TaskTree execution."
        )

    async def execute(self, **kwargs: Any) -> str:
        """Get TaskTree status."""
        session_key = kwargs.get("session_key", "sdk:direct")
        chat_id = session_key.split(":")[-1]
        status = await self._service.get_status(chat_id)
        return status


@tool_parameters(tool_parameters_schema())
class TaskCancelTool(Tool):
    """Cancel the currently running TaskTree task."""

    def __init__(self, tasktree_service: Any):
        self._service = tasktree_service

    @property
    def name(self) -> str:
        return "taskcancel"

    @property
    def description(self) -> str:
        return (
            "Cancel the currently running TaskTree task. "
            "Use this when the task is taking too long, the goal has changed, "
            "or you need to start a different task instead."
        )

    async def execute(self, **kwargs: Any) -> str:
        """Cancel TaskTree task."""
        session_key = kwargs.get("session_key", "sdk:direct")
        chat_id = session_key.split(":")[-1]
        cancelled = await self._service.cancel(chat_id)
        if cancelled:
            return "TaskTree task cancelled."
        return "No running TaskTree task found."
