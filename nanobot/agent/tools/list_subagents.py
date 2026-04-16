"""ListSubagents tool for querying running subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


@tool_parameters(
    tool_parameters_schema(
        session_key=StringSchema(
            "Optional session key to filter subagents by conversation session. "
            "If omitted, returns all running subagents.",
            nullable=True,
        ),
    )
)
class ListSubagentsTool(Tool):
    """List currently running subagents with their task IDs and session info."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "list_subagents"

    @property
    def description(self) -> str:
        return (
            "List all currently running subagents. "
            "Use this before update_context to find the right task_id. "
            "Each subagent shows task_id, label, and session_key."
        )

    async def execute(self, session_key: str | None = None, **kwargs: Any) -> str:
        """List running subagents."""
        import json
        subagents = self._manager.list_subagents(session_key=session_key)
        return json.dumps({"subagents": subagents, "count": len(subagents)})
