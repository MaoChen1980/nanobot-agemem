"""UpdateContext tool for pushing contextual updates to running subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


@tool_parameters(
    tool_parameters_schema(
        task_id=StringSchema("The subagent task id returned by spawn"),
        context=StringSchema("The updated context or new information to push to the subagent"),
        required=["task_id", "context"],
    )
)
class UpdateContextTool(Tool):
    """Push contextual updates to a running subagent."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "update_context"

    @property
    def description(self) -> str:
        return (
            "Push updated context to a running subagent. "
            "Use this when the user provides new information that should reach the subagent. "
            "The subagent receives it as a new message and can adjust its plan accordingly. "
            "Returns whether the update was successfully delivered."
        )

    async def execute(self, task_id: str, context: str, **kwargs: Any) -> str:
        """Push context update to a subagent."""
        import json
        delivered = await self._manager.update_context(task_id, context)
        return json.dumps({"task_id": task_id, "delivered": delivered})
