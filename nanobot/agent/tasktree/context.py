"""Node context builder: assembles LLM messages for a single node execution.

Minimal-dependency principle:
- Root Goal: always present (anchor)
- Parent result: one level only (summary + artifacts)
- Parent goal: present
- Constraints: present as structured block
- Sibling info: NOT present unless explicitly needed
- Failure context: NOT present unless parent needs it to replan
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tasktree.models import (
    ConstraintSet,
    NodeResult,
    TaskNode,
)

if TYPE_CHECKING:
    from nanobot.agent.context import ContextBuilder
    from nanobot.agent.tasktree.tree import TaskTree


def build_node_context(
    context_builder: ContextBuilder,
    tree: TaskTree,
    node: TaskNode,
    parent_result: NodeResult | None,
    constraints: ConstraintSet,
    history: list[dict[str, Any]] | None = None,
    channel: str | None = None,
    chat_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build the full message list for executing a single TaskNode.

    Args:
        context_builder: existing ContextBuilder for system prompt construction
        tree: the task tree (for path lookup)
        node: the node to execute
        parent_result: result of the parent node (or None for root)
        constraints: constraint set for this node
        history: optional message history from SessionManager
        channel: channel identifier for identity template
        chat_id: chat identifier for runtime context

    Returns:
        A message list ready for AgentRunSpec.initial_messages.
    """
    # 1. Build the system prompt via ContextBuilder
    system_prompt = context_builder.build_system_prompt(channel=channel)

    # 2. Build the node task block (injected into user message)
    task_block = _build_task_block(tree, node, parent_result, constraints)

    # 3. Assemble messages
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    runtime_ctx = context_builder._build_runtime_context(channel, chat_id, context_builder.timezone)
    user_content = f"{runtime_ctx}\n\n{task_block}"

    if messages[-1].get("role") == "user":
        # Merge with last user message
        last = dict(messages[-1])
        last["content"] = f"{last.get('content', '')}\n\n{user_content}"
        messages[-1] = last
    else:
        messages.append({"role": "user", "content": user_content})

    return messages


def _build_task_block(
    tree: TaskTree,
    node: TaskNode,
    parent_result: NodeResult | None,
    constraints: ConstraintSet,
) -> str:
    """Build the task context block injected into the user message."""
    parts = []

    # --- Root Goal (always present, as anchor) ---
    root_goal = tree.nodes[tree.root_id].goal if tree.root_id else ""
    parts.append(f"[Root Goal]\n{root_goal}\n[/Root Goal]")

    # --- Parent goal (what the parent asked this node to do) ---
    if node.parent_id is not None:
        parent = tree.nodes[node.parent_id]
        parts.append(f"[Parent Task]\n{parent.goal}\n[/Parent Task]")

    # --- Parent result (one level only) ---
    if parent_result is not None:
        parts.append(f"[Parent Result]\n{parent_result.summary}\n[/Parent Result]")
        if parent_result.artifacts:
            artifact_lines = [
                f"- [{a.type}] {a.path or ''}: {a.description}"
                for a in parent_result.artifacts
            ]
            parts.append("[Parent Artifacts]\n" + "\n".join(artifact_lines) + "\n[/Parent Artifacts]")

    # --- Constraint block ---
    constraint_lines = [f"- max_depth: {constraints.max_depth}"]
    if constraints.forbidden_actions:
        constraint_lines.append(f"- forbidden_actions: {', '.join(constraints.forbidden_actions)}")
    constraint_lines.append(f"- failure_count_limit: {constraints.failure_count_limit}")
    parts.append("[Constraints]\n" + "\n".join(constraint_lines) + "\n[/Constraints]")

    # --- Current node goal ---
    parts.append(f"[Your Task]\n{node.goal}\n[/Your Task]")

    # --- Verification failure context (for retry after verification failure) ---
    node_obj = tree.nodes.get(node.id)
    if node_obj and node_obj.verification_failure:
        parts.append(
            f"[Previous Attempt Failed — Verification]\n"
            f"{node_obj.verification_failure}\n"
            f"[/Previous Attempt Failed]"
        )

    # --- Root node: force structured decomposition output ---
    if node.parent_id is None:
        parts.append(
            "[Root Planning Context]\n"
            f"- Planning Level: ROOT (depth=0)\n"
            f"- Node: {node.id}\n"
            f"- Workspace State: {node.workspace_state.value}\n"
            "[/Root Planning Context]"
        )
        parts.append(
            "[Decomposition Instruction]\n"
            "You are the ROOT PLANNING NODE. After analyzing the Root Goal, you MUST either:\n"
            "1. Output a structured list of subtasks (if the goal requires multiple steps), OR\n"
            "2. Output a single line: DONE (if the goal is single-step and can be executed directly)\n"
            "\n"
            "Required output formats (choose ONE):\n"
            "  JSON array:   [{\"goal\": \"subtask description\", \"description\": \"why needed\"}]\n"
            "  Numbered list: \"1. First subtask\\n2. Second subtask\\n3. Third subtask\"\n"
            "  Markdown list: \"- First subtask\\n- Second subtask\\n- Third subtask\"\n"
            "\n"
            "IMPORTANT RULES:\n"
            "  - Do NOT ask clarifying questions — if the goal is clear, decompose immediately\n"
            "  - Do NOT execute the subtasks yourself — only plan them\n"
            "  - Each subtask should be independently meaningful and achievable\n"
            "  - Maximum subtasks per decomposition: 10\n"
            "[/Decomposition Instruction]"
        )

    return "\n\n".join(parts)


def build_result_from_agent_response(
    node_id: str,
    agent_content: str,
    artifacts: list[dict[str, Any]] | None = None,
    token_spent: int = 0,
    workspace_state: str = "clean",
) -> NodeResult:
    """Convert an ExecutionAgent's raw output into a NodeResult.

    Call this from ExecutionAgent.execute() after getting the LLM response.

    Args:
        node_id: the node's id
        agent_content: the assistant's final content
        artifacts: optional list of artifact dicts (type, path, description)
        token_spent: estimated token count for this execution
        workspace_state: "clean" | "partial" | "dirty" indicating workspace modifications

    Returns:
        A NodeResult ready to return to the Scheduler.
    """
    from nanobot.agent.tasktree.models import Artifact, TaskStatus, WorkspaceState

    artifact_objs = []
    if artifacts:
        for a in artifacts:
            artifact_objs.append(Artifact(
                type=a.get("type", "unknown"),
                path=a.get("path"),
                description=a.get("description", ""),
            ))

    # Build summary: prefer artifact descriptions over raw LLM output.
    if artifact_objs:
        summary_parts = [a.description or f"{a.type}: {a.path}" for a in artifact_objs]
        summary = "; ".join(summary_parts[:5])  # cap at 5 artifacts
    elif agent_content:
        summary = _summarize_content(agent_content)
    else:
        summary = "Task completed."

    return NodeResult(
        node_id=node_id,
        status=TaskStatus.DONE,
        summary=summary,
        artifacts=artifact_objs,
        constraints_respected=True,
        token_spent=token_spent,
        workspace_state=WorkspaceState(workspace_state),
    )


def build_failure_from_error(
    node_id: str,
    error_message: str,
    root_cause: str = "unknown",
) -> NodeResult:
    """Build a FailureReport from an execution error."""
    from nanobot.agent.tasktree.models import FailureReport, RootCause, TaskStatus, WorkspaceState

    cause_map = {
        "timeout": RootCause.API_TIMEOUT,
        "file_not_found": RootCause.FILE_NOT_FOUND,
        "constraint_veto": RootCause.CONSTRAINT_VETO,
        "no_remaining_options": RootCause.NO_REMAINING_OPTIONS,
        "max_replan": RootCause.MAX_REPLAN_REACHED,
    }
    return FailureReport(
        node_id=node_id,
        status=TaskStatus.FAILED,
        root_cause=cause_map.get(root_cause, RootCause.UNKNOWN),
        summary=error_message,
        tried=[],
        remaining_options=[],
        constraint_veto=False,
        workspace_state=WorkspaceState.CLEAN,
    )


def _summarize_content(content: str, max_len: int = 200) -> str:
    """Truncate or summarize content for the result summary field."""
    if not content:
        return ""
    content = content.strip()
    if len(content) <= max_len:
        return content
    return content[:max_len] + "..."
