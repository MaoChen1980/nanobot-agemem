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

import re
from typing import TYPE_CHECKING, Any

from nanobot.agent.tasktree.models import (
    ConstraintSet,
    NodeResult,
    TaskNode,
)
from nanobot.utils.helpers import strip_think

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
    # For TaskTree root nodes, we use a minimal planning-focused system prompt
    # that overrides the nanobot helpful assistant personality.
    if node.parent_id is None:
        # Minimal planning-only system prompt for root decomposition
        system_prompt = (
            "You are a task planning assistant. Your ONLY job is to decompose goals into subtasks.\n"
            "RULES:\n"
            "1. Output ONLY a ##[TASKS]...##[/TASKS] block in your response\n"
            "2. Never give explanations, refusals, or suggestions — ONLY output the TASKS block\n"
            "3. If you cannot do something directly, decompose it into subtasks that CAN be done\n"
            "4. Do NOT use tools. Do NOT try to act. Only output TASKS block.\n"
            "5. Format: ##[TASKS][\"subtask 1\", \"subtask 2\", \"subtask 3\"]##[/TASKS]\n"
            "6. Always provide at least one subtask — never output empty []\n"
            "7. If user asks for something impossible, STILL output subtasks for what CAN be done\n"
        )
    else:
        system_prompt = context_builder.build_system_prompt(channel=channel)

    # 2. Root node: retrieve relevant memories (tagged as imperfect reference)
    if node.parent_id is None:
        retriever = context_builder._get_retriever()
        if retriever:
            memories = retriever.retrieve(node.goal, top_k=3)
            if memories:
                lines = []
                for r in memories:
                    e = r.entry
                    ts = e.created_at[:16] if e.created_at else ""
                    src = "来源: nanobot记忆"
                    lines.append(f"- {src} {ts}: {e.content[:150]}{'...' if len(e.content) > 150 else ''}")
                memory_context = "\n".join(lines)
                system_prompt += (
                    f"\n\n[记忆参考 — 仅供参考，可能有偏差，不保证准确]\n"
                    f"{memory_context}\n"
                    f"[/记忆参考]"
                )

    # 3. Build the node task block (injected into user message)
    task_block = _build_task_block(tree, node, parent_result, constraints)

    # 4. Assemble messages
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
        # If parent asked user a question and got an answer, include it
        # so this node can act on the user's input.
        if parent_result.user_input_question:
            parts.append(
                f"[Parent User Input — Question]\n{parent_result.user_input_question}\n[/Parent User Input — Question]"
            )
            if parent_result.user_input_answer:
                parts.append(
                    f"[Parent User Input — Answer]\n{parent_result.user_input_answer}\n[/Parent User Input — Answer]"
                )

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

    # --- User input request (for all non-root nodes) ---
    # Root nodes use the TaskTree Decomposition block below instead.
    if node.parent_id is not None:
        parts.append(
            "[User Input — Optional]\n"
            "If you need information from the user to complete this task, you may ask.\n"
            "Format: output a JSON object starting with { and include a field:\n"
            '{"summary": "brief description of what you did", "user_input_request": "the question you need answered"}\n'
            "Example:\n"
            '{"summary": "Checking flight options", "user_input_request": "What city are you flying from and to?"}\n'
            "Or in plain text: include [USER_INPUT_REQUEST]Your question here[/USER_INPUT_REQUEST]\n"
            "[/User Input]"
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
            "[TaskTree Decomposition — MANDATORY OUTPUT]\n"
            "You are in TASK PLANNING MODE. Your ONLY job is to decompose goals into subtasks.\n"
            "You MUST output a ##[TASKS]...##[/TASKS] block — this is mandatory, not optional.\n"
            "\n"
            "MANDATORY RULES:\n"
            "1. You MUST output a ##[TASKS]...##[/TASKS] block for EVERY goal, no exceptions\n"
            "2. If you cannot do something directly, your ONLY option is to output subtasks\n"
            "3. NEVER give direct answers, explanations, or refusals — only output TASKS blocks\n"
            "4. If you say 'I cannot do X', you MUST still provide subtasks for how to accomplish X\n"
            "\n"
            "Example 1 (goal is achievable):\n"
            'Goal: "Write a calculator app"\n'
            'Output: ##[TASKS]["Write calculator Python file", "Test calculator", "Verify file exists"]##[/TASKS]\n'
            "\n"
            "Example 2 (goal seems impossible):\n"
            'Goal: "Book a flight" (no API available)\n'
            'Output: ##[TASKS]["Search for flights online", "Ask user for travel date and preferences", "Show search results to user", "Guide user to booking platform"]##[/TASKS]\n'
            "\n"
            "Example 3 (goal needs user input):\n"
            'Goal: "Book a flight from Beijing to Shanghai"\n'
            'Output: ##[TASKS][{"goal": "Book flight from Beijing to Shanghai", "user_input_request": "What date do you want to fly?"}]##[/TASKS]\n'
            "\n"
            "Output ONLY this exact block format:\n"
            "```\n"
            "##[TASKS]\n"
            '["subtask 1", "subtask 2", "subtask 3"]\n'
            "##[/TASKS]\n"
            "```\n"
            "\n"
            "Do NOT output anything else. Do NOT explain. Do NOT refuse.\n"
            "If you do not output a ##[TASKS]...##[/TASKS] block, the system will treat this as a failure.\n"
            "[/TaskTree Decomposition]"
        )

    return "\n\n".join(parts)


def build_result_from_agent_response(
    node_id: str,
    agent_content: str,
    artifacts: list[dict[str, Any]] | None = None,
    token_spent: int = 0,
    workspace_state: str = "clean",
    user_input_question: str | None = None,
) -> NodeResult:
    """Convert an ExecutionAgent's raw output into a NodeResult.

    Call this from ExecutionAgent.execute() after getting the LLM response.

    Args:
        node_id: the node's id
        agent_content: the assistant's final content
        artifacts: optional list of artifact dicts (type, path, description)
        token_spent: estimated token count for this execution
        workspace_state: "clean" | "partial" | "dirty" indicating workspace modifications
        user_input_question: if set, the node needs user input before it can complete

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

    # Parse ##[TASKS] block from agent_content to extract children_goals
    children_goals = _try_parse_tasks_block_for_result(agent_content)

    # Strip ##[TASKS] block before building summary to avoid pollution
    clean_content = _strip_tasks_block(agent_content)

    # Build summary: prefer artifact descriptions over raw LLM output.
    if artifact_objs:
        summary_parts = [a.description or f"{a.type}: {a.path}" for a in artifact_objs]
        summary = "; ".join(summary_parts[:5])  # cap at 5 artifacts
    elif clean_content and clean_content.strip():
        summary = _summarize_content(strip_think(clean_content))
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
        user_input_question=user_input_question,
        children_goals=children_goals,
    )


def _strip_tasks_block(text: str) -> str:
    """Remove ##[TASKS]...##[/TASKS] blocks from text, preserving outside content."""
    text = re.sub(r'##\[TASKS\]\s*[\s\S]*?\s*##\[/TASKS\]', '', text)
    text = re.sub(r'## TASKS\s*[\s\S]*?\s*## TASKS', '', text)
    return text.strip()


def _try_parse_tasks_block_for_result(content: str) -> list[str]:
    """Extract children goals from ##[TASKS]...##[/TASKS] block.

    Returns list of child goal strings, or empty list if no block found.
    """
    import json

    # Try ##[TASKS]...##[/TASKS] format
    match = re.search(r'##\[TASKS\]\s*([\s\S]*?)\s*##\[/TASKS\]', content)
    if match:
        json_str = match.group(1).strip()
        if not json_str:
            return []
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                if all(isinstance(x, str) for x in parsed):
                    return [g.strip() for g in parsed if g.strip()]
                if all(isinstance(x, dict) for x in parsed):
                    return [item.get("goal", "").strip() for item in parsed if item.get("goal", "").strip()]
        except json.JSONDecodeError:
            pass

    # Try legacy ## TASKS...## TASKS format
    match = re.search(r'## TASKS\s*([\s\S]*?)\s*## TASKS', content)
    if match:
        json_str = match.group(1).strip()
        if not json_str:
            return []
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                if all(isinstance(x, str) for x in parsed):
                    return [g.strip() for g in parsed if g.strip()]
                if all(isinstance(x, dict) for x in parsed):
                    return [item.get("goal", "").strip() for item in parsed if item.get("goal", "").strip()]
        except json.JSONDecodeError:
            pass

    return []


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
