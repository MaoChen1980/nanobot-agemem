"""Extractor: pull tool_call → result pairs from message history.

Extracts structured (tool, input, output, success) tuples from conversation
messages, which become candidate facts for causal memory.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ToolCallPair:
    """A single tool call and its result, extracted from message history."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str  # result as string (for now)
    tool_call_id: str
    success: bool
    timestamp: str  # ISO format

    def to_fact_content(self) -> dict[str, Any]:
        """Convert to fact content dict for causal storage."""
        return {
            "tool": self.tool_name,
            "input": self.tool_input,
            "output": self.tool_output[:500] if self.tool_output else "",  # truncate long output
            "result": "success" if self.success else "failure",
        }


def extract_tool_call_pairs(messages: list[dict[str, Any]]) -> list[ToolCallPair]:
    """Extract all tool_call → result pairs from a message history.

    Args:
        messages: List of message dicts with role, content, tool_calls, tool_call_id, etc.

    Returns:
        List of ToolCallPair objects in temporal order.
    """
    pairs: list[ToolCallPair] = []
    pending_calls: dict[str, dict[str, Any]] = {}

    for msg in messages:
        role = msg.get("role", "")
        ts = msg.get("timestamp") or datetime.now().isoformat()

        # Assistant message: collect tool calls
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                func_name = func.get("name", "unknown")
                func_args = func.get("arguments", "{}")
                if isinstance(func_args, str):
                    import json as _json
                    try:
                        func_args = _json.loads(func_args)
                    except Exception:
                        func_args = {"raw": func_args}

                pending_calls[tc_id] = {
                    "tool_name": func_name,
                    "tool_input": func_args,
                    "tool_call_id": tc_id,
                    "timestamp": ts,
                }

        # Tool message: match to pending call and create pair
        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id in pending_calls:
                call = pending_calls.pop(tc_id)
                content = msg.get("content", "")

                # Determine success: tool content doesn't indicate failure by itself,
                # but we can check for common error patterns
                success = not (
                    content.startswith("Error:") or
                    content.startswith("Exception:") or
                    '"success":false' in content.lower() or
                    content.startswith("Traceback")
                )

                pairs.append(ToolCallPair(
                    tool_name=call["tool_name"],
                    tool_input=call["tool_input"],
                    tool_output=content,
                    tool_call_id=call["tool_call_id"],
                    success=success,
                    timestamp=call["timestamp"],
                ))

    return pairs


def extract_facts_from_pairs(
    pairs: list[ToolCallPair],
    importance: float = 0.5,
) -> list[dict[str, Any]]:
    """Convert tool call pairs to fact dicts ready for causal storage.

    Args:
        pairs: List of ToolCallPair objects
        importance: Base importance for all extracted facts

    Returns:
        List of dicts suitable for CausalStore.add_fact()
    """
    facts = []
    for pair in pairs:
        facts.append({
            "content": pair.to_fact_content(),
            "fact_type": "tool_call",
            "importance": importance,
            "timestamp": pair.timestamp,
            "tags": ["extracted", f"tool:{pair.tool_name}"],
        })
    return facts
