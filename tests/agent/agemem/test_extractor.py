"""Unit tests for tool call extractor."""

import pytest

from nanobot.agent.agemem.extractor import (
    ToolCallPair,
    extract_facts_from_pairs,
    extract_tool_call_pairs,
)


class TestExtractToolCallPairs:
    def test_basic_tool_call_result_pair(self):
        """Assistant tool_call followed by tool result produces one pair."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.txt"}',
                        },
                    }
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "file contents here",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)

        assert len(pairs) == 1
        assert pairs[0].tool_name == "read_file"
        assert pairs[0].tool_input == {"path": "a.txt"}
        assert pairs[0].tool_output == "file contents here"
        assert pairs[0].tool_call_id == "call-1"
        assert pairs[0].success is True

    def test_multiple_pairs_ordered(self):
        """Multiple tool calls are returned in order."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "call-2", "function": {"name": "write_file", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "read result",
                "timestamp": "2026-04-18T10:00:01",
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "content": "write result",
                "timestamp": "2026-04-18T10:00:02",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert len(pairs) == 2
        assert pairs[0].tool_name == "read_file"
        assert pairs[1].tool_name == "write_file"

    def test_error_detected_as_failure(self):
        """Tool output starting with 'Error:' is marked as failure."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-bad", "function": {"name": "read_file", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-bad",
                "content": "Error: file not found",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert len(pairs) == 1
        assert pairs[0].success is False

    def test_exception_detected_as_failure(self):
        """Tool output starting with 'Traceback' is marked as failure."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-exp", "function": {"name": "run_command", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-exp",
                "content": "Traceback (most recent call last):\n  File 'main.py', line 1",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs[0].success is False

    def test_json_success_false_detected(self):
        """Tool output with '"success":false' is marked as failure."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-api", "function": {"name": "api_call", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-api",
                "content": '{"success": false, "error": "rate limited"}',
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs[0].success is False

    def test_arguments_as_dict(self):
        """Arguments passed as dict (not JSON string) are handled."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "read_file", "arguments": {"path": "b.txt"}}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "ok",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs[0].tool_input == {"path": "b.txt"}

    def test_unmatched_tool_result_ignored(self):
        """Tool result with no matching pending call is ignored."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "orphan",
                "content": "result without call",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs == []

    def test_regular_messages_ignored(self):
        """User and assistant messages without tool calls are ignored."""
        messages = [
            {"role": "user", "content": "hello", "timestamp": "2026-04-18T10:00:00"},
            {"role": "assistant", "content": "hi there", "timestamp": "2026-04-18T10:00:01"},
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs == []


class TestExtractFactsFromPairs:
    def test_converts_pair_to_fact_dict(self):
        """extract_facts_from_pairs creates correct fact-ready dicts."""
        pairs = [
            ToolCallPair(
                tool_name="read_file",
                tool_input={"path": "a.txt"},
                tool_output="hello",
                tool_call_id="call-1",
                success=True,
                timestamp="2026-04-18T10:00:00",
            )
        ]
        facts = extract_facts_from_pairs(pairs, importance=0.7)

        assert len(facts) == 1
        assert facts[0]["fact_type"] == "tool_call"
        assert facts[0]["importance"] == 0.7
        assert facts[0]["timestamp"] == "2026-04-18T10:00:00"
        assert facts[0]["content"]["tool"] == "read_file"
        assert facts[0]["content"]["result"] == "success"
        assert "extracted" in facts[0]["tags"]
        assert "tool:read_file" in facts[0]["tags"]

    def test_failure_sets_result_failure(self):
        """Failed tool call has result='failure' in fact content."""
        pairs = [
            ToolCallPair(
                tool_name="write_file",
                tool_input={"path": "x.txt"},
                tool_output="Error: disk full",
                tool_call_id="call-1",
                success=False,
                timestamp="2026-04-18T10:00:00",
            )
        ]
        facts = extract_facts_from_pairs(pairs)
        assert facts[0]["content"]["result"] == "failure"


class TestToolCallPair_toFactContent:
    def test_truncates_long_output(self):
        """to_fact_content() truncates output to 500 chars."""
        pair = ToolCallPair(
            tool_name="read_file",
            tool_input={},
            tool_output="x" * 1000,
            tool_call_id="call-1",
            success=True,
            timestamp="2026-04-18T10:00:00",
        )
        content = pair.to_fact_content()
        assert len(content["output"]) == 500
        assert content["output"] == "x" * 500

    def test_arguments_raw_string_not_json(self):
        """Arguments as raw string (not JSON) are wrapped in {"raw": ...}."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "run_command", "arguments": "ls -la"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "file listing",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs[0].tool_input == {"raw": "ls -la"}

    def test_missing_function_name(self):
        """Tool call with no function name uses 'unknown'."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "ok",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs[0].tool_name == "unknown"

    def test_empty_tool_calls_array(self):
        """Assistant message with empty tool_calls array produces no pairs."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [],
                "timestamp": "2026-04-18T10:00:00",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs == []

    def test_only_tool_result_without_call(self):
        """Orphan tool result (no matching call) is silently ignored."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "orphan-99",
                "content": "I have no call",
                "timestamp": "2026-04-18T10:00:01",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:02",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "content",
                "timestamp": "2026-04-18T10:00:03",
            },
        ]
        # Only the matched pair should be returned
        pairs = extract_tool_call_pairs(messages)
        assert len(pairs) == 1
        assert pairs[0].tool_call_id == "call-1"

    def test_timestamp_from_message(self):
        """Timestamp is taken from the assistant message (not tool result)."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "result",
                "timestamp": "2099-01-01T00:00:00",  # far future
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs[0].timestamp == "2026-04-18T10:00:00"

    def test_tool_result_with_unicode(self):
        """Tool result with unicode content is preserved."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "你好 世界 🌍",
                "timestamp": "2026-04-18T10:00:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        assert pairs[0].tool_output == "你好 世界 🌍"
        assert pairs[0].success is True
