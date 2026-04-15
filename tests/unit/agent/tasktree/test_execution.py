"""Tests for DefaultExecutionAgent._extract_artifacts()."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.execution.default import DefaultExecutionAgent, DefaultExecutionAgentConfig


class TestExtractArtifacts:
    def test_write_file_produces_artifact(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"file_path": "/tmp/output.txt", "content": "hello"},
                        }
                    }
                ],
            }
        ]
        artifacts, ws_state = agent._extract_artifacts(messages)
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "/tmp/output.txt"
        assert artifacts[0]["type"] == "file_written"
        assert ws_state == "partial"

    def test_edit_file_produces_artifact(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {"file_path": "/tmp/existing.txt", "old_string": "a", "new_string": "b"},
                        }
                    }
                ],
            }
        ]
        artifacts, ws_state = agent._extract_artifacts(messages)
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "file_modified"
        assert ws_state == "partial"

    def test_bash_marks_dirty(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "rm -rf /tmp/junk"},
                        }
                    }
                ],
            }
        ]
        _, ws_state = agent._extract_artifacts(messages)
        assert ws_state == "dirty"

    def test_git_marks_dirty(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "git",
                            "arguments": {"args": "commit -m 'fix'"},
                        }
                    }
                ],
            }
        ]
        _, ws_state = agent._extract_artifacts(messages)
        assert ws_state == "dirty"

    def test_multiple_writes_partial(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"file_path": "/tmp/a.txt", "content": "a"}}},
                    {"function": {"name": "write_file", "arguments": {"file_path": "/tmp/b.txt", "content": "b"}}},
                ],
            }
        ]
        artifacts, ws_state = agent._extract_artifacts(messages)
        assert len(artifacts) == 2
        assert ws_state == "partial"

    def test_write_then_bash_becomes_dirty(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"file_path": "/tmp/a.txt", "content": "a"}}},
                    {"function": {"name": "bash", "arguments": {"command": "echo done"}}},
                ],
            }
        ]
        _, ws_state = agent._extract_artifacts(messages)
        assert ws_state == "dirty"

    def test_no_tool_calls_clean(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {"role": "assistant", "content": "Here's my response"},
            {"role": "user", "content": "hello"},
        ]
        _, ws_state = agent._extract_artifacts(messages)
        assert ws_state == "clean"

    def test_arguments_string_parsed(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": '{"file_path": "/tmp/test.txt", "content": "hi"}',
                        }
                    }
                ],
            }
        ]
        artifacts, _ = agent._extract_artifacts(messages)
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "/tmp/test.txt"

    def test_path_alternate_keys(self):
        agent = DefaultExecutionAgent.__new__(DefaultExecutionAgent)
        # Test "path" key instead of "file_path"
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "/tmp/via_path.txt", "content": "hi"},
                        }
                    }
                ],
            }
        ]
        artifacts, _ = agent._extract_artifacts(messages)
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "/tmp/via_path.txt"