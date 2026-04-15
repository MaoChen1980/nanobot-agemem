"""Tests for LLMSubgoalParser: JSON, numbered list, markdown list, no structure."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.execution.subgoal import LLMSubgoalParser, SubgoalParser
from nanobot.agent.tasktree.models import NodeResult, TaskStatus


class TestSubgoalParser:
    def test_json_array(self):
        parser = LLMSubgoalParser()
        result = NodeResult(
            node_id="root",
            summary='[{"goal": "first task"}, {"goal": "second task"}]',
            status=TaskStatus.DONE,
        )

        goals = parser.parse(result)
        assert len(goals) == 2
        assert "first task" in goals[0]
        assert "second task" in goals[1]

    def test_numbered_list(self):
        parser = LLMSubgoalParser()

        result = NodeResult(
            node_id="root",
            summary='''
        The steps are:
        1. Analyze the requirements
        2. Write the implementation
        3. Test it
        ''',
            status=TaskStatus.DONE,
        )

        goals = parser.parse(result)
        assert len(goals) == 3
        assert "Analyze" in goals[0]
        assert "Write" in goals[1]
        assert "Test" in goals[2]

    def test_markdown_list(self):
        parser = LLMSubgoalParser()

        result = NodeResult(
            node_id="root",
            summary='''
        Subtasks:
        - Write the tests
        - Refactor the code
        - Update documentation
        ''',
            status=TaskStatus.DONE,
        )

        goals = parser.parse(result)
        assert len(goals) >= 2
        assert any("test" in g.lower() for g in goals)

    def test_done_response(self):
        parser = LLMSubgoalParser()

        result = NodeResult(node_id="root", summary="DONE", status=TaskStatus.DONE)

        goals = parser.parse(result)
        assert goals == []

    def test_no_structure_returns_empty(self):
        parser = LLMSubgoalParser()

        result = NodeResult(
            node_id="n",
            summary="I'll think about this step by step. This is complex.",
            status=TaskStatus.DONE,
        )

        goals = parser.parse(result)
        # No clear structure → empty list
        assert goals == []


class TestSubgoalParserProtocol:
    def test_interface_compliance(self):
        # Verify LLMSubgoalParser satisfies the SubgoalParser Protocol
        parser: SubgoalParser = LLMSubgoalParser()
        assert callable(parser.parse)