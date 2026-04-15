"""Unit tests for DefaultConstraintAgent."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.execution.constraint import (
    DefaultConstraintAgent,
    DefaultConstraintAgentConfig,
    _build_constraint_prompt,
    _parse_constraint_response,
)
from nanobot.agent.tasktree.models import ConstraintSet, NodeResult, TaskNode


class TestBuildConstraintPrompt:
    def test_includes_root_goal(self):
        node = TaskNode(id="root", goal="build a bot", parent_id=None, depth=0)
        prompt = _build_constraint_prompt(node, None, "build a bot")
        assert "build a bot" in prompt
        assert "Root Goal:" in prompt

    def test_includes_node_info(self):
        node = TaskNode(id="root.0", goal="subtask", parent_id="root", depth=1)
        prompt = _build_constraint_prompt(node, None, "root goal")
        assert "root.0" in prompt
        assert "depth 1" in prompt
        assert "subtask" in prompt

    def test_includes_parent_result_when_present(self):
        node = TaskNode(id="root.0", goal="child task", parent_id="root", depth=1)
        parent_result = NodeResult(node_id="root", summary="parent summary")
        prompt = _build_constraint_prompt(node, parent_result, "root goal")
        assert "parent summary" in prompt

    def test_excludes_parent_result_when_none(self):
        node = TaskNode(id="root", goal="root goal", parent_id=None, depth=0)
        prompt = _build_constraint_prompt(node, None, "root goal")
        assert "Parent result:" not in prompt


class TestParseConstraintResponse:
    def test_parses_valid_json(self):
        config = DefaultConstraintAgentConfig()
        result = _parse_constraint_response(
            '{"max_depth": 5, "forbidden_actions": ["rm"], "failure_count_limit": 3}',
            config,
        )
        assert result.max_depth == 5
        assert "rm" in result.forbidden_actions
        assert result.failure_count_limit == 3

    def test_falls_back_to_defaults_on_bad_json(self):
        config = DefaultConstraintAgentConfig(max_depth=7, forbidden_actions=["a"], failure_count_limit=5)
        result = _parse_constraint_response("not json at all", config)
        assert result.max_depth == 7
        assert result.forbidden_actions == ["a"]
        assert result.failure_count_limit == 5

    def test_partial_json_uses_defaults(self):
        config = DefaultConstraintAgentConfig(max_depth=9)
        result = _parse_constraint_response('{"max_depth": 3}', config)
        assert result.max_depth == 3
        # Other fields fall back to config defaults
        assert result.max_depth != config.max_depth  # overridden


class TestDefaultConstraintAgent:
    @pytest.mark.asyncio
    async def test_get_constraints_returns_parsed_from_llm(self):
        class FakeProvider:
            async def complete(self, prompt, model, max_tokens):
                return type("Resp", (), {
                    "content": '{"max_depth": 8, "forbidden_actions": ["delete"], "failure_count_limit": 4}'
                })()

            def get_default_model(self):
                return "test-model"

        agent = DefaultConstraintAgent(provider=FakeProvider())
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        constraints = await agent.get_constraints(node, None, "test goal")
        assert constraints.max_depth == 8
        assert "delete" in constraints.forbidden_actions
        assert constraints.failure_count_limit == 4

    @pytest.mark.asyncio
    async def test_get_constraints_falls_back_on_llm_error(self):
        class FailProvider:
            async def complete(self, prompt, model, max_tokens):
                raise RuntimeError("llm error")

            def get_default_model(self):
                return "test-model"

        agent = DefaultConstraintAgent(provider=FailProvider())
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        constraints = await agent.get_constraints(node, None, "test goal")
        # Should fall back to config defaults
        assert constraints.max_depth == agent.config.max_depth

    @pytest.mark.asyncio
    async def test_apply_memory_veto_no_similar_failures(self):
        class FakeRetriever:
            def retrieve(self, query, top_k):
                return []

        class FakeProvider:
            async def complete(self, prompt, model, max_tokens):
                return type("Resp", (), {"content": '{"max_depth": 10, "forbidden_actions": [], "failure_count_limit": 10}'})()

            def get_default_model(self):
                return "test-model"

        agent = DefaultConstraintAgent(
            provider=FakeProvider(),
            memory_retriever=FakeRetriever(),
        )
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        constraints = await agent.get_constraints(node, None, "test goal")
        # No memory veto applied
        assert constraints.failure_count_limit == 10
        assert constraints.forbidden_actions == []

    @pytest.mark.asyncio
    async def test_apply_memory_veto_reduces_limit_on_similar_failures(self):
        class FakeRetriever:
            def retrieve(self, query, top_k):
                # 2 similar failures found
                return [
                    type("ScoredEntry", (), {
                        "entry": type("Entry", (), {"tags": ["failure"]})()
                    })(),
                    type("ScoredEntry", (), {
                        "entry": type("Entry", (), {"tags": ["failure"]})()
                    })(),
                ]

        class FakeProvider:
            async def complete(self, prompt, model, max_tokens):
                return type("Resp", (), {"content": '{"max_depth": 10, "forbidden_actions": [], "failure_count_limit": 10}'})()

            def get_default_model(self):
                return "test-model"

        agent = DefaultConstraintAgent(
            provider=FakeProvider(),
            memory_retriever=FakeRetriever(),
            config=DefaultConstraintAgentConfig(failure_count_limit=10),
        )
        node = TaskNode(id="root.0", goal="test", parent_id="root", depth=1)
        constraints = await agent.get_constraints(node, None, "test goal")
        # 2 similar failures → limit reduced from 10 to 8
        assert constraints.failure_count_limit == 8

    @pytest.mark.asyncio
    async def test_apply_memory_veto_adds_forbidden_action_when_count_ge_3(self):
        class FakeRetriever:
            def retrieve(self, query, top_k):
                # 3+ similar failures
                return [
                    type("ScoredEntry", (), {
                        "entry": type("Entry", (), {"tags": ["failure"]})()
                    })(),
                    type("ScoredEntry", (), {
                        "entry": type("Entry", (), {"tags": ["failure"]})()
                    })(),
                    type("ScoredEntry", (), {
                        "entry": type("Entry", (), {"tags": ["failure"]})()
                    })(),
                ]

        class FakeProvider:
            async def complete(self, prompt, model, max_tokens):
                return type("Resp", (), {"content": '{"max_depth": 10, "forbidden_actions": [], "failure_count_limit": 10}'})()

            def get_default_model(self):
                return "test-model"

        agent = DefaultConstraintAgent(
            provider=FakeProvider(),
            memory_retriever=FakeRetriever(),
        )
        node = TaskNode(id="root.0", goal="test", parent_id="root", depth=1)
        constraints = await agent.get_constraints(node, None, "test goal")
        # count >= 3 → forbidden action added
        assert any("repeat_failure_pattern_root.0" in fa for fa in constraints.forbidden_actions)

    @pytest.mark.asyncio
    async def test_apply_memory_veto_gracefully_handles_retriever_error(self):
        class BadRetriever:
            def retrieve(self, query, top_k):
                raise RuntimeError("retriever error")

        class FakeProvider:
            async def complete(self, prompt, model, max_tokens):
                return type("Resp", (), {"content": '{"max_depth": 10, "forbidden_actions": [], "failure_count_limit": 10}'})()

            def get_default_model(self):
                return "test-model"

        agent = DefaultConstraintAgent(
            provider=FakeProvider(),
            memory_retriever=BadRetriever(),
        )
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        # Should not raise - errors are caught
        constraints = await agent.get_constraints(node, None, "test goal")
        assert constraints.max_depth == 10
