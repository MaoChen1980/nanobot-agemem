"""Tests for build_node_context, build_result_from_agent_response, build_failure_from_error."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.context import (
    build_failure_from_error,
    build_node_context,
    build_result_from_agent_response,
)
from nanobot.agent.tasktree.models import (
    Artifact,
    ConstraintSet,
    NodeResult,
    RootCause,
    TaskNode,
    TaskStatus,
    WorkspaceState,
)
from nanobot.agent.tasktree.tree import TaskTree


class DummyContextBuilder:
    def build_system_prompt(self, channel=None):
        return "You are a helpful assistant."

    def _build_runtime_context(self, channel, chat_id, timezone=None, session_summary=None):
        return "[runtime context placeholder]"

    @property
    def timezone(self):
        return "UTC"

    def _get_retriever(self):
        return None


def _child_id(tree: TaskTree, parent_id: str, index: int) -> str:
    return tree.nodes[parent_id].children[index]


class TestBuildNodeContext:
    def setup_method(self):
        self.builder = DummyContextBuilder()
        self.tree = TaskTree()
        self.tree.create_root(goal="root goal")

    def test_root_node_has_all_blocks(self):
        node = self.tree.nodes[self.tree.root_id]
        constraints = ConstraintSet(max_depth=20, forbidden_actions=[], failure_count_limit=2)

        messages = build_node_context(
            context_builder=self.builder,
            tree=self.tree,
            node=node,
            parent_result=None,
            constraints=constraints,
        )

        assert len(messages) >= 2
        content = messages[-1]["content"]
        assert "[Root Goal]" in content
        assert "root goal" in content
        assert "[Your Task]" in content
        assert "[TaskTree Decomposition —" in content
        # Root node should have rich planning context
        assert "[Root Planning Context]" in content
        assert "ROOT" in content
        assert "depth=0" in content
        # Decomposition instruction should be detailed
        assert "2-5" in content or "subtasks" in content

    def test_child_node_has_parent_result(self):
        self.tree.add_child("root", "child goal")
        child_id = _child_id(self.tree, "root", 0)
        child = self.tree.nodes[child_id]
        parent_result = NodeResult(
            node_id="root",
            summary="parent did something",
            artifacts=[Artifact(type="file_written", path="/tmp/out.txt", description="output")],
        )
        constraints = ConstraintSet()

        messages = build_node_context(
            context_builder=self.builder,
            tree=self.tree,
            node=child,
            parent_result=parent_result,
            constraints=constraints,
        )

        content = messages[-1]["content"]
        assert "[Parent Result]" in content
        assert "parent did something" in content
        assert "[Parent Artifacts]" in content
        assert "/tmp/out.txt" in content

    def test_verification_failure_injected(self):
        node = self.tree.nodes[self.tree.root_id]
        node.verification_failure = "Verification failed: output incomplete"

        constraints = ConstraintSet()
        messages = build_node_context(
            context_builder=self.builder,
            tree=self.tree,
            node=node,
            parent_result=None,
            constraints=constraints,
        )

        content = messages[-1]["content"]
        assert "[Previous Attempt Failed" in content
        assert "Verification failed: output incomplete" in content

    def test_constraint_block_included(self):
        node = self.tree.nodes[self.tree.root_id]
        constraints = ConstraintSet(
            max_depth=5,
            forbidden_actions=["rm", "format"],
            failure_count_limit=3,
        )

        messages = build_node_context(
            context_builder=self.builder,
            tree=self.tree,
            node=node,
            parent_result=None,
            constraints=constraints,
        )

        content = messages[-1]["content"]
        assert "[Constraints]" in content
        assert "max_depth: 5" in content
        assert "rm" in content
        assert "failure_count_limit: 3" in content


class TestBuildResultFromAgentResponse:
    def test_empty_response(self):
        result = build_result_from_agent_response(
            node_id="n1",
            agent_content="",
        )
        assert result.node_id == "n1"
        assert result.status == TaskStatus.DONE

    def test_uses_artifact_descriptions_for_summary(self):
        artifacts = [
            {"type": "file_written", "path": "/tmp/a.txt", "description": "file a"},
            {"type": "file_written", "path": "/tmp/b.txt", "description": "file b"},
        ]
        result = build_result_from_agent_response(
            node_id="n1",
            agent_content="Some long raw LLM output that should not be used as summary",
            artifacts=artifacts,
        )
        assert "file a" in result.summary
        assert "file b" in result.summary
        assert "raw LLM output" not in result.summary

    def test_falls_back_to_content(self):
        result = build_result_from_agent_response(
            node_id="n1",
            agent_content="The agent said hello",
            artifacts=None,
        )
        assert "The agent said hello" in result.summary

    def test_workspace_state(self):
        result = build_result_from_agent_response(
            node_id="n1",
            agent_content="done",
            workspace_state="dirty",
        )
        assert result.workspace_state == WorkspaceState.DIRTY

    def test_token_spent(self):
        result = build_result_from_agent_response(
            node_id="n1",
            agent_content="done",
            token_spent=500,
        )
        assert result.token_spent == 500


class TestBuildFailureFromError:
    def test_unknown_root_cause(self):
        result = build_failure_from_error(node_id="n1", error_message="something broke")
        assert result.status == TaskStatus.FAILED
        assert result.root_cause == RootCause.UNKNOWN
        assert result.summary == "something broke"

    def test_timeout_root_cause(self):
        result = build_failure_from_error(
            node_id="n1",
            error_message="request timed out",
            root_cause="timeout",
        )
        assert result.root_cause == RootCause.API_TIMEOUT

    def test_constraint_veto_root_cause(self):
        result = build_failure_from_error(
            node_id="n1",
            error_message="action not allowed",
            root_cause="constraint_veto",
        )
        assert result.root_cause == RootCause.CONSTRAINT_VETO

    def test_max_replan_root_cause(self):
        result = build_failure_from_error(
            node_id="n1",
            error_message="max replan reached",
            root_cause="max_replan",
        )
        assert result.root_cause == RootCause.MAX_REPLAN_REACHED