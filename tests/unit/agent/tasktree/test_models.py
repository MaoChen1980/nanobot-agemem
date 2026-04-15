"""Tests for TaskNode, TaskStatus, RootCause, Artifact, NodeResult, FailureReport, ConstraintSet."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.models import (
    Artifact,
    ConstraintSet,
    FailureReport,
    NodeResult,
    RootCause,
    TaskNode,
    TaskStatus,
    WorkspaceState,
)


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.BLOCKED.value == "blocked"


class TestWorkspaceState:
    def test_values(self):
        assert WorkspaceState.CLEAN.value == "clean"
        assert WorkspaceState.PARTIAL.value == "partial"
        assert WorkspaceState.DIRTY.value == "dirty"


class TestTaskNode:
    def test_defaults(self):
        node = TaskNode(id="n1", goal="test goal")
        assert node.id == "n1"
        assert node.goal == "test goal"
        assert node.status == TaskStatus.PENDING
        assert node.parent_id is None
        assert node.children == []
        assert node.depth == 0
        assert node.result is None
        assert node.failure is None
        assert node.replan_count == 0
        assert node.verification_failure is None
        assert node.workspace_state == WorkspaceState.CLEAN

    def test_is_leaf(self):
        node = TaskNode(id="n1", goal="leaf")
        assert node.is_leaf() is True
        node.children.append("child1")
        assert node.is_leaf() is False

    def test_is_terminal(self):
        node = TaskNode(id="n1", goal="test")
        assert node.is_terminal() is False
        node.status = TaskStatus.DONE
        assert node.is_terminal() is True
        node.status = TaskStatus.FAILED
        assert node.is_terminal() is True
        node.status = TaskStatus.BLOCKED
        assert node.is_terminal() is True
        node.status = TaskStatus.PENDING
        assert node.is_terminal() is False


class TestArtifact:
    def test_to_dict(self):
        artifact = Artifact(type="file_written", path="/tmp/test.txt", description="test file")
        d = artifact.to_dict()
        assert d["type"] == "file_written"
        assert d["path"] == "/tmp/test.txt"
        assert d["description"] == "test file"

    def test_from_dict(self):
        d = {"type": "file_modified", "path": "/tmp/foo.txt", "description": "modified foo"}
        artifact = Artifact.from_dict(d)
        assert artifact.type == "file_modified"
        assert artifact.path == "/tmp/foo.txt"
        assert artifact.description == "modified foo"

    def test_from_dict_missing_path(self):
        d = {"type": "api_call", "description": "some api"}
        artifact = Artifact.from_dict(d)
        assert artifact.path is None
        assert artifact.description == "some api"


class TestNodeResult:
    def test_defaults(self):
        result = NodeResult(node_id="n1")
        assert result.node_id == "n1"
        assert result.status == TaskStatus.DONE
        assert result.summary == ""
        assert result.artifacts == []
        assert result.constraints_respected is True
        assert result.token_spent == 0
        assert result.workspace_state == WorkspaceState.CLEAN

    def test_with_artifacts(self):
        artifact = Artifact(type="file_written", path="/tmp/out.txt", description="output")
        result = NodeResult(node_id="n1", summary="done", artifacts=[artifact], token_spent=100)
        assert len(result.artifacts) == 1
        assert result.artifacts[0].path == "/tmp/out.txt"
        assert result.token_spent == 100

    def test_to_dict(self):
        result = NodeResult(node_id="n1", summary="test summary")
        d = result.to_dict()
        assert d["node_id"] == "n1"
        assert d["status"] == "done"
        assert d["summary"] == "test summary"
        assert d["artifacts"] == []
        assert d["constraints_respected"] is True
        assert d["token_spent"] == 0

    def test_from_dict(self):
        d = {
            "node_id": "n2",
            "status": "done",
            "summary": "loaded",
            "artifacts": [],
            "constraints_respected": True,
            "token_spent": 50,
        }
        result = NodeResult.from_dict(d)
        assert result.node_id == "n2"
        assert result.summary == "loaded"
        assert result.token_spent == 50


class TestFailureReport:
    def test_defaults(self):
        report = FailureReport(node_id="n1", summary="something went wrong")
        assert report.node_id == "n1"
        assert report.status == TaskStatus.FAILED
        assert report.root_cause == RootCause.UNKNOWN
        assert report.summary == "something went wrong"
        assert report.tried == []
        assert report.remaining_options == []
        assert report.constraint_veto is False
        assert report.workspace_state == WorkspaceState.CLEAN

    def test_root_cause_mapping(self):
        # root_cause is a RootCause enum field; string mapping only happens in build_failure_from_error
        report = FailureReport(node_id="n1", summary="timeout", root_cause=RootCause.API_TIMEOUT)
        assert report.root_cause == RootCause.API_TIMEOUT

        report2 = FailureReport(node_id="n2", summary="no options", root_cause=RootCause.NO_REMAINING_OPTIONS)
        assert report2.root_cause == RootCause.NO_REMAINING_OPTIONS

    def test_to_dict(self):
        report = FailureReport(node_id="n1", summary="failed")
        d = report.to_dict()
        assert d["node_id"] == "n1"
        assert d["root_cause"] == "unknown"

    def test_from_dict(self):
        d = {
            "node_id": "n3",
            "status": "failed",
            "root_cause": "max_replan_reached",
            "summary": "too many replans",
            "tried": ["a", "b"],
            "remaining_options": [],
            "constraint_veto": False,
            "workspace_state": "dirty",
        }
        report = FailureReport.from_dict(d)
        assert report.node_id == "n3"
        assert report.root_cause == RootCause.MAX_REPLAN_REACHED
        assert report.workspace_state == WorkspaceState.DIRTY


class TestConstraintSet:
    def test_defaults(self):
        cs = ConstraintSet()
        assert cs.max_depth == 10
        assert cs.forbidden_actions == []
        assert cs.failure_count_limit == 2

    def test_custom_values(self):
        cs = ConstraintSet(max_depth=5, forbidden_actions=["bash", "rm"], failure_count_limit=3)
        assert cs.max_depth == 5
        assert "bash" in cs.forbidden_actions
        assert cs.failure_count_limit == 3