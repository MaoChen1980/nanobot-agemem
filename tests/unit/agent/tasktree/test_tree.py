"""Tests for TaskTree: add_child, pick_deepest_pending, mark_*, bubble_up, is_done, to_dict, from_dict."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.models import (
    ConstraintSet,
    FailureReport,
    NodeResult,
    TaskNode,
    TaskStatus,
    WorkspaceState,
)
from nanobot.agent.tasktree.tree import TaskTree


def _child_id(tree: TaskTree, parent_id: str, index: int) -> str:
    """Get the node_id of the index-th child of parent_id."""
    return tree.nodes[parent_id].children[index]


class TestTaskTree:
    def setup_method(self):
        self.tree = TaskTree()

    def test_create_root(self):
        self.tree.create_root(goal="root goal")
        assert self.tree.root_id == "root"
        root = self.tree.nodes["root"]
        assert root.goal == "root goal"
        assert root.depth == 0
        assert root.parent_id is None

    def test_add_child(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child 1")
        root = self.tree.nodes["root"]
        assert len(root.children) == 1
        child_id = root.children[0]
        child = self.tree.nodes[child_id]
        assert child.goal == "child 1"
        assert child.parent_id == "root"
        assert child.depth == 1

    def test_add_child_nested(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child 1")
        child_id = _child_id(self.tree, "root", 0)
        self.tree.add_child(child_id, "grandchild")
        gc_id = _child_id(self.tree, child_id, 0)
        gc = self.tree.nodes[gc_id]
        assert gc.depth == 2
        assert gc.parent_id == child_id

    def test_get_parent(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child")
        child_id = _child_id(self.tree, "root", 0)
        parent = self.tree.get_parent(child_id)
        assert parent is not None
        assert parent.id == "root"

    def test_get_parent_root(self):
        self.tree.create_root(goal="root")
        parent = self.tree.get_parent("root")
        assert parent is None

    def test_get_path(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child")
        child_id = _child_id(self.tree, "root", 0)
        self.tree.add_child(child_id, "grandchild")
        gc_id = _child_id(self.tree, child_id, 0)
        path = self.tree.get_path(gc_id)
        assert [n.id for n in path] == ["root", child_id, gc_id]

    def test_pick_deepest_pending_empty(self):
        assert self.tree.pick_deepest_pending() is None

    def test_pick_deepest_pending_single(self):
        self.tree.create_root(goal="root")
        assert self.tree.pick_deepest_pending() is not None
        assert self.tree.pick_deepest_pending().id == "root"

    def test_pick_deepest_pending_multiple(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child 1")
        self.tree.add_child("root", "child 2")
        child_1_id = _child_id(self.tree, "root", 0)
        self.tree.add_child(child_1_id, "grandchild")

        # Depth 2 should come first
        picked = self.tree.pick_deepest_pending()
        assert picked.depth == 2
        assert picked.parent_id == child_1_id

    def test_pick_deepest_pending_all_done(self):
        self.tree.create_root(goal="root")
        node = self.tree.pick_deepest_pending()
        self.tree.mark_done("root", NodeResult(node_id="root", summary="done"))
        assert self.tree.pick_deepest_pending() is None

    def test_mark_running(self):
        self.tree.create_root(goal="root")
        self.tree.mark_running("root")
        assert self.tree.nodes["root"].status == TaskStatus.RUNNING

    def test_mark_done(self):
        self.tree.create_root(goal="root")
        result = NodeResult(node_id="root", summary="done")
        self.tree.mark_done("root", result)
        assert self.tree.nodes["root"].status == TaskStatus.DONE
        assert self.tree.nodes["root"].result == result
        assert self.tree.nodes["root"].workspace_state == WorkspaceState.CLEAN

    def test_mark_done_with_workspace_state(self):
        self.tree.create_root(goal="root")
        result = NodeResult(node_id="root", summary="done", workspace_state=WorkspaceState.DIRTY)
        self.tree.mark_done("root", result)
        assert self.tree.nodes["root"].workspace_state == WorkspaceState.DIRTY

    def test_mark_failed(self):
        self.tree.create_root(goal="root")
        failure = FailureReport(node_id="root", summary="failed")
        self.tree.mark_failed("root", failure)
        assert self.tree.nodes["root"].status == TaskStatus.FAILED
        assert self.tree.nodes["root"].failure == failure

    def test_mark_blocked(self):
        self.tree.create_root(goal="root")
        self.tree.mark_blocked("root")
        assert self.tree.nodes["root"].status == TaskStatus.BLOCKED

    def test_mark_pending_resets(self):
        self.tree.create_root(goal="root")
        result = NodeResult(node_id="root", summary="done")
        self.tree.mark_done("root", result)
        self.tree.mark_pending("root")
        node = self.tree.nodes["root"]
        assert node.status == TaskStatus.PENDING
        assert node.result is None
        assert node.failure is None
        assert node.verification_failure is None
        assert node.workspace_state == WorkspaceState.CLEAN

    def test_increment_replan(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child")
        child_id = _child_id(self.tree, "root", 0)
        assert self.tree.nodes["root"].replan_count == 0
        self.tree.increment_replan(child_id)
        assert self.tree.nodes["root"].replan_count == 1

    def test_increment_replan_no_parent(self):
        self.tree.create_root(goal="root")
        self.tree.increment_replan("root")  # root has no parent
        # Should not raise

    def test_to_dict(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child")
        data = self.tree.to_dict()
        assert "root" in data
        assert len(data) == 2
        assert data["root"]["goal"] == "root"

    def test_from_dict(self):
        data = {
            "root": {
                "id": "root",
                "goal": "restored root",
                "status": "done",
                "parent_id": None,
                "children": [],
                "depth": 0,
                "result": None,
                "failure": None,
                "replan_count": 0,
                "verification_failure": None,
                "workspace_state": "clean",
                "created_at": 0.0,
            }
        }
        tree = TaskTree.from_dict(data)
        assert tree.root_id == "root"
        assert tree.nodes["root"].goal == "restored root"

    def test_from_dict_roundtrip(self):
        self.tree.create_root(goal="root goal")
        self.tree.add_child("root", "child goal")
        child_id = _child_id(self.tree, "root", 0)
        result = NodeResult(node_id=child_id, summary="child done", workspace_state=WorkspaceState.PARTIAL)
        self.tree.mark_done(child_id, result)

        data = self.tree.to_dict()
        restored = TaskTree.from_dict(data)

        assert restored.root_id == "root"
        assert len(restored.nodes) == 2
        assert restored.nodes[child_id].result is not None
        assert restored.nodes[child_id].result.workspace_state == WorkspaceState.PARTIAL

    def test_is_done(self):
        self.tree.create_root(goal="root")
        assert self.tree.is_done() is False
        self.tree.mark_done("root", NodeResult(node_id="root"))
        assert self.tree.is_done() is True

    def test_is_done_not_root(self):
        self.tree.create_root(goal="root")
        self.tree.add_child("root", "child")
        child_id = _child_id(self.tree, "root", 0)
        self.tree.mark_done(child_id, NodeResult(node_id=child_id))
        # Root not done yet
        assert self.tree.is_done() is False