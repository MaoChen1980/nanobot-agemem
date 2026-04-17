"""TaskTree: the core tree data structure."""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Iterator

from nanobot.agent.tasktree.models import (
    FailureReport,
    NodeResult,
    TaskNode,
    TaskStatus,
    WorkspaceState,
)

if TYPE_CHECKING:
    from nanobot.agent.tasktree.models import ConstraintSet


class TaskTree:
    """A tree of TaskNodes.

    The tree structure is immutable in terms of node deletion — nodes are only ever added.
    The scheduler drives execution by iterating over the tree.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, TaskNode] = {}
        self.root_id: str | None = None
        self._id_counter: Iterator[int] = itertools.count()

    # -------------------------------------------------------------------------
    # Tree construction
    # -------------------------------------------------------------------------

    def create_root(self, goal: str) -> TaskNode:
        """Create the root node. Must be called before any other nodes."""
        if self.root_id is not None:
            raise ValueError("Root node already exists")
        root = TaskNode(id="root", goal=goal, parent_id=None, depth=0)
        self.nodes["root"] = root
        self.root_id = "root"
        return root

    def add_child(self, parent_id: str, goal: str) -> TaskNode:
        """Create a new child node under parent_id.

        The new node's id is "{parent_id}.{N}" where N increments per parent.
        """
        parent = self.nodes[parent_id]
        child_index = len(parent.children)
        node_id = f"{parent_id}.{child_index}"
        child = TaskNode(
            id=node_id,
            goal=goal,
            parent_id=parent_id,
            depth=parent.depth + 1,
        )
        self.nodes[node_id] = child
        parent.children.append(node_id)
        return child

    # -------------------------------------------------------------------------
    # Tree traversal
    # -------------------------------------------------------------------------

    def get_node(self, node_id: str) -> TaskNode:
        """Return the node with the given id."""
        return self.nodes[node_id]

    def get_parent(self, node_id: str) -> TaskNode | None:
        """Return the parent of node_id, or None if node_id is the root."""
        node = self.nodes[node_id]
        if node.parent_id is None:
            return None
        return self.nodes[node.parent_id]

    def get_path(self, node_id: str) -> list[TaskNode]:
        """Return the path from root to node_id (inclusive), as a list."""
        path: list[TaskNode] = []
        current: TaskNode | None = self.nodes[node_id]
        while current is not None:
            path.append(current)
            current = self.get_parent(current.id)
        path.reverse()
        return path

    def pick_deepest_pending(self) -> TaskNode | None:
        """Pick the deepest pending node using depth-first order.

        Returns the left-most (first-added) pending node at the greatest depth.
        This drives the depth-first execution order.
        """
        if self.root_id is None:
            return None

        pending = [n for n in self.nodes.values() if n.status == TaskStatus.PENDING]
        if not pending:
            return None

        max_depth = max(n.depth for n in pending)
        at_max_depth = [n for n in pending if n.depth == max_depth]
        # Left-most = lowest child index in parent.children list
        at_max_depth.sort(key=lambda n: self._child_index(n))
        return at_max_depth[0]

    def _child_index(self, node: TaskNode) -> int:
        """Return the index of node in its parent's children list."""
        if node.parent_id is None:
            return 0
        parent = self.nodes[node.parent_id]
        return parent.children.index(node.id)

    # -------------------------------------------------------------------------
    # Status updates
    # -------------------------------------------------------------------------

    def mark_running(self, node_id: str) -> None:
        """Mark a node as running."""
        node = self.nodes[node_id]
        node.status = TaskStatus.RUNNING

    def mark_done(self, node_id: str, result: NodeResult) -> None:
        """Mark a node as done with its result."""
        node = self.nodes[node_id]
        node.status = TaskStatus.DONE
        node.result = result
        if result and result.workspace_state is not None:
            node.workspace_state = result.workspace_state

    def mark_failed(self, node_id: str, failure: FailureReport) -> None:
        """Mark a node as failed with its failure report."""
        node = self.nodes[node_id]
        node.status = TaskStatus.FAILED
        node.failure = failure

    def mark_blocked(self, node_id: str) -> None:
        """Mark a node as blocked (constraint veto)."""
        node = self.nodes[node_id]
        node.status = TaskStatus.BLOCKED

    def mark_wait_info(self, node_id: str) -> None:
        """Mark a node as waiting for user input."""
        node = self.nodes[node_id]
        node.status = TaskStatus.WAIT_INFO

    def mark_pending(self, node_id: str) -> None:
        """Reset a node to pending status for retry (after verification failure)."""
        node = self.nodes[node_id]
        node.status = TaskStatus.PENDING
        node.result = None
        node.failure = None
        node.verification_failure = None
        node.workspace_state = WorkspaceState.CLEAN

    def increment_replan(self, node_id: str) -> None:
        """Increment the replan counter for the parent of node_id."""
        parent = self.get_parent(node_id)
        if parent is not None:
            parent.replan_count += 1

    # -------------------------------------------------------------------------
    # Completion checks
    # -------------------------------------------------------------------------

    def is_done(self) -> bool:
        """Return True if all nodes are in a terminal state."""
        if not self.nodes:
            return False
        return all(n.is_terminal() for n in self.nodes.values())

    def get_root_result(self) -> NodeResult | None:
        """Return the result of the root node, if it is done."""
        if self.root_id is None:
            return None
        root = self.nodes[self.root_id]
        return root.result

    def get_all_results(self) -> dict[str, NodeResult]:
        """Return a dict of all done node_id -> NodeResult."""
        return {
            node_id: node.result
            for node_id, node in self.nodes.items()
            if node.result is not None
        }

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Serialize the full tree to a dict (for session persistence)."""
        return {
            node_id: {
                "id": node.id,
                "goal": node.goal,
                "status": node.status.value,
                "parent_id": node.parent_id,
                "children": node.children,
                "depth": node.depth,
                "result": node.result.to_dict() if node.result else None,
                "failure": node.failure.to_dict() if node.failure else None,
                "replan_count": node.replan_count,
                "verification_failure": node.verification_failure,
                "workspace_state": node.workspace_state.value,
                "created_at": node.created_at,
            }
            for node_id, node in self.nodes.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, object]]) -> TaskTree:
        """Reconstruct a TaskTree from a dict."""
        tree = cls()
        for node_id, d in data.items():
            node = TaskNode(
                id=d["id"],
                goal=d["goal"],
                status=TaskStatus(d["status"]),
                parent_id=d["parent_id"],
                children=d["children"],
                depth=d["depth"],
                result=NodeResult.from_dict(d["result"]) if d["result"] else None,
                failure=FailureReport.from_dict(d["failure"]) if d["failure"] else None,
                replan_count=d["replan_count"],
                verification_failure=d.get("verification_failure"),
                workspace_state=WorkspaceState(d.get("workspace_state", "clean")),
                created_at=d.get("created_at", time.time()),
            )
            tree.nodes[node_id] = node
            if node.parent_id is None:
                tree.root_id = node_id
        return tree
