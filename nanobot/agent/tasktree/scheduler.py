"""TaskScheduler: drives the depth-first tree execution loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from loguru import logger

from nanobot.agent.tasktree.models import (
    ConstraintSet,
    FailureReport,
    NodeResult,
    RootCause,
    TaskNode,
    TaskStatus,
    WorkspaceState,
)
from nanobot.agent.tasktree.tree import TaskTree


MAX_CHILDREN = 10


# ---------------------------------------------------------------------------
# Agent Protocols
# ---------------------------------------------------------------------------


class ExecutionAgent(Protocol):
    """Execute a single TaskNode and return either a result or a failure."""

    async def execute(
        self,
        node: TaskNode,
        constraints: ConstraintSet,
        parent_result: NodeResult | None,
        root_goal: str,
        root_goal_context: str,
    ) -> NodeResult | FailureReport:
        """Execute node and return NodeResult on success, FailureReport on failure."""
        ...


class ConstraintAgent(Protocol):
    """Provide execution constraints for a node."""

    async def get_constraints(
        self,
        node: TaskNode,
        parent_result: NodeResult | None,
        root_goal: str,
    ) -> ConstraintSet:
        """Return the constraint set for this node."""
        ...


class SubgoalParser(Protocol):
    """Parse the output of an ExecutionAgent to extract child goals."""

    def parse(self, result: NodeResult) -> list[str]:
        """Return list of child goal strings from a successful NodeResult."""
        ...


class SchedulerCallbacks(Protocol):
    """Optional callbacks for external observability."""

    def on_node_start(self, node: TaskNode) -> None: ...
    def on_node_done(self, node: TaskNode, result: NodeResult) -> None: ...
    def on_node_failed(self, node: TaskNode, failure: FailureReport) -> None: ...
    def on_node_blocked(self, node: TaskNode, failure: FailureReport) -> None: ...
    def save_checkpoint(self, tree: TaskTree) -> None: ...


@dataclass
class VerificationResult:
    """Result of a verification pass."""

    passed: bool
    failed_nodes: list[str] = field(default_factory=list)
    reason: str = ""
    evidence: list[str] = field(default_factory=list)


class VerificationAgent(Protocol):
    """Verify whether execution results satisfy the root goal."""

    async def verify(
        self,
        root_goal: str,
        results: dict[str, NodeResult],
    ) -> "VerificationResult":
        """Return VerificationResult indicating pass/fail and failed node ids."""
        ...


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    max_children: int = MAX_CHILDREN
    max_depth: int = 20
    max_verification_retries: int = 3  # Max verification failure cycles before giving up


@dataclass
class Scheduler:
    """Drives depth-first execution of a TaskTree.

    The scheduler is a pure state machine — it does not execute tasks itself.
    It coordinates ExecutionAgent, ConstraintAgent, and SubgoalParser.

    Execution flow per node:
        1. Pick deepest pending node (depth-first)
        2. Get constraints from ConstraintAgent
        3. Execute node via ExecutionAgent
        4. On success: mark done, spawn children (if any) from parsed subgoals
        5. On failure (constraint_veto): mark blocked, propagate to parent
        6. On failure (replan): let parent spawn sibling, continue DFS
    """

    config: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Injected dependencies
    execution_agent: ExecutionAgent | None = None
    constraint_agent: ConstraintAgent | None = None
    subgoal_parser: SubgoalParser | None = None
    callbacks: SchedulerCallbacks | None = None
    verification_agent: VerificationAgent | None = None

    # Internal state
    _tree: TaskTree | None = None
    _root_goal: str = ""

    @property
    def tree(self) -> TaskTree:
        if self._tree is None:
            raise RuntimeError("Scheduler has not been started")
        return self._tree

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(self, root_goal: str, resume_tree: TaskTree | None = None) -> NodeResult:
        """Run the task tree to completion and return the root's result.

        If resume_tree is provided, execution continues from that tree's state
        (for crash recovery or manual cancellation).
        """
        self._root_goal = root_goal
        if resume_tree is not None:
            self._tree = resume_tree
            logger.info("Scheduler resuming with existing tree, root status: {}",
                self._tree.nodes[self._tree.root_id].status.value
                if self._tree.root_id else "no root")
        else:
            self._tree = TaskTree()
            self._tree.create_root(goal=root_goal)
            logger.info("Scheduler started with root goal: {}", root_goal[:80])

        verification_retries = 0

        while True:
            # -----------------------------------------------------------------
            # Execute pending nodes until tree is complete
            # -----------------------------------------------------------------
            while not self._is_tree_complete():
                node = self._tree.pick_deepest_pending()
                if node is None:
                    logger.warning(
                        "Scheduler: pick_deepest_pending returned None but tree incomplete. "
                        "Breaking loop. Root status: {}",
                        self._tree.nodes[self._tree.root_id].status.value
                        if self._tree.root_id
                        else "no root",
                    )
                    break

                await self._execute_node(node)

            # -----------------------------------------------------------------
            # Verification phase (only when all nodes are terminal)
            # -----------------------------------------------------------------
            if self.verification_agent is None:
                break

            if verification_retries >= self.config.max_verification_retries:
                logger.warning(
                    "Scheduler: max verification retries ({}) reached",
                    self.config.max_verification_retries,
                )
                break

            all_results = {
                nid: n.result
                for nid, n in self._tree.nodes.items()
                if n.result is not None
            }
            vr = await self.verification_agent.verify(self._root_goal, all_results)

            if vr.passed or not vr.failed_nodes:
                break  # Verification passed or no failed nodes to retry

            verification_retries += 1
            logger.info(
                "Verification failed (attempt {}/{}): {}",
                verification_retries,
                self.config.max_verification_retries,
                vr.reason,
            )

            # Reset failed nodes to pending and wake up parents for retry
            for node_id in vr.failed_nodes:
                if node_id not in self._tree.nodes:
                    continue
                node = self._tree.nodes[node_id]
                node.verification_failure = vr.reason
                self._tree.mark_pending(node_id)
                logger.debug(
                    "Verification failed node {} marked pending for retry",
                    node_id,
                )
                # Parent of a failed node needs to re-enter RUNNING state so
                # bubble-up does not treat it as terminal
                if node.parent_id is not None:
                    parent = self._tree.nodes[node.parent_id]
                    if parent.status == TaskStatus.DONE:
                        parent.status = TaskStatus.RUNNING

            # Loop continues to retry failed nodes
            continue

        root_result = self._tree.get_root_result()
        if root_result is None:
            # Tree is done but root has no result — return empty result
            root_result = NodeResult(
                node_id="root",
                summary="Task completed with no result",
            )
            self._tree.nodes["root"].result = root_result

        logger.info("Scheduler finished")
        return root_result

    def get_tree(self) -> TaskTree:
        """Return the current task tree (for inspection or persistence)."""
        return self._tree

    # -------------------------------------------------------------------------
    # Core execution step
    # -------------------------------------------------------------------------

    async def _execute_node(self, node: TaskNode) -> None:
        """Execute a single node end-to-end."""
        if node.parent_id is None:
            parent_result: NodeResult | None = None
        else:
            parent_node = self._tree.get_node(node.parent_id)
            parent_result = parent_node.result

        # 1. Pick deepest pending node → mark running
        self._tree.mark_running(node.id)
        logger.debug("Executing node {}", node.id)
        if self.callbacks:
            self.callbacks.on_node_start(node)

        # 2. Get constraints
        constraints = await self._get_constraints(node, parent_result)

        # 3. Check depth constraint
        if node.depth > constraints.max_depth:
            failure = FailureReport(
                node_id=node.id,
                status=TaskStatus.FAILED,
                root_cause=RootCause.CONSTRAINT_VETO,
                summary=f"Node depth {node.depth} exceeds max_depth {constraints.max_depth}",
                constraint_veto=True,
                workspace_state=WorkspaceState.CLEAN,
            )
            await self._handle_failure(node, failure)
            await self._bubble_up(node)
            return

        # 4. Execute via agent
        result_or_failure = await self.execution_agent.execute(
            node=node,
            constraints=constraints,
            parent_result=parent_result,
            root_goal=self._root_goal,
            root_goal_context=self._build_root_context(),
            tree=self._tree,
        )

        # 5. Route result
        if isinstance(result_or_failure, NodeResult):
            await self._handle_success(node, result_or_failure)
            await self._bubble_up(node)
        else:
            await self._handle_failure(node, result_or_failure)
            await self._bubble_up(node)

        # 6. Save checkpoint after every node completion
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Save current tree state if the callback supports it."""
        if self.callbacks is not None:
            cb = self.callbacks
            if hasattr(cb, 'save_checkpoint'):
                cb.save_checkpoint(self._tree)

    async def _get_constraints(
        self, node: TaskNode, parent_result: NodeResult | None
    ) -> ConstraintSet:
        if self.constraint_agent is None:
            return ConstraintSet(
                max_depth=self.config.max_depth,
                forbidden_actions=[],
                failure_count_limit=2,
            )
        return await self.constraint_agent.get_constraints(
            node=node,
            parent_result=parent_result,
            root_goal=self._root_goal,
        )

    # -------------------------------------------------------------------------
    # Success path
    # -------------------------------------------------------------------------

    async def _handle_success(self, node: TaskNode, result: NodeResult) -> None:
        """Handle a successful node execution."""
        self._tree.mark_done(node.id, result)
        logger.debug("Node {} done: {}", node.id, result.summary[:80])
        if self.callbacks:
            self.callbacks.on_node_done(node, result)

        # Try to spawn child nodes from the result
        if self.subgoal_parser is not None:
            child_goals = self.subgoal_parser.parse(result)
            for goal in child_goals[: self.config.max_children]:
                self._tree.add_child(node.id, goal)

        # After spawning (or not), continue DFS.
        # If there are new pending children, the next pick_deepest_pending() will get them.
        # Otherwise, bubble up.

    # -------------------------------------------------------------------------
    # Failure path
    # -------------------------------------------------------------------------

    async def _handle_failure(self, node: TaskNode, failure: FailureReport) -> None:
        """Handle a failed node."""
        if failure.constraint_veto:
            self._tree.mark_blocked(node.id)
            logger.debug("Node {} blocked (constraint veto)", node.id)
            if self.callbacks:
                self.callbacks.on_node_blocked(node, failure)
            # Constraint veto is terminal — do not replan, propagate blocked upward
            return

        # Non-veto failure
        parent = self._tree.get_parent(node.id)
        self._tree.mark_failed(node.id, failure)
        logger.debug("Node {} failed: {}", node.id, failure.summary[:80])
        if self.callbacks:
            self.callbacks.on_node_failed(node, failure)

        if parent is None:
            # Root node failed — nothing to propagate to
            logger.warning("Root node {} failed: {}", node.id, failure.summary)
            return

        # Increment parent's replan counter
        self._tree.increment_replan(node.id)

        # Check if parent can still spawn a replacement sibling
        if len(parent.children) < self.config.max_children:
            # Spawn a replacement sibling using remaining options from the failure
            replacement_goal = self._spawn_replacement_goal(failure)
            self._tree.add_child(parent.id, replacement_goal)
            logger.debug(
                "Node {} failed, spawned replacement under parent {}, replan count {}",
                node.id,
                parent.id,
                parent.replan_count,
            )
        else:
            # Parent at MAX_CHILDREN — mark parent as failed and block its pending children
            parent_failure = FailureReport(
                node_id=parent.id,
                status=TaskStatus.FAILED,
                root_cause=RootCause.MAX_REPLAN_REACHED,
                summary=f"Child {node.id} failed: {failure.summary}; parent at max children",
                tried=[f"Child {node.id} failed"],
                remaining_options=[],
                constraint_veto=False,
                workspace_state=failure.workspace_state,
            )
            self._tree.mark_failed(parent.id, parent_failure)
            # Block all pending children of the newly-failed parent
            for child_id in parent.children:
                child = self._tree.get_node(child_id)
                if child.status == TaskStatus.PENDING:
                    self._tree.mark_blocked(child_id)
            # Propagate failure up to grandparent
            grandparent = self._tree.get_parent(parent.id)
            if grandparent is not None:
                await self._handle_failure(grandparent, parent_failure)

    def _spawn_replacement_goal(self, failure: FailureReport) -> str:
        """Build a replacement goal from a failure report."""
        if failure.remaining_options:
            return failure.remaining_options[0]
        return f"Retry: {failure.summary}"

    # -------------------------------------------------------------------------
    # Bubble up: propagate terminal states to ancestors
    # -------------------------------------------------------------------------

    async def _bubble_up(self, completed_node: TaskNode) -> None:
        """Check if the parent of completed_node needs to be updated.

        Called after every node completion (success or failure).
        Propagates state upward only when a parent has no more pending children.
        """
        parent_id = completed_node.parent_id
        while parent_id is not None:
            parent = self._tree.get_node(parent_id)
            # If parent is already terminal, stop bubbling
            if parent.is_terminal():
                return
            # Check if all of parent's children are terminal
            all_terminal = all(
                self._tree.get_node(cid).is_terminal()
                for cid in parent.children
            )
            if not all_terminal:
                # Parent still has pending children — nothing to propagate yet
                return
            # All children done — evaluate parent's terminal state
            child_statuses = [self._tree.get_node(cid).status for cid in parent.children]
            if TaskStatus.FAILED in child_statuses or TaskStatus.BLOCKED in child_statuses:
                # At least one child failed/blocked — parent is failed
                parent_failure = FailureReport(
                    node_id=parent.id,
                    status=TaskStatus.FAILED,
                    root_cause=RootCause.NO_REMAINING_OPTIONS,
                    summary=f"Child failure — all children terminal: {child_statuses}",
                    tried=[],
                    remaining_options=[],
                    constraint_veto=False,
                    workspace_state=WorkspaceState.CLEAN,
                )
                self._tree.mark_failed(parent.id, parent_failure)
            else:
                # All children succeeded — parent is done
                child_results = [
                    self._tree.get_node(cid).result
                    for cid in parent.children
                    if self._tree.get_node(cid).result is not None
                ]
                combined_summary = "; ".join(
                    r.summary for r in child_results if r
                )
                parent_result = NodeResult(
                    node_id=parent.id,
                    summary=combined_summary or "Completed",
                )
                self._tree.mark_done(parent.id, parent_result)
                # Spawn children for the now-done parent
                if self.subgoal_parser is not None:
                    child_goals = self.subgoal_parser.parse(parent_result)
                    for goal in child_goals[: self.config.max_children]:
                        self._tree.add_child(parent.id, goal)
            # Continue bubbling up
            parent_id = parent.parent_id

    # -------------------------------------------------------------------------
    # Completion check
    # -------------------------------------------------------------------------

    def _is_tree_complete(self) -> bool:
        """Return True when the root node is in a terminal state."""
        if self._tree is None or self._tree.root_id is None:
            return False
        root = self._tree.nodes[self._tree.root_id]
        return root.is_terminal()

    # -------------------------------------------------------------------------
    # Context building helpers
    # -------------------------------------------------------------------------

    def _build_root_context(self) -> str:
        """Build a root context block (used as anchor in all node contexts)."""
        return (
            f"[Root Goal]\n{self._root_goal}\n"
            f"[End Root Goal]"
        )
