"""Tests for TaskScheduler: depth-first, bubble_up, replan, verification retry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tasktree.models import (
    ConstraintSet,
    FailureReport,
    NodeResult,
    TaskNode,
    TaskStatus,
    RootCause,
    WorkspaceState,
)
from nanobot.agent.tasktree.scheduler import Scheduler, SchedulerConfig, VerificationResult
from nanobot.agent.tasktree.tree import TaskTree


def _child_id(tree: TaskTree, parent_id: str, index: int) -> str:
    """Get the node_id of the index-th child of parent_id."""
    return tree.nodes[parent_id].children[index]


class MockExecutionAgent:
    """Mock ExecutionAgent that returns configured results."""

    def __init__(self, results: dict[str, NodeResult | FailureReport]):
        self.results = results
        self.call_count = 0

    async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
        self.call_count += 1
        result = self.results.get(node.id)
        if result is None:
            return FailureReport(
                node_id=node.id,
                summary=f"No mock result for {node.id}",
                root_cause=RootCause.UNKNOWN,
            )
        return result


class MockConstraintAgent:
    def __init__(self, constraints: ConstraintSet | None = None):
        self.constraints = constraints or ConstraintSet(max_depth=20)

    async def get_constraints(self, node, parent_result, root_goal):
        return self.constraints


class MockSubgoalParser:
    def __init__(self, children_map: dict[str, list[str]]):
        self.children_map = children_map

    def parse(self, result: NodeResult) -> list[str]:
        return self.children_map.get(result.node_id, [])


@pytest.mark.asyncio
async def test_scheduler_simple_success():
    """Root node completes successfully with no children."""
    tree = TaskTree()
    tree.create_root(goal="root task")

    result = NodeResult(node_id="root", summary="done")
    agent = MockExecutionAgent({"root": result})
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser({}),
    )

    final = await scheduler.run("root task", resume_tree=tree)
    assert final.summary == "done"


@pytest.mark.asyncio
async def test_scheduler_run_returns_root_result():
    """scheduler.run() should return the root node's result."""
    tree = TaskTree()
    tree.create_root(goal="my task")

    result = NodeResult(node_id="root", summary="task completed successfully")
    agent = MockExecutionAgent({"root": result})
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser({}),
    )

    final = await scheduler.run("my task", resume_tree=tree)
    assert final.node_id == "root"
    assert final.summary == "task completed successfully"


@pytest.mark.asyncio
async def test_scheduler_resume_tree():
    """Resume from existing tree state."""
    tree = TaskTree()
    tree.create_root(goal="root")
    tree.mark_done("root", NodeResult(node_id="root", summary="already done"))

    agent = MockExecutionAgent({})
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser({}),
    )

    final = await scheduler.run("root", resume_tree=tree)
    assert final.summary == "already done"


@pytest.mark.asyncio
async def test_verification_failure_triggers_retry():
    """Verification failure should mark nodes pending and retry."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    results = {
        root_id: NodeResult(node_id=root_id, summary="done"),
    }

    agent = MockExecutionAgent(results)

    class FailOnceVerifier:
        call_count = 0

        async def verify(self, root_goal, all_results):
            FailOnceVerifier.call_count += 1
            if FailOnceVerifier.call_count == 1:
                return VerificationResult(
                    passed=False,
                    failed_nodes=[root_id],
                    reason="first attempt failed",
                )
            return VerificationResult(passed=True, failed_nodes=[], reason="ok")

    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser({}),
        verification_agent=FailOnceVerifier(),
        config=SchedulerConfig(max_verification_retries=3),
    )

    await scheduler.run("root")
    assert FailOnceVerifier.call_count == 2


@pytest.mark.asyncio
async def test_verification_passes_without_retry():
    """Verification passes without retry when all good."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    results = {
        root_id: NodeResult(node_id=root_id, summary="done"),
    }

    agent = MockExecutionAgent(results)

    class AlwaysPassVerifier:
        async def verify(self, root_goal, all_results):
            return VerificationResult(passed=True, failed_nodes=[], reason="all good")

    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser({}),
        verification_agent=AlwaysPassVerifier(),
    )

    final = await scheduler.run("root", resume_tree=tree)
    # Verify the returned result indicates success
    assert final.summary == "done"


@pytest.mark.asyncio
async def test_scheduler_mark_done_sets_result():
    """mark_done properly sets the node result and status."""
    tree = TaskTree()
    tree.create_root(goal="root")
    result = NodeResult(node_id="root", summary="done", workspace_state=WorkspaceState.PARTIAL)
    tree.mark_done("root", result)
    assert tree.nodes["root"].status == TaskStatus.DONE
    assert tree.nodes["root"].result.summary == "done"
    assert tree.nodes["root"].workspace_state == WorkspaceState.PARTIAL


@pytest.mark.asyncio
async def test_scheduler_mark_pending_resets_state():
    """mark_pending properly resets node state for retry."""
    tree = TaskTree()
    tree.create_root(goal="root")
    result = NodeResult(node_id="root", summary="done")
    tree.mark_done("root", result)
    tree.mark_pending("root")
    node = tree.nodes["root"]
    assert node.status == TaskStatus.PENDING
    assert node.result is None
    assert node.verification_failure is None
    assert node.workspace_state == WorkspaceState.CLEAN


@pytest.mark.asyncio
async def test_scheduler_increment_replan():
    """increment_replan increments parent's replan counter."""
    tree = TaskTree()
    tree.create_root(goal="root")
    tree.add_child("root", "child")
    child_id = _child_id(tree, "root", 0)
    assert tree.nodes["root"].replan_count == 0
    tree.increment_replan(child_id)
    assert tree.nodes["root"].replan_count == 1


@pytest.mark.asyncio
async def test_scheduler_mark_blocked():
    """mark_blocked sets constraint_veto terminal state."""
    tree = TaskTree()
    tree.create_root(goal="root")
    tree.mark_blocked("root")
    assert tree.nodes["root"].status == TaskStatus.BLOCKED
    assert tree.nodes["root"].is_terminal() is True