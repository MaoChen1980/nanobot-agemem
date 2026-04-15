"""Integration tests for verification retry and replan flows."""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.tasktree.tree import TaskTree
from nanobot.agent.tasktree.scheduler import Scheduler, SchedulerConfig, VerificationResult
from nanobot.agent.tasktree.models import (
    ConstraintSet,
    FailureReport,
    NodeResult,
    TaskStatus,
    RootCause,
)
from nanobot.agent.tasktree.execution.subgoal import LLMSubgoalParser


class MockExecutionAgent:
    def __init__(self, results: dict[str, NodeResult | FailureReport]):
        self.results = results

    async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
        return self.results.get(node.id, FailureReport(
            node_id=node.id,
            summary="unknown",
            root_cause=RootCause.UNKNOWN,
        ))


class MockConstraintAgent:
    def __init__(self):
        self.constraints = ConstraintSet(max_depth=20)

    async def get_constraints(self, node, parent_result, root_goal):
        return self.constraints


@pytest.mark.asyncio
async def test_verification_fails_all_retries():
    """After max_verification_retries, scheduler stops retrying."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    agent = MockExecutionAgent({
        root_id: NodeResult(node_id=root_id, summary="done"),
    })

    class AlwaysFailVerifier:
        async def verify(self, root_goal, all_results):
            return VerificationResult(
                passed=False,
                failed_nodes=[root_id],
                reason="never good enough",
            )

    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=LLMSubgoalParser(),
        verification_agent=AlwaysFailVerifier(),
        config=SchedulerConfig(max_verification_retries=2),
    )

    final = await scheduler.run("root", resume_tree=tree)
    # Should return a result (not raise), even though verification always fails
    assert final is not None
    assert final.node_id == "root"


@pytest.mark.asyncio
async def test_replan_spawns_sibling_on_failure():
    """Child failure causes parent to spawn a replacement sibling."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    child_id = tree.add_child(root_id, "child_1")

    # Child fails but has remaining options
    failure_with_options = FailureReport(
        node_id=child_id,
        summary="child 1 failed",
        root_cause=RootCause.UNKNOWN,
        tried=["approach1"],
        remaining_options=["approach2"],
    )

    replan_count = 0

    class ReplanAgent:
        async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
            nonlocal replan_count
            if node.id == child_id:
                return failure_with_options
            if node.id == root_id:
                # Parent spawns sibling after child failure
                if replan_count == 0:
                    tree.add_child(root_id, "replacement_child")
                    replan_count += 1
                return NodeResult(node_id=root_id, summary="root ok")
            return NodeResult(node_id=node.id, summary="done")

    scheduler = Scheduler(
        execution_agent=ReplanAgent(),
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=LLMSubgoalParser(),
    )

    await scheduler.run("root", resume_tree=tree)

    # Root should be done (replacement child spawned and succeeded)
    assert tree.nodes[root_id].status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_bubble_up_when_all_children_terminal():
    """Parent becomes done when all children are done (bubble-up)."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    tree.add_child(root_id, "child_1")
    c1 = tree.nodes[root_id].children[0]
    tree.add_child(root_id, "child_2")
    c2 = tree.nodes[root_id].children[1]

    agent = MockExecutionAgent({
        root_id: NodeResult(node_id=root_id, summary="root done"),
        c1: NodeResult(node_id=c1, summary="child1 done"),
        c2: NodeResult(node_id=c2, summary="child2 done"),
    })

    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=LLMSubgoalParser(),
    )

    await scheduler.run("root", resume_tree=tree)

    assert tree.nodes[root_id].status == TaskStatus.DONE
    assert tree.nodes[c1].status == TaskStatus.DONE
    assert tree.nodes[c2].status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_constraint_veto_is_terminal():
    """Constraint veto failures are terminal and don't trigger replan."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    tree.add_child(root_id, "child_1")
    child_id = tree.nodes[root_id].children[0]

    class VetoAgent:
        async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
            if node.id == child_id:
                return FailureReport(
                    node_id=child_id,
                    summary="depth exceeded",
                    root_cause=RootCause.CONSTRAINT_VETO,
                    constraint_veto=True,
                )
            return NodeResult(node_id=node.id, summary="done")

    scheduler = Scheduler(
        execution_agent=VetoAgent(),
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=LLMSubgoalParser(),
    )

    await scheduler.run("root", resume_tree=tree)

    # Child should be BLOCKED (not failed)
    assert tree.nodes[child_id].status == TaskStatus.BLOCKED