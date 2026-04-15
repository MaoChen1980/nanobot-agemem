"""Integration tests for replan, bubble-up failure propagation, and MAX_CHILDREN boundary."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.models import (
    ConstraintSet,
    FailureReport,
    NodeResult,
    TaskStatus,
    RootCause,
)
from nanobot.agent.tasktree.scheduler import Scheduler, SchedulerConfig, MAX_CHILDREN
from nanobot.agent.tasktree.tree import TaskTree


class MockExecutionAgent:
    def __init__(self, results: dict[str, NodeResult | FailureReport]):
        self.results = results

    async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
        return self.results.get(
            node.id,
            FailureReport(
                node_id=node.id,
                summary="unknown",
                root_cause=RootCause.UNKNOWN,
            ),
        )


class MockConstraintAgent:
    def __init__(self):
        self.constraints = ConstraintSet(max_depth=20)

    async def get_constraints(self, node, parent_result, root_goal):
        return self.constraints


class MockSubgoalParser:
    def parse(self, result: NodeResult) -> list[str]:
        return []


def _child_id(tree: TaskTree, parent_id: str, index: int) -> str:
    return tree.nodes[parent_id].children[index]


@pytest.mark.asyncio
async def test_max_children_constant_is_10():
    """MAX_CHILDREN should be 10."""
    assert MAX_CHILDREN == 10


@pytest.mark.asyncio
async def test_scheduler_config_max_children_defaults_to_10():
    """SchedulerConfig.max_children should default to MAX_CHILDREN."""
    config = SchedulerConfig()
    assert config.max_children == MAX_CHILDREN == 10


@pytest.mark.asyncio
async def test_handle_failure_parent_at_max_children_fails_parent():
    """When a child fails and parent is at MAX_CHILDREN, parent is marked failed."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    # Fill parent to max children (10)
    for i in range(10):
        tree.add_child(root_id, f"child_{i}")
    child_ids = list(tree.nodes[root_id].children)

    # Mark all children done except the last one
    for cid in child_ids[:-1]:
        tree.mark_done(cid, NodeResult(node_id=cid, summary=f"{cid} done"))

    # Last child fails (simulate by marking it failed before scheduler runs)
    # But we need to test the scheduler's _handle_failure path
    # Let's set up: child_ids[-1] fails and parent is at max_children
    agent = MockExecutionAgent({
        child_ids[-1]: FailureReport(
            node_id=child_ids[-1],
            summary="child failed",
            root_cause=RootCause.UNKNOWN,
            remaining_options=[],
        ),
        root_id: NodeResult(node_id=root_id, summary="root done"),
    })

    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        config=SchedulerConfig(max_children=10),
    )

    await scheduler.run("root", resume_tree=tree)

    # Parent should be failed (because child failed and parent at MAX_CHILDREN)
    assert tree.nodes[root_id].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_bubble_up_failed_child_propagates_to_parent():
    """When all children terminal with FAILED/BLOCKED, parent becomes FAILED."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    # Add two children
    tree.add_child(root_id, "child_1")
    c1 = _child_id(tree, root_id, 0)
    tree.add_child(root_id, "child_2")
    c2 = _child_id(tree, root_id, 1)

    # Both children fail
    tree.mark_failed(c1, FailureReport(
        node_id=c1, status=TaskStatus.FAILED,
        root_cause=RootCause.UNKNOWN, summary="failed 1",
    ))
    tree.mark_failed(c2, FailureReport(
        node_id=c2, status=TaskStatus.FAILED,
        root_cause=RootCause.UNKNOWN, summary="failed 2",
    ))

    agent = MockExecutionAgent({})
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
    )

    await scheduler.run("root", resume_tree=tree)

    # Root should be FAILED via bubble-up
    assert tree.nodes[root_id].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_bubble_up_continues_to_grandparent():
    """After parent fails, failure continues to bubble up to grandparent."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    # root -> parent -> child
    tree.add_child(root_id, "parent_task")
    parent = _child_id(tree, root_id, 0)
    tree.add_child(parent, "child_task")
    child = _child_id(tree, parent, 0)

    # Child fails, parent has no remaining options and at MAX_CHILDREN
    # Simulate parent at MAX_CHILDREN with no room for siblings
    agent = MockExecutionAgent({
        parent: FailureReport(
            node_id=parent,
            summary="parent failed",
            root_cause=RootCause.MAX_REPLAN_REACHED,
            remaining_options=[],
        ),
        root_id: NodeResult(node_id=root_id, summary="root done"),
    })

    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
    )

    await scheduler.run("root", resume_tree=tree)

    # Root should be failed (MAX_REPLAN_REACHED propagated)
    assert tree.nodes[root_id].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_spawn_replacement_goal_uses_remaining_options():
    """_spawn_replacement_goal uses failure.remaining_options if available."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id
    tree.add_child(root_id, "child_1")
    c1 = _child_id(tree, root_id, 0)

    # Track which node is the replacement
    replacement_id = None

    class TrackingAgent:
        def __init__(self):
            self.results = {
                c1: FailureReport(
                    node_id=c1,
                    summary="child 1 failed",
                    root_cause=RootCause.UNKNOWN,
                    remaining_options=["approach_2", "approach_3"],
                ),
                root_id: NodeResult(node_id=root_id, summary="root done"),
            }

        async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
            nonlocal replacement_id
            # The replacement child was just added - store its ID and give it a success result
            if node.id not in self.results:
                replacement_id = node.id
                self.results[node.id] = NodeResult(node_id=node.id, summary=f"{node.id} done")
            return self.results[node.id]

    scheduler = Scheduler(
        execution_agent=TrackingAgent(),
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        config=SchedulerConfig(max_children=11),
    )

    await scheduler.run("root", resume_tree=tree)

    # Should have spawned "approach_2" as replacement
    children = tree.nodes[root_id].children
    assert len(children) == 2  # original child + replacement
    assert replacement_id is not None
    assert tree.nodes[replacement_id].goal == "approach_2"


@pytest.mark.asyncio
async def test_spawn_replacement_goal_falls_back_to_retry():
    """_spawn_replacement_goal falls back to 'Retry: <summary>' when no remaining options."""
    scheduler = Scheduler()

    failure = FailureReport(
        node_id="n1",
        summary="something went wrong",
        root_cause=RootCause.UNKNOWN,
        remaining_options=[],
    )

    goal = scheduler._spawn_replacement_goal(failure)
    assert goal == "Retry: something went wrong"


@pytest.mark.asyncio
async def test_constraint_veto_marks_blocked_no_replan():
    """Constraint veto failures mark node as BLOCKED and do not trigger replan."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id
    tree.add_child(root_id, "child_1")
    c1 = _child_id(tree, root_id, 0)

    agent = MockExecutionAgent({
        c1: FailureReport(
            node_id=c1,
            summary="depth exceeded",
            root_cause=RootCause.CONSTRAINT_VETO,
            constraint_veto=True,
        ),
        root_id: NodeResult(node_id=root_id, summary="root done"),
    })

    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
    )

    await scheduler.run("root", resume_tree=tree)

    # Child should be BLOCKED (not failed)
    assert tree.nodes[c1].status == TaskStatus.BLOCKED
    # No replacement should be spawned for BLOCKED nodes
    assert len(tree.nodes[root_id].children) == 1


@pytest.mark.asyncio
async def test_callback_on_node_start_called():
    """on_node_start callback is invoked before node execution."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    start_calls = []

    class CountingCallback:
        def on_node_start(self, node):
            start_calls.append(node.id)

        def on_node_done(self, node, result):
            pass

        def on_node_failed(self, node, failure):
            pass

        def on_node_blocked(self, node, failure):
            pass

        def save_checkpoint(self, tree):
            pass

    agent = MockExecutionAgent({root_id: NodeResult(node_id=root_id, summary="done")})
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        callbacks=CountingCallback(),
    )

    await scheduler.run("root", resume_tree=tree)

    assert root_id in start_calls


@pytest.mark.asyncio
async def test_callback_on_node_done_called():
    """on_node_done callback is invoked after successful node execution."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    done_calls = []

    class CountingCallback:
        def on_node_start(self, node):
            pass

        def on_node_done(self, node, result):
            done_calls.append((node.id, result.summary))

        def on_node_failed(self, node, failure):
            pass

        def on_node_blocked(self, node, failure):
            pass

        def save_checkpoint(self, tree):
            pass

    agent = MockExecutionAgent({root_id: NodeResult(node_id=root_id, summary="done ok")})
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        callbacks=CountingCallback(),
    )

    await scheduler.run("root", resume_tree=tree)

    assert (root_id, "done ok") in done_calls


@pytest.mark.asyncio
async def test_callback_save_checkpoint_called():
    """save_checkpoint callback is invoked after every node completion."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    checkpoint_calls = []

    class CountingCallback:
        def on_node_start(self, node):
            pass

        def on_node_done(self, node, result):
            pass

        def on_node_failed(self, node, failure):
            pass

        def on_node_blocked(self, node, failure):
            pass

        def save_checkpoint(self, tree):
            checkpoint_calls.append(tree)

    agent = MockExecutionAgent({root_id: NodeResult(node_id=root_id, summary="done")})
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        callbacks=CountingCallback(),
    )

    await scheduler.run("root", resume_tree=tree)

    assert len(checkpoint_calls) >= 1


@pytest.mark.asyncio
async def test_callback_on_node_failed_called():
    """on_node_failed callback is invoked after node failure."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    failed_calls = []

    class CountingCallback:
        def on_node_start(self, node):
            pass

        def on_node_done(self, node, result):
            pass

        def on_node_failed(self, node, failure):
            failed_calls.append(node.id)

        def on_node_blocked(self, node, failure):
            pass

        def save_checkpoint(self, tree):
            pass

    # Root fails
    agent = MockExecutionAgent({
        root_id: FailureReport(
            node_id=root_id,
            summary="root failed",
            root_cause=RootCause.UNKNOWN,
        ),
    })
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        callbacks=CountingCallback(),
    )

    await scheduler.run("root", resume_tree=tree)

    assert root_id in failed_calls


@pytest.mark.asyncio
async def test_callback_on_node_blocked_called():
    """on_node_blocked callback is invoked after constraint veto."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    blocked_calls = []

    class CountingCallback:
        def on_node_start(self, node):
            pass

        def on_node_done(self, node, result):
            pass

        def on_node_failed(self, node, failure):
            pass

        def on_node_blocked(self, node, failure):
            blocked_calls.append(node.id)

        def save_checkpoint(self, tree):
            pass

    # Constraint agent that enforces max_depth=0 (so depth=1 fails the check)
    class ZeroDepthConstraintAgent:
        async def get_constraints(self, node, parent_result, root_goal):
            return ConstraintSet(max_depth=0, forbidden_actions=[], failure_count_limit=2)

    # Depth exceeds constraint → blocked
    tree.add_child(root_id, "deep_child")
    c1 = _child_id(tree, root_id, 0)

    agent = MockExecutionAgent({})  # no results needed, depth check fails first
    scheduler = Scheduler(
        execution_agent=agent,
        constraint_agent=ZeroDepthConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        callbacks=CountingCallback(),
    )

    await scheduler.run("root", resume_tree=tree)

    assert c1 in blocked_calls
