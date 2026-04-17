"""Integration tests for replan, bubble-up failure propagation, and max_planning_children boundary."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.models import (
    ConstraintSet,
    FailureReport,
    NodeResult,
    TaskNode,
    TaskStatus,
    RootCause,
)
from nanobot.agent.tasktree.scheduler import Scheduler, SchedulerConfig
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
async def test_scheduler_config_max_planning_children_defaults_to_5():
    """SchedulerConfig.max_planning_children should default to 5."""
    config = SchedulerConfig()
    assert config.max_planning_children == 5


@pytest.mark.asyncio
async def test_handle_failure_replan_exhausted_fails_parent():
    """When a child fails and replan_count >= replan_max, parent is marked failed."""
    tree = TaskTree()
    tree.create_root(goal="root")
    root_id = tree.root_id

    # root has one child
    tree.add_child(root_id, "child")
    child_id = tree.nodes[root_id].children[0]

    # Set replan_count to replan_max so next failure exhausts replan
    tree.nodes[root_id].replan_count = 4  # replan_max=5, so 5th attempt fails

    # Child fails (simulate by marking it failed before scheduler runs)
    agent = MockExecutionAgent({
        child_id: FailureReport(
            node_id=child_id,
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
        config=SchedulerConfig(replan_max=5),
    )

    await scheduler.run("root", resume_tree=tree)

    # Parent (root) should be failed (because replan exhausted)
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
    """When a child fails and parent REPLANs, LLM generates replacement goals."""
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

    class MockLLMProvider:
        async def chat(self, messages, model, max_tokens):
            # Return a new child goal
            class R:
                content = '##[TASKS]\n["approach_2"]\n##[/TASKS]'
            return R()

        def get_default_model(self):
            return "test"

    scheduler = Scheduler(
        execution_agent=TrackingAgent(),
        constraint_agent=MockConstraintAgent(),
        subgoal_parser=MockSubgoalParser(),
        provider=MockLLMProvider(),
        config=SchedulerConfig(replan_max=5),
    )

    await scheduler.run("root", resume_tree=tree)

    # Should have spawned a replacement child
    replacement_count = len(tree.nodes[root_id].children)
    assert replacement_count >= 1
    assert replacement_id is not None


@pytest.mark.asyncio
async def test_llm_replan_returns_none_on_failure():
    """LLM REPLAN returns None when LLM call fails."""
    scheduler = Scheduler()

    class FailingProvider:
        async def chat(self, messages, model, max_tokens):
            raise RuntimeError("LLM unavailable")

        def get_default_model(self):
            return "test"

    scheduler.provider = FailingProvider()

    result = await scheduler._llm_replan(
        TaskNode(id="n1", goal="test goal"),
        "context"
    )
    assert result is None


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
