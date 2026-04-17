"""Tests for new TaskTree behaviors: TRIED, WAIT_INFO, REPLAN, children_goals, summary stripping."""

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
from nanobot.agent.tasktree.scheduler import Scheduler, SchedulerConfig
from nanobot.agent.tasktree.tree import TaskTree
from nanobot.agent.tasktree.context import (
    build_result_from_agent_response,
    _strip_tasks_block,
    _try_parse_tasks_block_for_result,
)


# ---------------------------------------------------------------------------
# _strip_tasks_block
# ---------------------------------------------------------------------------

class TestStripTasksBlock:
    def test_strips_modern_tasks_block(self):
        text = "Some explanation\n##[TASKS]\n[\"child1\", \"child2\"]\n##[/TASKS]\nMore text"
        result = _strip_tasks_block(text)
        assert "##[TASKS]" not in result
        assert "child1" not in result
        assert "Some explanation" in result
        assert "More text" in result

    def test_strips_legacy_tasks_block(self):
        text = "Explanation\n## TASKS\n[\"child1\"]\n## TASKS\nAfter"
        result = _strip_tasks_block(text)
        assert "## TASKS" not in result
        assert "child1" not in result
        assert "Explanation" in result
        assert "After" in result

    def test_strips_empty_tasks_block(self):
        text = "Before\n##[TASKS]\n[]\n##[/TASKS]\nAfter"
        result = _strip_tasks_block(text)
        assert "##[TASKS]" not in result
        assert "Before" in result
        assert "After" in result

    def test_no_tasks_block_unchanged(self):
        text = "Just plain text without any tasks block"
        result = _strip_tasks_block(text)
        assert result == text

    def test_only_tasks_block(self):
        text = "##[TASKS]\n[\"child1\"]\n##[/TASKS]"
        result = _strip_tasks_block(text)
        assert result == ""


# ---------------------------------------------------------------------------
# _try_parse_tasks_block_for_result
# ---------------------------------------------------------------------------

class TestParseTasksBlockForResult:
    def test_parses_modern_format_string_array(self):
        content = 'Some text\n##[TASKS]\n["goal 1", "goal 2"]\n##[/TASKS]\nEnd'
        goals = _try_parse_tasks_block_for_result(content)
        assert goals == ["goal 1", "goal 2"]

    def test_parses_modern_format_dict_array(self):
        content = '##[TASKS]\n[{"goal": "task 1", "description": "why"}, {"goal": "task 2"}]\n##[/TASKS]'
        goals = _try_parse_tasks_block_for_result(content)
        assert goals == ["task 1", "task 2"]

    def test_parses_empty_array(self):
        content = '##[TASKS]\n[]\n##[/TASKS]'
        goals = _try_parse_tasks_block_for_result(content)
        assert goals == []

    def test_no_tasks_block_returns_empty(self):
        goals = _try_parse_tasks_block_for_result("Just plain text")
        assert goals == []

    def test_legacy_format(self):
        content = "## TASKS\n[\"old task\"]\n## TASKS"
        goals = _try_parse_tasks_block_for_result(content)
        assert goals == ["old task"]


# ---------------------------------------------------------------------------
# build_result_from_agent_response — children_goals + summary stripping
# ---------------------------------------------------------------------------

class TestBuildResultChildrenGoals:
    def test_children_goals_extracted(self):
        content = 'Result text\n##[TASKS]\n["child A", "child B"]\n##[/TASKS]'
        result = build_result_from_agent_response(node_id="n1", agent_content=content)
        assert result.children_goals == ["child A", "child B"]

    def test_summary_stripped_of_tasks_block(self):
        content = 'Some explanation\n##[TASKS]\n["child"]\n##[/TASKS]'
        result = build_result_from_agent_response(node_id="n1", agent_content=content)
        assert "##[TASKS]" not in result.summary
        assert "Some explanation" in result.summary

    def test_empty_tasks_block_gives_default_summary(self):
        content = '##[TASKS]\n[]\n##[/TASKS]'
        result = build_result_from_agent_response(node_id="n1", agent_content=content)
        assert result.summary == "Task completed."
        assert result.children_goals == []

    def test_no_tasks_block_uses_full_content(self):
        content = "The agent completed the task successfully."
        result = build_result_from_agent_response(node_id="n1", agent_content=content)
        assert result.summary == "The agent completed the task successfully."


# ---------------------------------------------------------------------------
# TaskStatus.TRIED and WAIT_INFO are NOT terminal
# ---------------------------------------------------------------------------

class TestTerminalStates:
    def test_tried_is_not_terminal(self):
        node = TaskNode(id="n1", goal="test", status=TaskStatus.TRIED)
        assert node.is_terminal() is False

    def test_wait_info_is_not_terminal(self):
        node = TaskNode(id="n1", goal="test", status=TaskStatus.WAIT_INFO)
        assert node.is_terminal() is False

    def test_pending_is_not_terminal(self):
        node = TaskNode(id="n1", goal="test", status=TaskStatus.PENDING)
        assert node.is_terminal() is False

    def test_running_is_not_terminal(self):
        node = TaskNode(id="n1", goal="test", status=TaskStatus.RUNNING)
        assert node.is_terminal() is False

    def test_done_is_terminal(self):
        node = TaskNode(id="n1", goal="test", status=TaskStatus.DONE)
        assert node.is_terminal() is True

    def test_failed_is_terminal(self):
        node = TaskNode(id="n1", goal="test", status=TaskStatus.FAILED)
        assert node.is_terminal() is True

    def test_blocked_is_terminal(self):
        node = TaskNode(id="n1", goal="test", status=TaskStatus.BLOCKED)
        assert node.is_terminal() is True


# ---------------------------------------------------------------------------
# TaskTree mark_wait_info
# ---------------------------------------------------------------------------

class TestMarkWaitInfo:
    def test_mark_wait_info(self):
        tree = TaskTree()
        tree.create_root(goal="root")
        tree.mark_wait_info("root")
        assert tree.nodes["root"].status == TaskStatus.WAIT_INFO


# ---------------------------------------------------------------------------
# REPLAN: failed child becomes TRIED
# ---------------------------------------------------------------------------

class TestReplanTried:
    @pytest.mark.asyncio
    async def test_failed_child_becomes_tried_on_replan(self):
        """When parent REPLANs after child failure, failed child is marked TRIED."""
        tree = TaskTree()
        tree.create_root(goal="root")
        tree.add_child("root", "child")
        root_id = tree.root_id
        child_id = tree.nodes[root_id].children[0]

        # Root executes: produces child
        # Child executes: fails (constraint_veto=False)
        child_result = FailureReport(
            node_id=child_id,
            summary="child failed",
            root_cause=RootCause.UNKNOWN,
            constraint_veto=False,
        )

        # Execute root -> spawns child
        async def mock_execute_root(node, constraints, parent_result, root_goal, root_goal_context, tree):
            if node.id == root_id:
                return NodeResult(node_id=root_id, summary="root done", children_goals=["child"])
            return child_result

        async def mock_execute_child(node, constraints, parent_result, root_goal, root_goal_context, tree):
            return child_result

        mock_agent = AsyncMock()
        mock_agent.execute = mock_execute_root  # Will be called for root first

        scheduler = Scheduler(
            execution_agent=_MockAgentWithChildren({"root": NodeResult(node_id=root_id, summary="root", children_goals=["child"])}),
            constraint_agent=_MockConstraintAgent(),
            subgoal_parser=None,
            provider=_MockProvider(),
            config=SchedulerConfig(replan_max=2),
        )

        # Run root which spawns child
        tree2 = TaskTree()
        tree2.create_root(goal="root")
        tree2.add_child("root", "child")
        # Mark child as done so finalize_parent triggers replan
        tree2.mark_done("root", NodeResult(node_id="root", summary="done"))
        child_id2 = tree2.nodes["root"].children[0]
        tree2.mark_failed(child_id2, FailureReport(node_id=child_id2, summary="failed", root_cause=RootCause.UNKNOWN))

        # This tests the _finalize_parent path: when child is FAILED, parent should replan
        # and mark child as TRIED
        scheduler2 = Scheduler(
            execution_agent=AsyncMock(),
            constraint_agent=_MockConstraintAgent(),
            subgoal_parser=None,
            provider=_MockProvider(),
            config=SchedulerConfig(replan_max=2),
        )
        scheduler2._tree = tree2
        scheduler2._root_goal = "root"

        # Trigger finalize_parent manually
        parent = tree2.nodes["root"]
        await scheduler2._finalize_parent(parent)

        # Failed child should now be TRIED
        child_node = tree2.nodes[child_id2]
        assert child_node.status == TaskStatus.TRIED


# ---------------------------------------------------------------------------
# WAIT_INFO: scheduler.run returns None
# ---------------------------------------------------------------------------

class TestWaitInfo:
    @pytest.mark.asyncio
    async def test_wait_info_returns_none(self):
        """When a node needs user input, scheduler.run() returns None."""
        tree = TaskTree()
        tree.create_root(goal="root")
        root_id = tree.root_id

        wait_result = NodeResult(
            node_id=root_id,
            summary="need info",
            user_input_question="What file path?",
        )

        scheduler = Scheduler(
            execution_agent=_WaitInfoAgent(wait_result),
            constraint_agent=_MockConstraintAgent(),
            subgoal_parser=None,
        )

        result = await scheduler.run("root", resume_tree=tree)
        # WAIT_INFO -> run returns None
        assert result is None
        # Tree root should be WAIT_INFO
        assert tree.nodes[root_id].status == TaskStatus.WAIT_INFO
        # user_input_question should be cleared to prevent re-trigger on resume
        assert tree.nodes[root_id].result.user_input_question is None

    @pytest.mark.asyncio
    async def test_wait_info_node_resumes_after_user_input(self):
        """After user provides input, WAIT_INFO node resumes and completes."""
        tree = TaskTree()
        tree.create_root(goal="root")
        root_id = tree.root_id

        # First execution: needs user input
        wait_result = NodeResult(
            node_id=root_id,
            summary="need info",
            user_input_question="What file path?",
        )

        scheduler = Scheduler(
            execution_agent=_WaitInfoAgent(wait_result),
            constraint_agent=_MockConstraintAgent(),
            subgoal_parser=None,
        )

        result = await scheduler.run("root", resume_tree=tree)
        assert result is None
        assert tree.nodes[root_id].status == TaskStatus.WAIT_INFO

        # Simulate //taskinfo: provide user input and resume
        tree.nodes[root_id].result.user_input_answer = "/tmp/myfile.txt"
        tree.nodes[root_id].result.user_input_question = None  # cleared
        tree.nodes[root_id].status = TaskStatus.PENDING  # reset for resume

        # Resume - should now complete (WaitInfoAgent returns done on second call)
        result = await scheduler.run("root", resume_tree=tree)
        assert result is not None
        assert tree.nodes[root_id].status == TaskStatus.DONE

    def test_rebuild_path_stack_prioritizes_wait_info_over_pending(self):
        """WAIT_INFO node takes priority over PENDING node at same depth."""
        tree = TaskTree()
        tree.create_root(goal="root")
        root_id = tree.root_id

        # Add two children
        tree.add_child(root_id, "child1")
        tree.add_child(root_id, "child2")
        child1_id = tree.nodes[root_id].children[0]
        child2_id = tree.nodes[root_id].children[1]

        # Mark root RUNNING
        tree.mark_running(root_id)
        # child1 is WAIT_INFO, child2 is PENDING
        tree.mark_wait_info(child1_id)
        tree.mark_pending(child2_id)

        scheduler = Scheduler(
            execution_agent=_MockAgentAlwaysDone(),
            constraint_agent=_MockConstraintAgent(),
            subgoal_parser=None,
        )
        scheduler._tree = tree
        scheduler._root_goal = "root"

        path = scheduler._rebuild_path_stack()
        # Should rebuild path to child1 (WAIT_INFO), not child2 (PENDING)
        assert path == [root_id, child1_id]

    def test_rebuild_path_stack_deepest_wait_info(self):
        """If deepest WAIT_INFO is deeper than deepest PENDING, WAIT_INFO wins."""
        tree = TaskTree()
        tree.create_root(goal="root")
        root_id = tree.root_id

        tree.add_child(root_id, "child1")
        child1_id = tree.nodes[root_id].children[0]
        tree.add_child(child1_id, "grandchild")
        child1_grandchild = tree.nodes[child1_id].children[0]

        # root RUNNING, child1 DONE, grandchild WAIT_INFO
        tree.mark_running(root_id)
        tree.mark_done(child1_id, NodeResult(node_id=child1_id, summary="done"))
        tree.mark_wait_info(child1_grandchild)

        scheduler = Scheduler(
            execution_agent=_MockAgentAlwaysDone(),
            constraint_agent=_MockConstraintAgent(),
            subgoal_parser=None,
        )
        scheduler._tree = tree
        scheduler._root_goal = "root"

        path = scheduler._rebuild_path_stack()
        # Should rebuild to grandchild (WAIT_INFO) via root -> child1 -> grandchild
        assert path == [root_id, child1_id, child1_grandchild]


class _MockAgentAlwaysDone:
    """Mock agent that always returns a simple done result."""
    async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
        return NodeResult(node_id=node.id, summary="done")


# ---------------------------------------------------------------------------
# Helper mocks
# ---------------------------------------------------------------------------

class _MockConstraintAgent:
    async def get_constraints(self, node, parent_result, root_goal):
        return ConstraintSet(max_depth=20)


class _MockProvider:
    async def chat(self, messages, model, max_tokens):
        class R:
            content = '##[TASKS]\n[]\n##[/TASKS]'
        return R()

    def get_default_model(self):
        return "test"


class _MockAgentWithChildren:
    """Returns NodeResult with children_goals set."""
    def __init__(self, results: dict[str, NodeResult | FailureReport]):
        self.results = results

    async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
        result = self.results.get(node.id)
        if result is None:
            return FailureReport(
                node_id=node.id,
                summary=f"No mock for {node.id}",
                root_cause=RootCause.UNKNOWN,
            )
        return result


class _WaitInfoAgent:
    """Returns NodeResult with user_input_question set on first call."""
    def __init__(self, wait_result: NodeResult):
        self.wait_result = wait_result
        self.call_count = 0

    async def execute(self, node, constraints, parent_result, root_goal, root_goal_context, tree):
        self.call_count += 1
        if self.call_count == 1:
            return self.wait_result
        return NodeResult(node_id=node.id, summary="done after wait info")
