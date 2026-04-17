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

    max_planning_children: int = 5  # LLM planning output limit: max subgoals in ## TASKS
    max_depth: int = 5             # Max tree depth; nodes at depth >= max_depth do not decompose
    replan_max: int = 5           # Max replan attempts per node before giving up
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
    provider: Any = None  # LLMProvider, used for LLM replan calls

    # Internal state
    _tree: TaskTree | None = None
    _root_goal: str = ""
    # Execution path stack: chain of node IDs from root to current executing node.
    # Top of stack = node whose result we are waiting to handle.
    _path_stack: list[str] = field(default_factory=list)

    @property
    def tree(self) -> TaskTree:
        if self._tree is None:
            raise RuntimeError("Scheduler has not been started")
        return self._tree

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(self, root_goal: str, resume_tree: TaskTree | None = None) -> NodeResult | None:
        """Run the task tree to completion via loop-based DFS and return the root's result.

        If resume_tree is provided, execution continues from that tree's state.

        Returns None when a WAIT_INFO node is reached (loop exits, caller should wait for //taskinfo).
        """
        self._root_goal = root_goal
        if resume_tree is not None:
            self._tree = resume_tree
            logger.info("Scheduler resuming with existing tree, root status: {}",
                self._tree.nodes[self._tree.root_id].status.value
                if self._tree.root_id else "no root")
            self._path_stack = self._rebuild_path_stack()
        else:
            self._tree = TaskTree()
            self._tree.create_root(goal=root_goal)
            logger.info("Scheduler started with root goal: {}", root_goal[:80])
            self._path_stack = [self._tree.root_id]

        # Loop-based DFS
        iterations = 0
        while self._path_stack:
            iterations += 1
            if iterations > 1000:
                logger.warning("DFS loop exceeded 1000 iterations, breaking")
                break
            node = self._tree.get_node(self._path_stack[-1])
            logger.debug("[DFS] iter={} stack={} node={} status={}", iterations, self._path_stack, node.id, node.status.value)

            if node.status == TaskStatus.PENDING:
                result = await self._execute_node_uncallbacked(node)
                wait_info = await self._handle_result(node, result)
                self._save_checkpoint()
                if wait_info:
                    # Node needs user input — loop exits, caller waits for //taskinfo
                    return None
            elif node.status == TaskStatus.WAIT_INFO:
                # Execute WAIT_INFO node directly (user has provided input via //taskinfo)
                result = await self._execute_node_uncallbacked(node)
                wait_info = await self._handle_result(node, result)
                self._save_checkpoint()
                if wait_info:
                    return None
            elif node.status == TaskStatus.RUNNING:
                # RUNNING node (waiting for children) — try to descend to pending child
                pending_child = self._find_next_pending_child(node.id)
                logger.debug("[DFS] RUNNING node={} pending_child={}", node.id, pending_child)
                if pending_child:
                    self._path_stack.append(pending_child)
                else:
                    logger.debug("[DFS] no pending children for RUNNING node={}, advancing", node.id)
                    await self._advance_path()
                    self._save_checkpoint()
            else:
                # Terminal node on top of stack — advance to next
                await self._advance_path()
                self._save_checkpoint()

        logger.info("[DFS] loop exited, iterations={}, stack={}", iterations, self._path_stack)

        # Post-loop: if root is PENDING but has no pending children, finalize it.
        # This handles the case where root has RUNNING children that all failed/terminated
        # while root was waiting. Delegate to _finalize_parent so REPLAN logic is shared.
        if self._tree and self._tree.root_id:
            root = self._tree.nodes.get(self._tree.root_id)
            if root and root.status == TaskStatus.PENDING:
                all_children = [self._tree.nodes[cid] for cid in root.children]
                logger.debug("[post-loop] root status={} children_count={} children_statuses={}",
                    root.status, len(all_children),
                    [c.status for c in all_children])
                all_terminal = all(c.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.TRIED) for c in all_children)
                if all_terminal and all_children:
                    logger.debug("[post-loop] all children terminal, calling _finalize_parent")
                    # Let _finalize_parent handle REPLAN logic for root
                    await self._finalize_parent(root)

        # Execution + Verification loop: executes PENDING nodes, then verifies.
        # On verification failure, marks nodes PENDING for retry.
        # The loop runs until: (a) verification passes, or (b) no pending nodes remain
        # after execution, or (c) max_verification_retries failures occur.
        verification_attempts = 0
        while True:
            # Step 1: Execute all PENDING nodes
            has_pending = any(n.status == TaskStatus.PENDING for n in self._tree.nodes.values())
            if has_pending:
                logger.info("Execution round {}: {} PENDING nodes", verification_attempts + 1,
                    sum(1 for n in self._tree.nodes.values() if n.status == TaskStatus.PENDING))
                self._path_stack = [self._tree.root_id] if self._tree.root_id else []
                while self._path_stack:
                    node = self._tree.get_node(self._path_stack[-1])
                    if node.status == TaskStatus.PENDING:
                        result = await self._execute_node_uncallbacked(node)
                        wait_info = await self._handle_result(node, result)
                        self._save_checkpoint()
                        if wait_info:
                            # Node needs user input — loop exits, caller waits for //taskinfo
                            return None
                    elif node.status == TaskStatus.WAIT_INFO:
                        result = await self._execute_node_uncallbacked(node)
                        wait_info = await self._handle_result(node, result)
                        self._save_checkpoint()
                        if wait_info:
                            return None
                    elif node.status == TaskStatus.DONE:
                        # DONE node with PENDING children — descend to first pending child
                        pending_child = self._find_next_pending_child(node.id)
                        if pending_child:
                            self._path_stack.append(pending_child)
                        else:
                            await self._advance_path()
                            self._save_checkpoint()
                    elif node.status == TaskStatus.RUNNING:
                        # RUNNING node (waiting for children) — try to descend to pending child
                        pending_child = self._find_next_pending_child(node.id)
                        if pending_child:
                            self._path_stack.append(pending_child)
                        else:
                            # No pending children yet — advance (bubble up)
                            await self._advance_path()
                            self._save_checkpoint()
                    else:
                        await self._advance_path()
                        self._save_checkpoint()

            # Step 2: After execution loop, check if new work appeared.
            # _finalize_parent may spawn new PENDING children (REPLAN case).
            # If root status changed to PENDING (new children spawned) or we have
            # pending children anywhere, continue to process them.
            did_step2_continue = False
            if self._tree and self._tree.root_id:
                root = self._tree.nodes.get(self._tree.root_id)
                if root and root.status == TaskStatus.PENDING:
                    all_children = [self._tree.nodes[cid] for cid in root.children]
                    any_terminal = any(c.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.TRIED) for c in all_children)
                    if any_terminal:
                        await self._finalize_parent(root)
                    if root.status == TaskStatus.PENDING:
                        did_step2_continue = True
                elif root and root.status == TaskStatus.RUNNING:
                    # root is RUNNING but all children are terminal — finalize it
                    all_children = [self._tree.nodes[cid] for cid in root.children]
                    all_terminal = all(c.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.TRIED) for c in all_children)
                    if all_terminal and all_children:
                        await self._finalize_parent(root)
                        if root.status == TaskStatus.PENDING:
                            did_step2_continue = True
                elif root and root.status == TaskStatus.DONE:
                    pass  # Step 1 now handles DONE with PENDING children via _find_next_pending_child

            if did_step2_continue:
                continue

            # Step 3: Run verification to check if execution results are acceptable.
            # On failure, _run_verification marks failed nodes PENDING for retry.
            if self.verification_agent is not None:
                verification_passed = await self._run_verification()
                if verification_passed:
                    break  # Verification passed — we're done
                verification_attempts += 1
                if verification_attempts >= self.config.max_verification_retries:
                    logger.info("Verification retries exhausted ({})", verification_attempts)
                    break
                # Verification failed — loop continues to Step 1 to execute PENDING nodes

            # Step 4: If no verifier and no pending work, we're done
            has_pending = any(n.status == TaskStatus.PENDING for n in self._tree.nodes.values())
            if not has_pending:
                break

        root_result = self._tree.get_root_result()
        if root_result is None:
            # Root failed without setting a result — convert failure to NodeResult
            root_node = self._tree.nodes.get(self._tree.root_id)
            if root_node and root_node.failure:
                root_result = NodeResult(
                    node_id=self._tree.root_id,
                    summary=f"[FAILED] {root_node.failure.summary}",
                    constraints_respected=False,
                    token_spent=0,
                )
                self._tree.nodes[self._tree.root_id].result = root_result
            else:
                root_result = NodeResult(
                    node_id="root",
                    summary="Task completed with no result",
                )
                self._tree.nodes["root"].result = root_result

        logger.info("Scheduler finished")
        return root_result

    # -------------------------------------------------------------------------
    # Path stack helpers
    # -------------------------------------------------------------------------

    def _rebuild_path_stack(self) -> list[str]:
        """Rebuild path stack from a resumed tree.

        Finds the deepest PENDING or WAIT_INFO node as the current execution point.
        WAIT_INFO nodes are prioritized over PENDING at the same depth.
        """
        if self._tree is None or self._tree.root_id is None:
            return []

        # Find deepest WAIT_INFO node (highest priority)
        deepest_wait: str | None = None
        deepest_wait_depth = -1
        for nid, node in self._tree.nodes.items():
            if node.status == TaskStatus.WAIT_INFO and node.depth > deepest_wait_depth:
                deepest_wait = nid
                deepest_wait_depth = node.depth

        # Find deepest PENDING node at depth >= deepest_wait_depth
        deepest_pend: str | None = None
        deepest_pend_depth = -1
        for nid, node in self._tree.nodes.items():
            if node.status == TaskStatus.PENDING and node.depth > deepest_pend_depth:
                deepest_pend = nid
                deepest_pend_depth = node.depth

        # WAIT_INFO takes priority over PENDING at same depth
        if deepest_wait is not None and deepest_wait_depth >= deepest_pend_depth:
            target = deepest_wait
        elif deepest_pend is not None:
            target = deepest_pend
        else:
            # No PENDING/WAIT_INFO — find deepest RUNNING
            deepest_run: str | None = None
            deepest_run_depth = -1
            for nid, node in self._tree.nodes.items():
                if node.status == TaskStatus.RUNNING and node.depth > deepest_run_depth:
                    deepest_run = nid
                    deepest_run_depth = node.depth
            if deepest_run is None:
                return [self._tree.root_id]
            target = deepest_run

        # Build stack: walk from root to target
        stack = []
        current = target
        while current is not None:
            stack.append(current)
            current = self._tree.nodes[current].parent_id
        stack.reverse()
        return stack

    def _find_next_pending_child(self, parent_id: str) -> str | None:
        """Return the first pending child of parent_id, or None if all children are terminal."""
        parent = self._tree.nodes[parent_id]
        for cid in parent.children:
            if self._tree.nodes[cid].status == TaskStatus.PENDING:
                return cid
        return None

    async def _advance_path(self) -> None:
        """Pop completed node, push next sibling or bubble up.

        Called when a node is terminal (DONE/FAILED/BLOCKED) and ready to
        be removed from the path stack. Finds the next pending child of the
        new top of stack, or bubbles up further.
        """
        self._path_stack.pop()

        while self._path_stack:
            parent = self._tree.nodes[self._path_stack[-1]]

            # Try to continue with the next pending child of this parent
            next_child = self._find_next_pending_child(parent.id)
            if next_child:
                self._path_stack.append(next_child)
                return

            # No more pending children — parent is fully evaluated.
            # If parent is RUNNING (waiting for children), finalize it.
            if parent.status == TaskStatus.RUNNING:
                await self._finalize_parent(parent)

            # Bubble up
            self._path_stack.pop()

        # Stack empty — DFS complete

    async def _finalize_parent(self, parent: TaskNode) -> None:
        """Called when all children of parent are terminal.

        Marks parent DONE (all succeeded) or handles REPLAN (some failed).
        Failed children are marked TRIED (not blocking, only providing context).
        If REPLAN generates new children, the first new child is pushed
        onto the path stack.
        """
        all_children = [self._tree.nodes[cid] for cid in parent.children]
        any_failed = any(c.status == TaskStatus.FAILED for c in all_children)

        if any_failed:
            # Some child failed — check REPLAN
            if parent.replan_count >= self.config.replan_max:
                self._tree.mark_failed(parent.id, FailureReport(
                    node_id=parent.id, status=TaskStatus.FAILED,
                    root_cause=RootCause.MAX_REPLAN_REACHED,
                    summary=f"replan_max={self.config.replan_max} exhausted",
                    constraint_veto=False, workspace_state=WorkspaceState.CLEAN,
                ))
                if self.callbacks:
                    self.callbacks.on_node_failed(parent, self._tree.nodes[parent.id].failure)
                return

            parent.replan_count += 1

            # Mark failed children as TRIED (they provide context but don't block)
            for child in all_children:
                if child.status == TaskStatus.FAILED:
                    child.status = TaskStatus.TRIED

            accumulated = self._collect_all_children_results(parent)
            new_goals = await self._llm_replan(parent, accumulated)

            if new_goals is None or new_goals == []:
                # REPLAN exhausted — parent gives up
                self._tree.mark_failed(parent.id, FailureReport(
                    node_id=parent.id, status=TaskStatus.FAILED,
                    root_cause=RootCause.NO_REMAINING_OPTIONS,
                    summary="replan returned no options",
                    constraint_veto=False, workspace_state=WorkspaceState.CLEAN,
                ))
                if self.callbacks:
                    self.callbacks.on_node_failed(parent, self._tree.nodes[parent.id].failure)
                return

            # Replace failed children with new LLM-proposed goals
            self._replace_children_with_new_goals(parent, new_goals)

            # Push first new child
            next_child = self._find_next_pending_child(parent.id)
            if next_child:
                self._path_stack.append(next_child)
                return
        else:
            # All children succeeded — mark parent done
            if parent.result:
                self._tree.mark_done(parent.id, parent.result)
            else:
                self._tree.mark_done(parent.id, NodeResult(node_id=parent.id, summary="Task completed."))
            if self.callbacks:
                self.callbacks.on_node_done(parent, parent.result)

    # -------------------------------------------------------------------------
    # Handle result — decide next action after node execution
    # -------------------------------------------------------------------------

    async def _handle_result(self, node: TaskNode, result_or_failure: NodeResult | FailureReport) -> bool:
        """Handle the result of executing a node.

        Returns True if node is WAIT_INFO (caller should exit loop and wait for //taskinfo).
        On success: decompose into children, spawn them, and push first child.
        On failure: mark failed, then advance path to bubble up.
        """
        if isinstance(result_or_failure, NodeResult):
            # Success
            if self.callbacks:
                self.callbacks.on_node_done(node, result_or_failure)

            # Check if node needs user input
            if result_or_failure.user_input_question:
                self._tree.mark_wait_info(node.id)
                node.result = result_or_failure
                # Clear path stack since we're waiting for user input
                self._path_stack.clear()
                # Clear user_input_question so that when node is resumed via //taskinfo,
                # it does not trigger WAIT_INFO again on re-execution
                result_or_failure.user_input_question = None
                return True  # Signal to exit loop

            # max_depth: nodes at depth >= max_depth do not decompose further
            if node.depth >= self.config.max_depth:
                self._tree.mark_done(node.id, result_or_failure)
                await self._advance_path()
                return False

            # Use children_goals from result (parsed in build_result_from_agent_response)
            # Fall back to subgoal_parser for backward compatibility
            children = result_or_failure.children_goals
            if not children and self.subgoal_parser:
                children = self.subgoal_parser.parse(result_or_failure)

            logger.debug("[_handle_result] node={} result_summary={} parsed_children={}",
                node.id, result_or_failure.summary[:200], children)

            # Root node with no children and no artifacts: the LLM said "leaf" (or
            # returned nothing useful) but did no actual work. Only fail if the summary
            # is the default "Task completed." (LLM produced nothing meaningful).
            # If summary has real content, treat as a legitimate leaf completion.
            is_meaningful_summary = result_or_failure.summary and result_or_failure.summary != "Task completed."
            if node.parent_id is None and not children and not result_or_failure.artifacts and not is_meaningful_summary:
                failure = FailureReport(
                    node_id=node.id,
                    status=TaskStatus.FAILED,
                    root_cause=RootCause.NO_REMAINING_OPTIONS,
                    summary=result_or_failure.summary or "Root produced no children and no artifacts",
                    constraint_veto=False,
                    workspace_state=result_or_failure.workspace_state,
                )
                node.failure = failure
                node.result = None
                self._tree.nodes[node.id].status = TaskStatus.FAILED
                self._tree.nodes[node.id].failure = failure
                # Fall through to root-level REPLAN logic below
                result_or_failure = failure
                # eslint-disable-next-line no-constant-condition
                if False:  # dummy to allow next block to compile after edit
                    pass
            elif not children:
                # Leaf node (no children) — mark done, advance path
                self._tree.mark_done(node.id, result_or_failure)
                await self._advance_path()
                return False

            # Node has children — switch it to RUNNING (waiting for children)
            self._tree.mark_running(node.id)
            node.result = result_or_failure

            # Clear existing children (from prior REPLAN attempts)
            for cid in list(node.children):
                if self._tree.nodes[cid].status == TaskStatus.PENDING:
                    self._tree.mark_blocked(cid)
            node.children = []

            # Spawn children
            for goal in children[:self.config.max_planning_children]:
                self._tree.add_child(node.id, goal)

            # Push first child onto stack
            if node.children:
                self._path_stack.append(node.children[0])
            return False
        else:
            # Failure
            failure = result_or_failure
            if failure.constraint_veto:
                self._tree.mark_blocked(node.id)
                if self.callbacks:
                    self.callbacks.on_node_blocked(node, failure)
                await self._advance_path()
                return False

            # Root-level REPLAN: root has no parent, so it replans itself.
            # Try REPLAN before marking permanently failed (unless replan exhausted).
            if node.parent_id is None:
                if node.replan_count >= self.config.replan_max:
                    self._tree.mark_failed(node.id, failure)
                    logger.debug("Root replan_max exhausted, root stays failed")
                    if self.callbacks:
                        self.callbacks.on_node_failed(node, failure)
                else:
                    node.replan_count += 1
                    logger.info("Root failed (replan {}/{}), calling REPLAN", node.replan_count, self.config.replan_max)
                    new_goals = await self._llm_replan(node, failure.summary)
                    if new_goals:
                        # REPLAN produced children — switch root to RUNNING and spawn them
                        self._tree.mark_running(node.id)
                        for goal in new_goals[:self.config.max_planning_children]:
                            self._tree.add_child(node.id, goal)
                        self._path_stack.append(node.children[0])
                        self._save_checkpoint()
                        return False
                    # REPLAN returned nothing — mark permanently failed
                    self._tree.mark_failed(node.id, failure)
                    if self.callbacks:
                        self.callbacks.on_node_failed(node, failure)
            else:
                self._tree.mark_failed(node.id, failure)
                logger.debug("Node {} failed: {}", node.id, failure.summary[:80])
                if self.callbacks:
                    self.callbacks.on_node_failed(node, failure)

            # Advance path — this pops the failed node and bubbles up
            await self._advance_path()
            return False

    def get_tree(self) -> TaskTree:
        """Return the current task tree (for inspection or persistence)."""
        return self._tree

    # -------------------------------------------------------------------------
    # Node executor
    # -------------------------------------------------------------------------

    async def _execute_node_uncallbacked(self, node: TaskNode) -> NodeResult | FailureReport:
        """Execute a single node. Returns NodeResult or FailureReport. No callback cascade."""
        parent_node = self._tree.get_node(node.parent_id) if node.parent_id else None
        parent_result = parent_node.result if parent_node else None

        self._tree.mark_running(node.id)
        logger.debug("Executing node {}", node.id)
        if self.callbacks:
            self.callbacks.on_node_start(node)

        constraints = await self._get_constraints(node, parent_result)

        if node.depth > constraints.max_depth:
            return FailureReport(
                node_id=node.id,
                status=TaskStatus.FAILED,
                root_cause=RootCause.CONSTRAINT_VETO,
                summary=f"depth {node.depth} > max_depth {constraints.max_depth}",
                constraint_veto=True,
                workspace_state=WorkspaceState.CLEAN,
            )

        result = await self.execution_agent.execute(
            node=node,
            constraints=constraints,
            parent_result=parent_result,
            root_goal=self._root_goal,
            root_goal_context=self._build_root_context(),
            tree=self._tree,
        )

        # If node already has user_input_answer (from previous WAIT_INFO + //taskinfo),
        # copy it to the fresh result so we don't re-trigger WAIT_INFO on re-execution.
        # If node already has user_input_answer (from previous WAIT_INFO + //taskinfo),
        # copy it to the fresh result so we don't re-trigger WAIT_INFO on re-execution.
        if isinstance(result, NodeResult) and node.result and node.result.user_input_answer:
            result.user_input_answer = node.result.user_input_answer
            result.user_input_question = None

        return result

    # -------------------------------------------------------------------------
    # LLM Replan
    # -------------------------------------------------------------------------

    async def _llm_replan(self, parent: TaskNode, accumulated: str) -> list[str] | None:
        """Call LLM to re-decompose parent goal with all children context.

        Returns list of new child goal titles, or None if LLM says exhausted.
        """
        # Import here to avoid circular dependency at module load time
        from nanobot.agent.tasktree.execution.subgoal import _try_parse_tasks_block

        prompt = f"""TaskTree: a child subtask failed. Re-decompose the parent task.

Parent Goal: {parent.goal}

Children execution results so far:
{accumulated}

Output new subtasks (max {self.config.max_planning_children}):
##[TASKS]
[{{"goal": "subtask title", "description": "why different from failed approach"}}]
##[/TASKS]

Rules:
- Do NOT repeat the approach that already failed
- If no more options remain, output: ##[TASKS]
[]
##[/TASKS]
- Output ONLY the ##[TASKS]...##[/TASKS] block
"""
        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.provider.get_default_model(),
                max_tokens=512,
            )
            text = response.content if hasattr(response, "content") else str(response)
            logger.debug("[_llm_replan] raw_response={}", text[:500])
            goals = _try_parse_tasks_block(text)
            # None = no ##[TASKS] block found (failed to parse); [] = explicit empty (LLM says no more options)
            return goals
        except Exception:
            logger.warning("LLM replan call failed")
            return None

    def _collect_all_children_results(self, parent: TaskNode) -> str:
        """Build context of ALL children results across parent's entire lifetime.

        Includes DONE, FAILED, and TRIED children for REPLAN context.
        """
        lines = []
        for cid in parent.children:
            child = self._tree.get_node(cid)
            if child.status == TaskStatus.DONE:
                lines.append(f"[SUCCEEDED] {child.goal}: {child.result.summary}")
            elif child.status == TaskStatus.FAILED:
                lines.append(f"[FAILED] {child.goal}: {child.failure.summary}")
            elif child.status == TaskStatus.TRIED:
                lines.append(f"[TRIED] {child.goal}: {child.failure.summary if child.failure else 'no details'}")
        return "\n".join(lines) if lines else "All children failed."

    def _replace_children_with_new_goals(self, parent: TaskNode, new_goals: list[str]) -> None:
        """REPLAN: replace all children with new LLM-proposed goals.

        Previous children are kept (with their TRIED/FAILED status) as context.
        New children are added as PENDING.
        """
        for goal_text in new_goals[:self.config.max_planning_children]:
            self._tree.add_child(parent.id, goal_text)

    # -------------------------------------------------------------------------
    # Bubble up — simple walk-up
    # -------------------------------------------------------------------------

    async def _bubble_up(self, node: TaskNode) -> None:
        """Simply walk up to parent. Parent's for loop handles sibling judgment."""
        if node.parent_id is None:
            return  # Reached root

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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

    def _build_root_context(self) -> str:
        """Build a root context block (used as anchor in all node contexts)."""
        return f"[Root Goal]\n{self._root_goal}\n[End Root Goal]"

    # -------------------------------------------------------------------------
    # Verification phase
    # -------------------------------------------------------------------------

    async def _run_verification(self) -> bool:
        """Run verification phase after DFS completes.

        Returns:
            True  — verification passed (or skipped), caller should return root_result
            False — verification failed, caller should check has_pending and retry
        """
        if self.verification_agent is None:
            return True  # No verification needed

        all_nodes = list(self._tree.nodes.values())
        has_meaningful_content = any(
            (n.result and n.result.artifacts) or
            (n.result and n.result.summary and n.result.summary != "Task completed.")
            for n in all_nodes
        )
        if not has_meaningful_content:
            logger.debug("Skipping verification (planning-only task)")
            return True  # No content to verify

        all_results = {
            nid: n.result
            for nid, n in self._tree.nodes.items()
            if n.result is not None
        }
        vr = await self.verification_agent.verify(self._root_goal, all_results)
        if vr.passed or not vr.failed_nodes:
            return True  # Verification passed

        logger.info("Verification failed: {}", vr.reason)

        for node_id in vr.failed_nodes:
            if node_id not in self._tree.nodes:
                continue
            node = self._tree.nodes[node_id]
            node.verification_failure = vr.reason
            self._tree.mark_pending(node_id)
            if node.parent_id is not None:
                parent = self._tree.nodes[node.parent_id]
                if parent.status == TaskStatus.DONE:
                    parent.status = TaskStatus.RUNNING

        return False  # Verification failed, nodes marked PENDING for retry
