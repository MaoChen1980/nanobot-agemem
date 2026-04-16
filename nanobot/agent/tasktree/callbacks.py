"""Scheduler callbacks for session persistence and external observability."""

from __future__ import annotations

from loguru import logger

from nanobot.agent.tasktree.models import (
    FailureReport,
    NodeResult,
    TaskNode,
)
from nanobot.agent.tasktree.scheduler import SchedulerCallbacks
from nanobot.agent.tasktree.tree import TaskTree
from nanobot.session.manager import SessionManager

_TREE_CHECKPOINT_KEY = "_tasktree_checkpoint"


class SessionPersistenceCallback:
    """SchedulerCallback that persists node results to SessionManager.

    Also saves the full tree state as a checkpoint after each node,
    enabling task resumption after crash or cancellation.
    """

    def __init__(self, session_manager: SessionManager, session_key: str):
        self.session_manager = session_manager
        self.session_key = session_key

    def on_node_start(self, node: TaskNode) -> None:
        msg = f"[TaskTree] Starting node: {node.id} — {node.goal}"
        logger.debug(msg)
        self._append_to_session({"role": "system", "content": msg})

    def on_node_done(self, node: TaskNode, result: NodeResult) -> None:
        msg = f"[TaskTree] Node {node.id} done: {result.summary}"
        logger.debug(msg)
        self._append_to_session({"role": "system", "content": msg})

    def on_node_failed(self, node: TaskNode, failure: FailureReport) -> None:
        msg = f"[TaskTree] Node {node.id} failed: {failure.summary} (root_cause={failure.root_cause.value})"
        logger.warning(msg)
        self._append_to_session({"role": "system", "content": msg})

    def on_node_blocked(self, node: TaskNode, failure: FailureReport) -> None:
        msg = f"[TaskTree] Node {node.id} blocked: {failure.summary} (constraint_veto={failure.constraint_veto})"
        logger.warning(msg)
        self._append_to_session({"role": "system", "content": msg})

    def save_checkpoint(self, tree: TaskTree) -> None:
        """Save the full tree state to session metadata for crash recovery."""
        try:
            session = self.session_manager.get_or_create(self.session_key)
            session.metadata[_TREE_CHECKPOINT_KEY] = tree.to_dict()
            self.session_manager.save(session)
            logger.debug("Tree checkpoint saved, {} nodes", len(tree.nodes))
        except Exception as e:
            logger.warning("Failed to save tree checkpoint: {}", e)

    def load_checkpoint(self) -> TaskTree | None:
        """Load a saved tree from session metadata, or None if not found."""
        try:
            session = self.session_manager.get_or_create(self.session_key)
            checkpoint = session.metadata.get(_TREE_CHECKPOINT_KEY)
            if checkpoint is None:
                return None
            return TaskTree.from_dict(checkpoint)
        except Exception as e:
            logger.warning("Failed to load tree checkpoint: {}", e)
            return None

    def clear_checkpoint(self) -> None:
        """Delete the saved tree checkpoint."""
        try:
            session = self.session_manager.get_or_create(self.session_key)
            if _TREE_CHECKPOINT_KEY in session.metadata:
                del session.metadata[_TREE_CHECKPOINT_KEY]
                self.session_manager.save(session)
                logger.debug("Tree checkpoint cleared for {}", self.session_key)
        except Exception as e:
            logger.warning("Failed to clear tree checkpoint: {}", e)

    def _append_to_session(self, message: dict) -> None:
        """Append a message to the current session."""
        try:
            session = self.session_manager.get_or_create(self.session_key)
            session.add_message(message["role"], message["content"])
            self.session_manager.save(session)
        except Exception as e:
            logger.warning("Failed to persist to session: {}", e)
