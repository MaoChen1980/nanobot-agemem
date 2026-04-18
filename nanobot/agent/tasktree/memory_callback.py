"""Memory integration: writes node summaries to AgeMem on completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.tasktree.models import (
    FailureReport,
    NodeResult,
    TaskNode,
)
from nanobot.agent.tasktree.scheduler import SchedulerCallbacks

if TYPE_CHECKING:
    from nanobot.agent.agemem.store import MemoryStoreV2


class MemoryCallback:
    """SchedulerCallback that writes node summaries to AgeMem MemoryStoreV2.

    Writes a MemoryEntry for each completed node, tagged with:
    - node_id
    - "tasktree" tag
    - "success" or "failure" tag
    """

    def __init__(self, memory_store: MemoryStoreV2):
        self.memory_store = memory_store

    def on_node_start(self, node: TaskNode) -> None:
        pass  # No-op for memory

    def on_node_done(self, node: TaskNode, result: NodeResult) -> None:
        try:
            self.memory_store.add(
                content={"text": f"[TaskTree Node {node.id}] {result.summary}"},
                importance=0.5,
                tags=["tasktree", "success", f"depth:{node.depth}"],
            )
            logger.debug("Wrote success memory for node {}", node.id)
        except Exception as e:
            logger.warning("Failed to write success memory for node {}: {}", node.id, e)

    def on_node_failed(self, node: TaskNode, failure: FailureReport) -> None:
        try:
            self.memory_store.add(
                content={"text": f"[TaskTree Node {node.id} FAILED] {failure.summary} (root_cause={failure.root_cause.value})"},
                importance=0.7,  # Failures are more important to remember
                tags=["tasktree", "failure", failure.root_cause.value, f"depth:{node.depth}"],
            )
            logger.debug("Wrote failure memory for node {}", node.id)
        except Exception as e:
            logger.warning("Failed to write failure memory for node {}: {}", node.id, e)

    def on_node_blocked(self, node: TaskNode, failure: FailureReport) -> None:
        try:
            self.memory_store.add(
                content={"text": f"[TaskTree Node {node.id} BLOCKED] {failure.summary} (constraint_veto={failure.constraint_veto})"},
                importance=0.7,
                tags=["tasktree", "blocked", failure.root_cause.value, f"depth:{node.depth}"],
            )
            logger.debug("Wrote blocked memory for node {}", node.id)
        except Exception as e:
            logger.warning("Failed to write blocked memory for node {}: {}", node.id, e)
