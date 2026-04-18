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
    from nanobot.agent.agemem.causal_store import CausalStore
    from nanobot.agent.agemem.store import MemoryStoreV2


class MemoryCallback:
    """SchedulerCallback that writes node summaries to AgeMem MemoryStoreV2 and CausalStore.

    Writes a MemoryEntry for each completed node, tagged with:
    - node_id
    - "tasktree" tag
    - "success" or "failure" tag

    Also writes a TimestampedFact to CausalStore with causal links if parent node exists.
    """

    def __init__(self, memory_store: MemoryStoreV2, causal_store: CausalStore | None = None):
        self.memory_store = memory_store
        self.causal_store = causal_store

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

        # Write causal fact
        if self.causal_store is not None:
            self._write_causal_fact(node, result=result, failure=None)

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

        # Write causal fact
        if self.causal_store is not None:
            self._write_causal_fact(node, result=None, failure=failure)

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

        # Write causal fact
        if self.causal_store is not None:
            self._write_causal_fact(node, result=None, failure=failure)

    def _write_causal_fact(
        self,
        node: TaskNode,
        result: NodeResult | None,
        failure: FailureReport | None,
    ) -> None:
        """Write a TimestampedFact for a TaskTree node."""
        try:
            from datetime import datetime

            if result is not None:
                content = {"node_id": node.id, "status": "success", "summary": result.summary}
                fact_type = "task_success"
                importance = 0.5
            elif failure is not None:
                content = {
                    "node_id": node.id,
                    "status": "failed",
                    "summary": failure.summary,
                    "root_cause": failure.root_cause.value,
                }
                fact_type = "task_failure"
                importance = 0.7
            else:
                return

            self.causal_store.add_fact(
                content=content,
                fact_type=fact_type,
                importance=importance,
                timestamp=datetime.now().isoformat(),
                tags=["tasktree", f"depth:{node.depth}"],
            )
            logger.debug("Wrote causal fact for node {}", node.id)

        except Exception as e:
            logger.warning("Failed to write causal fact for node {}: {}", node.id, e)

