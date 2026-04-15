"""Unit tests for MemoryCallback."""

from __future__ import annotations

import pytest

from nanobot.agent.tasktree.memory_callback import MemoryCallback
from nanobot.agent.tasktree.models import (
    FailureReport,
    NodeResult,
    RootCause,
    TaskNode,
    TaskStatus,
)


class FakeMemoryStore:
    def __init__(self):
        self.entries = []

    def add(self, content: str, importance: float, tags: list[str]):
        self.entries.append({
            "content": content,
            "importance": importance,
            "tags": tags,
        })


class TestMemoryCallback:
    def test_on_node_start_is_noop(self):
        """on_node_start should not raise or call memory_store."""
        store = FakeMemoryStore()
        cb = MemoryCallback(store)
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        cb.on_node_start(node)  # Should not raise
        assert store.entries == []

    def test_on_node_done_writes_memory(self):
        """on_node_done writes success entry to memory store."""
        store = FakeMemoryStore()
        cb = MemoryCallback(store)
        node = TaskNode(id="root.0", goal="subtask", parent_id="root", depth=1)
        result = NodeResult(node_id="root.0", summary="analyzed requirements")

        cb.on_node_done(node, result)

        assert len(store.entries) == 1
        entry = store.entries[0]
        assert "root.0" in entry["content"]
        assert "analyzed requirements" in entry["content"]
        assert entry["importance"] == 0.5
        assert "tasktree" in entry["tags"]
        assert "success" in entry["tags"]
        assert "depth:1" in entry["tags"]

    def test_on_node_failed_writes_memory(self):
        """on_node_failed writes failure entry with root_cause tag."""
        store = FakeMemoryStore()
        cb = MemoryCallback(store)
        node = TaskNode(id="root.0", goal="subtask", parent_id="root", depth=1)
        failure = FailureReport(
            node_id="root.0",
            status=TaskStatus.FAILED,
            root_cause=RootCause.API_TIMEOUT,
            summary="request timed out",
        )

        cb.on_node_failed(node, failure)

        assert len(store.entries) == 1
        entry = store.entries[0]
        assert "root.0" in entry["content"]
        assert "FAILED" in entry["content"]
        assert "request timed out" in entry["content"]
        assert entry["importance"] == 0.7
        assert "failure" in entry["tags"]
        assert "api_timeout" in entry["tags"]
        assert "depth:1" in entry["tags"]

    def test_on_node_blocked_writes_memory(self):
        """on_node_blocked writes blocked entry."""
        store = FakeMemoryStore()
        cb = MemoryCallback(store)
        node = TaskNode(id="root.0", goal="subtask", parent_id="root", depth=1)
        failure = FailureReport(
            node_id="root.0",
            status=TaskStatus.BLOCKED,
            root_cause=RootCause.CONSTRAINT_VETO,
            summary="depth exceeded",
            constraint_veto=True,
        )

        cb.on_node_blocked(node, failure)

        assert len(store.entries) == 1
        entry = store.entries[0]
        assert "BLOCKED" in entry["content"]
        assert "depth exceeded" in entry["content"]
        assert entry["importance"] == 0.7
        assert "blocked" in entry["tags"]
        assert "constraint_veto" in entry["tags"]

    def test_on_node_done_catches_exception(self):
        """on_node_done suppresses memory store errors."""
        class BadStore:
            def add(self, content, importance, tags):
                raise RuntimeError("store error")

        cb = MemoryCallback(BadStore())
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        result = NodeResult(node_id="root", summary="done")
        # Should not raise
        cb.on_node_done(node, result)

    def test_on_node_failed_catches_exception(self):
        """on_node_failed suppresses memory store errors."""
        class BadStore:
            def add(self, content, importance, tags):
                raise RuntimeError("store error")

        cb = MemoryCallback(BadStore())
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        failure = FailureReport(
            node_id="root", status=TaskStatus.FAILED,
            root_cause=RootCause.UNKNOWN, summary="failed",
        )
        cb.on_node_failed(node, failure)  # Should not raise

    def test_on_node_blocked_catches_exception(self):
        """on_node_blocked suppresses memory store errors."""
        class BadStore:
            def add(self, content, importance, tags):
                raise RuntimeError("store error")

        cb = MemoryCallback(BadStore())
        node = TaskNode(id="root", goal="test", parent_id=None, depth=0)
        failure = FailureReport(
            node_id="root", status=TaskStatus.BLOCKED,
            root_cause=RootCause.CONSTRAINT_VETO, summary="blocked",
        )
        cb.on_node_blocked(node, failure)  # Should not raise

    def test_multiple_callbacks_all_write(self):
        """Multiple callback invocations each write one entry."""
        store = FakeMemoryStore()
        cb = MemoryCallback(store)

        node1 = TaskNode(id="root.0", goal="task1", parent_id="root", depth=1)
        node2 = TaskNode(id="root.1", goal="task2", parent_id="root", depth=1)

        cb.on_node_done(node1, NodeResult(node_id="root.0", summary="first done"))
        cb.on_node_done(node2, NodeResult(node_id="root.1", summary="second done"))

        assert len(store.entries) == 2
        assert "first done" in store.entries[0]["content"]
        assert "second done" in store.entries[1]["content"]
