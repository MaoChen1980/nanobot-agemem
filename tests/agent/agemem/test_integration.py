"""Integration tests for causal memory pipeline.

Tests the full flow:
1. Session messages with tool_calls → extract_tool_call_pairs
2. → CausalStore (via Consolidator._extract_and_store_causal_facts)
3. → query (get_all, query_by_content, causal chain)
4. MemoryCallback with CausalStore → writes TaskTree facts
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.agemem.causal_store import CausalStore
from nanobot.agent.agemem.extractor import extract_tool_call_pairs, extract_facts_from_pairs
from nanobot.agent.tasktree.memory_callback import MemoryCallback
from nanobot.agent.tasktree.models import FailureReport, NodeResult, RootCause, TaskNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeMemoryStore:
    """Fake MemoryStoreV2 for testing MemoryCallback."""

    def __init__(self):
        self.entries = []

    def add(self, content, importance, tags=None):
        self.entries.append({"content": content, "importance": importance, "tags": tags or []})


# ---------------------------------------------------------------------------
# Integration: extract → CausalStore → query
# ---------------------------------------------------------------------------


class TestCausalMemoryPipeline:
    """Full pipeline: messages → extractor → CausalStore → query."""

    def test_messages_to_facts_to_store_to_query(self, tmp_path: Path):
        """End-to-end: messages → extract → store → retrieve."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "read_file", "arguments": '{"path": "config.json"}'},
                    }
                ],
                "timestamp": "2026-04-18T10:00:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"host": "localhost", "port": 8080}',
                "timestamp": "2026-04-18T10:00:01",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "function": {"name": "write_file", "arguments": '{"path": "out.json", "content": "ok"}'},
                    }
                ],
                "timestamp": "2026-04-18T10:00:02",
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "content": "Successfully wrote out.json",
                "timestamp": "2026-04-18T10:00:03",
            },
        ]

        # Step 1: Extract tool_call pairs
        pairs = extract_tool_call_pairs(messages)
        assert len(pairs) == 2
        assert pairs[0].tool_name == "read_file"
        assert pairs[1].tool_name == "write_file"

        # Step 2: Convert to facts
        facts = extract_facts_from_pairs(pairs, importance=0.5)
        assert len(facts) == 2
        assert facts[0]["fact_type"] == "tool_call"
        assert facts[0]["content"]["tool"] == "read_file"
        assert facts[1]["content"]["tool"] == "write_file"

        # Step 3: Store in CausalStore
        store = CausalStore(tmp_path)
        for fd in facts:
            store.add_fact(
                content=fd["content"],
                fact_type=fd["fact_type"],
                importance=fd["importance"],
                timestamp=fd["timestamp"],
                tags=fd["tags"],
            )

        # Step 4: Query by content
        read_facts = store.query_by_content({"tool": "read_file"})
        assert len(read_facts) == 1
        assert read_facts[0].content["input"]["path"] == "config.json"

        # Step 5: Query by time range
        recent = store.query_by_time_range("2026-04-18T09:00:00", "2026-04-18T11:00:00")
        assert len(recent) == 2

        # Step 6: Query all
        all_facts = store.get_all()
        assert len(all_facts) == 2

    def test_facts_have_timestamps(self, tmp_path: Path):
        """Facts extracted from messages carry ISO timestamps."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-x", "function": {"name": "api_call", "arguments": "{}"}},
                ],
                "timestamp": "2026-04-18T15:30:00",
            },
            {
                "role": "tool",
                "tool_call_id": "call-x",
                "content": '{"status": "ok"}',
                "timestamp": "2026-04-18T15:30:01",
            },
        ]
        pairs = extract_tool_call_pairs(messages)
        facts = extract_facts_from_pairs(pairs)

        assert facts[0]["timestamp"] == "2026-04-18T15:30:00"
        assert facts[0]["content"]["tool"] == "api_call"

    def test_empty_messages_produces_empty_facts(self):
        """Empty message list produces empty facts list."""
        pairs = extract_tool_call_pairs([])
        assert pairs == []
        facts = extract_facts_from_pairs(pairs)
        assert facts == []


# ---------------------------------------------------------------------------
# Integration: MemoryCallback with CausalStore
# ---------------------------------------------------------------------------


class TestMemoryCallbackCausalStore:
    """MemoryCallback writes TaskTree node results to CausalStore."""

    def test_on_node_done_writes_causal_fact(self, tmp_path: Path):
        """on_node_done writes a TimestampedFact to CausalStore."""
        store = CausalStore(tmp_path)
        memory_store = FakeMemoryStore()
        cb = MemoryCallback(memory_store=memory_store, causal_store=store)

        node = TaskNode(id="root.1", goal="subtask", parent_id="root", depth=1)
        result = NodeResult(node_id="root.1", summary="read requirements doc")

        cb.on_node_done(node, result)

        # Should write to CausalStore
        facts = store.get_all()
        assert len(facts) == 1
        assert facts[0].type == "task_success"
        assert facts[0].content["node_id"] == "root.1"
        assert facts[0].content["status"] == "success"
        assert facts[0].content["summary"] == "read requirements doc"
        assert "tasktree" in facts[0].tags

    def test_on_node_failed_writes_failure_fact(self, tmp_path: Path):
        """on_node_failed writes a failure fact with root_cause."""
        store = CausalStore(tmp_path)
        memory_store = FakeMemoryStore()
        cb = MemoryCallback(memory_store=memory_store, causal_store=store)

        node = TaskNode(id="root.2", goal="fetch data", parent_id="root", depth=1)
        failure = FailureReport(
            node_id="root.2",
            status=3,  # FAILED
            root_cause=RootCause.API_TIMEOUT,
            summary="request timed out after 30s",
        )

        cb.on_node_failed(node, failure)

        facts = store.get_all()
        assert len(facts) == 1
        assert facts[0].type == "task_failure"
        assert facts[0].content["status"] == "failed"
        assert facts[0].content["root_cause"] == "api_timeout"
        assert facts[0].importance == 0.7  # failures are more important

    def test_on_node_blocked_writes_blocked_fact(self, tmp_path: Path):
        """on_node_blocked writes a blocked fact."""
        store = CausalStore(tmp_path)
        memory_store = FakeMemoryStore()
        cb = MemoryCallback(memory_store=memory_store, causal_store=store)

        node = TaskNode(id="root.3", goal="delete", parent_id="root", depth=2)
        failure = FailureReport(
            node_id="root.3",
            status=4,  # BLOCKED
            root_cause=RootCause.CONSTRAINT_VETO,
            summary="depth exceeded",
            constraint_veto=True,
        )

        cb.on_node_blocked(node, failure)

        facts = store.get_all()
        assert len(facts) == 1
        assert facts[0].type == "task_failure"
        assert facts[0].content["status"] == "failed"

    def test_causal_store_optional_memory_callback(self):
        """MemoryCallback without CausalStore doesn't crash (graceful degradation)."""
        memory_store = FakeMemoryStore()
        cb = MemoryCallback(memory_store=memory_store, causal_store=None)

        node = TaskNode(id="root.1", goal="test", parent_id="root", depth=1)
        result = NodeResult(node_id="root.1", summary="done")

        # Should not raise even though causal_store is None
        cb.on_node_done(node, result)

        # MemoryStoreV2 should still be written
        assert len(memory_store.entries) == 1

    def test_multiple_nodes_produce_multiple_facts(self, tmp_path: Path):
        """Multiple TaskTree nodes each produce a separate causal fact."""
        store = CausalStore(tmp_path)
        memory_store = FakeMemoryStore()
        cb = MemoryCallback(memory_store=memory_store, causal_store=store)

        node1 = TaskNode(id="root.1", goal="task1", parent_id="root", depth=1)
        node2 = TaskNode(id="root.2", goal="task2", parent_id="root", depth=1)

        cb.on_node_done(node1, NodeResult(node_id="root.1", summary="step 1 complete"))
        cb.on_node_done(node2, NodeResult(node_id="root.2", summary="step 2 complete"))

        facts = store.get_all()
        assert len(facts) == 2
        node_ids = {f.content["node_id"] for f in facts}
        assert node_ids == {"root.1", "root.2"}


# ---------------------------------------------------------------------------
# Integration: CausalStore persistence
# ---------------------------------------------------------------------------


class TestCausalStorePersistence:
    """CausalStore survives restart."""

    def test_facts_persist_across_store_instances(self, tmp_path: Path):
        """New CausalStore instance reads previously written facts."""
        store1 = CausalStore(tmp_path)
        store1.add_fact(content={"tool": "read_file"}, fact_type="tool_call")

        store2 = CausalStore(tmp_path)
        facts = store2.get_all()

        assert len(facts) == 1
        assert facts[0].content["tool"] == "read_file"

    def test_causal_links_persist(self, tmp_path: Path):
        """Causal links survive store restart."""
        store1 = CausalStore(tmp_path)
        a = store1.add_fact(content={"s": "a"}, fact_type="action")
        b = store1.add_fact(content={"s": "b"}, fact_type="event")
        store1.link_causal(a.id, b.id)

        store2 = CausalStore(tmp_path)
        b_facts = store2.get_causes(b.id)
        assert len(b_facts) == 1
        assert b_facts[0].id == a.id
