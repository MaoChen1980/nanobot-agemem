"""Unit tests for CausalStore."""

from datetime import datetime
from pathlib import Path

import pytest

from nanobot.agent.agemem.causal_store import CausalStore
from nanobot.agent.agemem.fact import TimestampedFact


@pytest.fixture
def store(tmp_path: Path) -> CausalStore:
    return CausalStore(tmp_path)


class TestCausalStore_CRUD:
    def test_add_and_get(self, store: CausalStore):
        """add() stores a fact and get() retrieves it."""
        fact = TimestampedFact(
            id="fact-1",
            timestamp=datetime.now().isoformat(),
            type="action",
            content={"tool": "read_file"},
            importance=0.7,
        )
        store.add(fact)

        retrieved = store.get("fact-1")
        assert retrieved is not None
        assert retrieved.id == "fact-1"
        assert retrieved.type == "action"

    def test_get_nonexistent_returns_none(self, store: CausalStore):
        assert store.get("missing") is None

    def test_add_fact_convenience(self, store: CausalStore):
        """add_fact() creates and stores a fact."""
        fact = store.add_fact(
            content={"tool": "write_file"},
            fact_type="action",
            importance=0.6,
            tags=["test"],
        )
        assert fact.id is not None
        assert store.get(fact.id) is not None

    def test_get_all_sorted_by_timestamp(self, store: CausalStore):
        """get_all() returns facts sorted newest first."""
        store.add_fact(content={"t": "old"}, fact_type="event", timestamp="2026-01-01T00:00:00")
        store.add_fact(content={"t": "new"}, fact_type="event", timestamp="2026-04-18T00:00:00")
        store.add_fact(content={"t": "mid"}, fact_type="event", timestamp="2026-03-01T00:00:00")

        all_facts = store.get_all()
        timestamps = [f.timestamp for f in all_facts]
        assert timestamps == sorted(timestamps, reverse=True)


class TestCausalStore_CausalLinks:
    def test_link_causal_bidirectional(self, store: CausalStore):
        """link_causal() creates cause→effect links on both facts."""
        cause = store.add_fact(content={"a": 1}, fact_type="action", timestamp="2026-04-18T10:00:00")
        effect = store.add_fact(content={"b": 2}, fact_type="event", timestamp="2026-04-18T10:01:00")

        result = store.link_causal(cause.id, effect.id)
        assert result is True

        # Both facts should have updated links
        cause_updated = store.get(cause.id)
        effect_updated = store.get(effect.id)
        assert effect.id in cause_updated.effects
        assert cause.id in effect_updated.causes

    def test_link_causal_returns_false_for_missing(self, store: CausalStore):
        """link_causal() returns False if either fact is missing."""
        fact = store.add_fact(content={"x": 1}, fact_type="action")
        result = store.link_causal(fact.id, "missing-id")
        assert result is False
        result = store.link_causal("missing-id", fact.id)
        assert result is False

    def test_get_causes(self, store: CausalStore):
        """get_causes() returns direct cause facts."""
        c1 = store.add_fact(content={"c": 1}, fact_type="action", timestamp="2026-04-18T09:00:00")
        c2 = store.add_fact(content={"c": 2}, fact_type="action", timestamp="2026-04-18T09:30:00")
        e = store.add_fact(content={"e": 1}, fact_type="event", timestamp="2026-04-18T10:00:00")
        store.link_causal(c1.id, e.id)
        store.link_causal(c2.id, e.id)

        causes = store.get_causes(e.id)
        cause_ids = {c.id for c in causes}
        assert cause_ids == {c1.id, c2.id}

    def test_get_effects(self, store: CausalStore):
        """get_effects() returns direct effect facts."""
        c = store.add_fact(content={"c": 1}, fact_type="action", timestamp="2026-04-18T09:00:00")
        e1 = store.add_fact(content={"e": 1}, fact_type="event", timestamp="2026-04-18T10:00:00")
        e2 = store.add_fact(content={"e": 2}, fact_type="event", timestamp="2026-04-18T11:00:00")
        store.link_causal(c.id, e1.id)
        store.link_causal(c.id, e2.id)

        effects = store.get_effects(c.id)
        effect_ids = {e.id for e in effects}
        assert effect_ids == {e1.id, e2.id}

    def test_get_causal_chain_depth(self, store: CausalStore):
        """get_causal_chain() traverses up to depth levels."""
        # A → B → C (chain)
        a = store.add_fact(content={"s": "a"}, fact_type="action", timestamp="2026-04-18T09:00:00")
        b = store.add_fact(content={"s": "b"}, fact_type="action", timestamp="2026-04-18T10:00:00")
        c = store.add_fact(content={"s": "c"}, fact_type="event", timestamp="2026-04-18T11:00:00")
        store.link_causal(a.id, b.id)
        store.link_causal(b.id, c.id)

        # depth=1: only B is cause of C
        chain_1 = store.get_causal_chain(c.id, depth=1, direction="causes")
        assert {f.id for f in chain_1} == {b.id}

        # depth=2: A and B are causes of C
        chain_2 = store.get_causal_chain(c.id, depth=2, direction="causes")
        assert {f.id for f in chain_2} == {a.id, b.id}

        # depth=3: still only A and B
        chain_3 = store.get_causal_chain(c.id, depth=3, direction="causes")
        assert {f.id for f in chain_3} == {a.id, b.id}


class TestCausalStore_Queries:
    def test_query_by_time_range(self, store: CausalStore):
        """query_by_time_range() returns facts within [start, end]."""
        store.add_fact(content={"t": "early"}, fact_type="event", timestamp="2026-01-01T00:00:00")
        store.add_fact(content={"t": "mid"}, fact_type="event", timestamp="2026-04-01T00:00:00")
        store.add_fact(content={"t": "late"}, fact_type="event", timestamp="2026-04-18T00:00:00")

        results = store.query_by_time_range("2026-03-01T00:00:00", "2026-04-15T00:00:00")
        assert len(results) == 1
        assert results[0].content["t"] == "mid"

    def test_query_by_content(self, store: CausalStore):
        """query_by_content() matches exact key-value pairs."""
        store.add_fact(content={"tool": "read_file", "path": "a.txt"}, fact_type="action")
        store.add_fact(content={"tool": "write_file", "path": "b.txt"}, fact_type="action")
        store.add_fact(content={"tool": "read_file", "path": "c.txt"}, fact_type="action")

        results = store.query_by_content({"tool": "read_file"})
        assert len(results) == 2

        results = store.query_by_content({"tool": "read_file", "path": "a.txt"})
        assert len(results) == 1


class TestCausalStore_Persistence:
    def test_persists_to_disk(self, store: CausalStore, tmp_path: Path):
        """Facts are written to causal.jsonl."""
        store.add_fact(content={"x": 1}, fact_type="action")

        # New store instance reads from same path
        store2 = CausalStore(tmp_path)
        assert len(store2.get_all()) == 1

    def test_compaction_trims_low_importance(self, tmp_path: Path):
        """When over max_facts, lowest importance facts are removed."""
        small_store = CausalStore(tmp_path, max_facts=3)

        for i in range(5):
            small_store.add_fact(content={"i": i}, fact_type="event", importance=i / 10)

        assert len(small_store.get_all()) == 3
        # Highest importance (0.4, 0.3, 0.2) should remain
        remaining = {f.content["i"] for f in small_store.get_all()}
        assert remaining == {2, 3, 4}


class TestCausalStore_CausalLinksAdvanced:
    def test_get_causal_chain_both_directions(self, store: CausalStore):
        """get_causal_chain with direction='both' returns both causes and effects."""
        a = store.add_fact(content={"s": "a"}, fact_type="action", timestamp="2026-04-18T09:00:00")
        b = store.add_fact(content={"s": "b"}, fact_type="action", timestamp="2026-04-18T10:00:00")
        c = store.add_fact(content={"s": "c"}, fact_type="event", timestamp="2026-04-18T11:00:00")
        store.link_causal(a.id, b.id)
        store.link_causal(b.id, c.id)

        # From b, both causes (a) and effects (c) should be found
        chain = store.get_causal_chain(b.id, depth=1, direction="both")
        chain_ids = {f.id for f in chain}
        assert chain_ids == {a.id, c.id}

    def test_get_causal_chain_no_links(self, store: CausalStore):
        """Orphan fact with no causal links returns empty chain."""
        orphan = store.add_fact(content={"x": 1}, fact_type="action")
        chain = store.get_causal_chain(orphan.id, depth=3)
        assert chain == []

    def test_get_causes_empty_for_orphan(self, store: CausalStore):
        """get_causes returns empty list for fact with no causes."""
        fact = store.add_fact(content={"x": 1}, fact_type="action")
        assert store.get_causes(fact.id) == []

    def test_get_effects_empty_for_orphan(self, store: CausalStore):
        """get_effects returns empty list for fact with no effects."""
        fact = store.add_fact(content={"x": 1}, fact_type="action")
        assert store.get_effects(fact.id) == []


class TestCausalStore_QueryEdgeCases:
    def test_query_by_time_range_no_match(self, store: CausalStore):
        """Time range query with no matching facts returns empty."""
        store.add_fact(content={"t": "a"}, fact_type="event", timestamp="2026-01-01T00:00:00")
        results = store.query_by_time_range("2026-06-01T00:00:00", "2026-06-30T00:00:00")
        assert results == []

    def test_query_by_content_no_match(self, store: CausalStore):
        """Content query with no match returns empty."""
        store.add_fact(content={"tool": "read_file", "path": "a.txt"}, fact_type="action")
        results = store.query_by_content({"tool": "delete_file"})
        assert results == []

    def test_query_by_content_partial_match(self, store: CausalStore):
        """Content query requires all key-value pairs to match."""
        store.add_fact(content={"tool": "read_file", "path": "a.txt"}, fact_type="action")
        # Missing 'path' key
        results = store.query_by_content({"tool": "read_file", "path": None})
        assert results == []

    def test_query_by_time_range_limit(self, store: CausalStore):
        """Time range query respects limit parameter."""
        for i in range(5):
            store.add_fact(content={"i": i}, fact_type="event", timestamp=f"2026-04-{i+1:02d}T00:00:00")
        results = store.query_by_time_range("2026-01-01T00:00:00", "2026-12-31T00:00:00", limit=2)
        assert len(results) == 2
