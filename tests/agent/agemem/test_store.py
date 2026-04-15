"""Unit tests for MemoryStoreV2."""

from pathlib import Path

import pytest

from nanobot.agent.agemem.store import MemoryStoreV2


@pytest.fixture
def ws(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


class TestMemoryStoreV2_CRUD:
    def test_add_returns_entry_with_id(self, ws):
        store = MemoryStoreV2(ws)
        entry = store.add("Remember my name is Bob", importance=0.8, tags=["name"])

        assert entry.id != ""
        assert entry.content == "Remember my name is Bob"
        assert entry.importance == 0.8
        assert "name" in entry.tags

    def test_get_retrieves_added_entry(self, ws):
        store = MemoryStoreV2(ws)
        added = store.add("Test content")
        retrieved = store.get(added.id)

        assert retrieved is not None
        assert retrieved.id == added.id
        assert retrieved.content == "Test content"

    def test_get_nonexistent_returns_none(self, ws):
        store = MemoryStoreV2(ws)
        result = store.get("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_update_changes_fields(self, ws):
        store = MemoryStoreV2(ws)
        added = store.add("Original content", importance=0.5)
        updated = store.update(added.id, importance=0.9, content="Updated content")

        assert updated is not None
        assert updated.importance == 0.9
        assert updated.content == "Updated content"
        assert updated.updated_at != added.created_at

    def test_update_nonexistent_returns_none(self, ws):
        store = MemoryStoreV2(ws)
        result = store.update("00000000-0000-0000-0000-000000000000", importance=0.9)
        assert result is None

    def test_soft_delete_removes_from_get(self, ws):
        store = MemoryStoreV2(ws)
        added = store.add("To be deleted")
        deleted = store.delete(added.id)

        assert deleted is True
        assert store.get(added.id) is None

    def test_soft_delete_excluded_from_get_all(self, ws):
        store = MemoryStoreV2(ws)
        added = store.add("To be deleted")
        store.delete(added.id)

        all_entries = store.get_all()
        ids = [e.id for e in all_entries]
        assert added.id not in ids

    def test_get_all_includes_active_entries(self, ws):
        store = MemoryStoreV2(ws)
        e1 = store.add("Entry 1")
        e2 = store.add("Entry 2")
        store.delete(e1.id)

        all_entries = store.get_all()
        ids = [e.id for e in all_entries]
        assert e2.id in ids
        assert e1.id not in ids

    def test_record_access_increments_count(self, ws):
        store = MemoryStoreV2(ws)
        added = store.add("Frequently used")
        assert added.access_count == 0

        store.record_access(added.id)
        store.record_access(added.id)

        retrieved = store.get(added.id)
        assert retrieved.access_count == 2


class TestMemoryStoreV2_Query:
    def test_query_by_tag(self, ws):
        store = MemoryStoreV2(ws)
        store.add("Preference item", tags=["preference"])
        store.add("Other item")

        results = store.query(tags=["preference"])
        assert len(results) == 1
        assert "preference" in results[0].tags

    def test_query_by_min_importance(self, ws):
        store = MemoryStoreV2(ws)
        store.add("Low", importance=0.2)
        store.add("High", importance=0.9)

        results = store.query(min_importance=0.8)
        assert len(results) == 1
        assert results[0].importance >= 0.8

    def test_query_with_limit(self, ws):
        store = MemoryStoreV2(ws)
        for i in range(5):
            store.add(f"Entry {i}", importance=0.5)

        results = store.query(limit=2)
        assert len(results) == 2


class TestMemoryStoreV2_Compaction:
    def test_auto_compact_on_add_exceeds_max(self, ws):
        # compaction is triggered automatically inside add() when exceeding max_entries
        store = MemoryStoreV2(ws, max_entries=3)

        # Add 5 entries with varying importance
        for i in range(5):
            store.add(f"Entry {i}", importance=0.5 - i * 0.1)

        # After adding 5th entry (exceeds max_entries=3), auto-compact runs
        remaining = store.get_all()
        assert len(remaining) <= 3

    def test_high_importance_survives_auto_compact(self, ws):
        store = MemoryStoreV2(ws, max_entries=2)
        low = store.add("Low importance", importance=0.1)
        high = store.add("High importance", importance=0.95)

        # Auto-compact triggered when adding second entry (exceeds max=2)
        ids = [e.id for e in store.get_all()]
        assert high.id in ids
        # Low importance may or may not survive depending on ordering


class TestMemoryStoreV2_Persistence:
    def test_cross_instance_reload(self, ws):
        store1 = MemoryStoreV2(ws)
        e = store1.add("Persisted content", importance=0.7, tags=["test"])

        store2 = MemoryStoreV2(ws)
        reloaded = store2.get(e.id)

        assert reloaded is not None
        assert reloaded.content == "Persisted content"
        assert reloaded.importance == 0.7
        assert "test" in reloaded.tags

    def test_adds_persist_to_disk(self, ws):
        store = MemoryStoreV2(ws)
        entry = store.add("File test", importance=0.6)

        ltm_file = ws / "memory" / "ltm.jsonl"
        assert ltm_file.exists()
        content = ltm_file.read_text(encoding="utf-8")
        assert "File test" in content
        assert entry.id[:8] in content
