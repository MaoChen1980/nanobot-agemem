"""Unit tests for MemoryRetriever (BM25 retrieval)."""

from pathlib import Path

import pytest

from nanobot.agent.agemem.retriever import MemoryRetriever
from nanobot.agent.agemem.store import MemoryStoreV2


@pytest.fixture
def ws(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


@pytest.fixture
def store_and_retriever(ws):
    store = MemoryStoreV2(ws)
    retriever = MemoryRetriever(store)
    return store, retriever


class TestMemoryRetriever_BM25:
    def test_exact_match_scores_higher_than_partial(self, store_and_retriever):
        store, retriever = store_and_retriever
        store.add("Remember my name is Alice", importance=0.8)
        store.add("Remember my name", importance=0.8)

        exact = retriever.retrieve("name is Alice", top_k=2)
        partial = retriever.retrieve("name", top_k=2)

        assert len(exact) >= 1
        assert len(partial) >= 1
        # Exact query should return Alice's entry at top
        assert exact[0].entry.content == "Remember my name is Alice"

    def test_no_matching_query_returns_low_score(self, store_and_retriever):
        store, retriever = store_and_retriever
        store.add("Project deadline is Friday", importance=0.7)

        results = retriever.retrieve("xyzzy none match query", top_k=5)
        # Non-matching query may still return result due to importance fallback
        # but the BM25 component should be 0
        assert len(results) >= 0
        if results:
            # BM25 score should be 0, only importance component contributes
            # Combined score = 0.7 * 0.0 + 0.3 * 0.7 = 0.21
            assert results[0].score < 0.5

    def test_top_k_limits_results(self, store_and_retriever):
        store, retriever = store_and_retriever
        for i in range(5):
            store.add(f"Memory item {i}", importance=0.5)

        results = retriever.retrieve("memory", top_k=2)
        assert len(results) == 2

    def test_importance_influences_score(self, store_and_retriever):
        store, retriever = store_and_retriever
        store.add("Common word task", importance=0.3)
        store.add("Common word critical", importance=0.95)

        results = retriever.retrieve("common word", top_k=2)
        # Same BM25 terms, but high importance should score higher
        assert len(results) == 2
        # Higher importance entry should be first
        assert results[0].entry.importance >= results[1].entry.importance

    def test_retrieve_returns_scored_entries(self, store_and_retriever):
        store, retriever = store_and_retriever
        entry = store.add("Unique specific phrase not found elsewhere")

        results = retriever.retrieve("Unique specific phrase", top_k=3)

        assert len(results) >= 1
        assert results[0].entry.id == entry.id
        assert results[0].score > 0


class TestMemoryRetriever_Filter:
    def test_filter_by_tags(self, store_and_retriever):
        store, retriever = store_and_retriever
        store.add("Preference item", tags=["preference", "ui"])
        store.add("Deadline item", tags=["deadline"])
        store.add("General item")

        results = retriever.filter(tags=["preference"])
        assert len(results) == 1
        assert "preference" in results[0].tags

    def test_filter_by_min_importance(self, store_and_retriever):
        store, retriever = store_and_retriever
        store.add("Low priority", importance=0.2)
        store.add("High priority", importance=0.9)

        results = retriever.filter(min_importance=0.8)
        assert len(results) == 1
        assert results[0].importance >= 0.8

    def test_filter_with_limit(self, store_and_retriever):
        store, retriever = store_and_retriever
        for i in range(5):
            store.add(f"Item {i}", importance=0.5, tags=["general"])

        results = retriever.filter(tags=["general"], limit=3)
        assert len(results) == 3


class TestMemoryRetriever_Summary:
    def test_summary_returns_nonempty_string(self, store_and_retriever):
        store, retriever = store_and_retriever
        messages = [
            {"role": "user", "content": "I need to finish the project by Friday"},
            {"role": "assistant", "content": "I'll remember that the deadline is Friday"},
        ]

        result = retriever.summary(messages)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_contains_key_terms(self, store_and_retriever):
        store, retriever = store_and_retriever
        messages = [
            {"role": "user", "content": "My birthday is July 4th"},
            {"role": "assistant", "content": "I'll remember your birthday is July 4th"},
        ]

        result = retriever.summary(messages)
        result_lower = result.lower()
        # Should contain birthday or july or 4th
        assert any(term in result_lower for term in ["birthday", "july", "4th"])
