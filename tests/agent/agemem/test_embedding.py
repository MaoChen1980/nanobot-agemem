"""Unit tests for embedding module (cosine similarity, EmbeddingIndex)."""

import pytest

from nanobot.agent.agemem.embedding import (
    EmbeddingIndex,
    batch_cosine_similarity,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Cosine similarity of identical vectors is 1.0."""
        v = [0.1, 0.2, 0.3, 0.4]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors is -1.0."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_negative_similarity(self):
        """Partially opposite vectors have negative similarity."""
        a = [1.0, 0.0]
        b = [-0.5, 0.866]
        sim = cosine_similarity(a, b)
        assert sim < 0

    def test_zero_vector(self):
        """Zero vector returns 0.0 (avoids division by zero)."""
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        """Mismatched vector dimensions raise ValueError."""
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


class TestBatchCosineSimilarity:
    def test_batch_with_multiple_documents(self):
        """batch_cosine_similarity returns one score per document."""
        query = [1.0, 0.0]
        docs = [
            [1.0, 0.0],  # identical to query
            [0.0, 1.0],  # orthogonal
            [-1.0, 0.0],  # opposite
        ]
        scores = batch_cosine_similarity(query, docs)

        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(-1.0)

    def test_empty_documents(self):
        """Empty document list returns empty list."""
        scores = batch_cosine_similarity([1.0, 0.0], [])
        assert scores == []


class TestEmbeddingIndex:
    def test_add_and_search(self):
        """add() stores embeddings, search() returns top-k by similarity."""
        idx = EmbeddingIndex()

        idx.add("id-1", [1.0, 0.0])
        idx.add("id-2", [0.0, 1.0])
        idx.add("id-3", [0.707, 0.707])  # ~45 degrees

        # Query with id-1 direction
        results = idx.search([1.0, 0.0], top_k=2)

        assert len(results) == 2
        # id-1 should be top (identical)
        assert results[0][0] == "id-1"
        assert results[0][1] == pytest.approx(1.0)
        # id-3 should be second (positive correlation)
        assert results[1][0] == "id-3"

    def test_search_empty_index(self):
        """search() on empty index returns empty list."""
        idx = EmbeddingIndex()
        results = idx.search([1.0, 0.0])
        assert results == []

    def test_search_top_k_limits(self):
        """top_k parameter limits results."""
        idx = EmbeddingIndex()
        for i in range(5):
            idx.add(f"id-{i}", [float(i), 0.0])

        results = idx.search([1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_len(self):
        """__len__ returns correct count."""
        idx = EmbeddingIndex()
        assert len(idx) == 0

        idx.add("a", [1.0])
        idx.add("b", [0.5])
        assert len(idx) == 2

    def test_clear(self):
        """clear() empties the index."""
        idx = EmbeddingIndex()
        idx.add("a", [1.0])
        idx.add("b", [0.5])

        idx.clear()
        assert len(idx) == 0
        assert idx.search([1.0]) == []

    def test_search_sorted_by_score_descending(self):
        """Results are sorted highest similarity first."""
        idx = EmbeddingIndex()
        # Various angles from query [1, 0]
        idx.add("worst", [-1.0, 0.0])  # similarity = -1
        idx.add("neutral", [0.0, 1.0])  # similarity = 0
        idx.add("best", [1.0, 0.0])  # similarity = 1

        results = idx.search([1.0, 0.0], top_k=3)

        assert results[0][0] == "best"
        assert results[1][0] == "neutral"
        assert results[2][0] == "worst"
        assert results[0][1] > results[1][1] > results[2][1]
