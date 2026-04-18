"""Embedding wrapper and cosine similarity for AgeMem.

Provides:
- Text embedding via OpenAI or Anthropic API
- Cosine similarity between embedding vectors
- Top-k retrieval by embedding similarity
"""

import math
from typing import Any

# Note: Import openai lazily to avoid requiring it when not used
_client: Any = None


def _get_client():
    """Get or create OpenAI embedding client (lazy init)."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI()
        except ImportError:
            return None
    return _client


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]] | None:
    """Get embeddings for a list of texts via OpenAI API.

    Args:
        texts: List of strings to embed
        model: Embedding model name

    Returns:
        List of embedding vectors, or None if API unavailable
    """
    client = _get_client()
    if client is None:
        return None

    try:
        response = client.embeddings.create(
            model=model,
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception:
        return None


def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float] | None:
    """Get embedding for a single text."""
    results = embed_texts([text], model)
    return results[0] if results else None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity = (A · B) / (||A|| × ||B||)
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def batch_cosine_similarity(query_embedding: list[float], document_embeddings: list[list[float]]) -> list[float]:
    """Compute cosine similarity between a query and multiple documents.

    Args:
        query_embedding: Query vector
        document_embeddings: List of document vectors

    Returns:
        List of similarity scores (same order as input)
    """
    return [cosine_similarity(query_embedding, doc_emb) for doc_emb in document_embeddings]


class EmbeddingIndex:
    """Simple in-memory embedding index for fact content.

    Stores fact IDs with their embeddings for fast similarity search.
    """

    def __init__(self):
        self._ids: list[str] = []
        self._embeddings: list[list[float]] = []

    def add(self, id: str, embedding: list[float]) -> None:
        """Add a fact embedding to the index."""
        self._ids.append(id)
        self._embeddings.append(embedding)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Find top-k facts by embedding similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of (fact_id, similarity_score) tuples, sorted by score desc
        """
        if not self._embeddings:
            return []

        scores = batch_cosine_similarity(query_embedding, self._embeddings)
        indexed = list(zip(self._ids, scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def clear(self) -> None:
        """Clear the index."""
        self._ids.clear()
        self._embeddings.clear()

    def __len__(self) -> int:
        return len(self._ids)
