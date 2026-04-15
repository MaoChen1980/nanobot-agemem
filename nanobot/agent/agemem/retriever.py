"""Memory retriever for AgeMem: semantic search, summarization, and filtering.

Provides embedding-based retrieval over structured LTM entries, replacing
the flat MEMORY.md injection with selective recall.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from nanobot.agent.agemem.entry import MemoryEntry
from nanobot.agent.agemem.store import MemoryStoreV2


@dataclass
class ScoredEntry:
    """A memory entry with its relevance score."""
    entry: MemoryEntry
    score: float


class MemoryRetriever:
    """Semantic retriever over MemoryStoreV2 using BM25-style scoring.

    Phase 2: Uses keyword-based BM25 scoring (no external embedder required).
    Phase 3: Swap to EmbeddingProvider for true semantic embeddings.
    """

    def __init__(self, store: MemoryStoreV2):
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredEntry]:
        """Find memories most relevant to the query using BM25 scoring.

        Returns top-k entries sorted by relevance score (descending).
        """
        entries = self._store.get_all()
        if not entries:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            # No query terms: return top by importance
            return [ScoredEntry(e, e.importance) for e in entries[:top_k]]

        # Build BM25 scores
        scored = []
        avg_dl = sum(len(self._tokenize(e.content)) for e in entries) / max(len(entries), 1)

        for entry in entries:
            doc_terms = self._tokenize(entry.content)
            if not doc_terms:
                continue
            score = self._bm25_score(query_terms, doc_terms, avg_dl, len(entries))
            # Combine BM25 with importance as a tiebreaker
            combined = 0.7 * score + 0.3 * entry.importance
            scored.append(ScoredEntry(entry, combined))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def summary(self, messages: list[dict[str, Any]], max_length: int = 500) -> str:
        """Summarize a list of session messages into a concise memory string.

        This is the STM Summary operation. In Phase 2 it uses a simple
        extraction-based approach. Phase 3 would call the LLM for abstractive
        summarization.
        """
        if not messages:
            return ""

        # Simple extraction-based summarization
        key_sentences: list[str] = []
        all_text = ""

        for msg in messages:
            content = msg.get("content", "") or ""
            if not content or not isinstance(content, str):
                continue
            # Strip think tags
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = content.strip()
            if content:
                all_text += " " + content

        sentences = re.split(r"[.!?\n]+", all_text)
        sentence_scores: list[tuple[str, float]] = []

        query_terms = set(self._tokenize(all_text[:1000]))  # Use first 1000 chars as pseudo-query

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            sent_terms = self._tokenize(sent)
            if not sent_terms:
                continue
            # Score by term overlap with itself (TF) and length penalty
            tf = sum(1 for t in sent_terms if t in query_terms)
            len_bonus = min(len(sent_terms) / 20.0, 1.0)  # prefer medium-length sentences
            score = tf * (0.5 + 0.5 * len_bonus)
            if score > 0:
                sentence_scores.append((sent, score))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        result_parts = []
        total_len = 0
        for sent, _ in sentence_scores:
            if total_len + len(sent) > max_length:
                break
            result_parts.append(sent)
            total_len += len(sent) + 1

        summary = ". ".join(result_parts)
        if summary and not summary.endswith("."):
            summary += "."
        return summary

    def filter(
        self,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Filter memories by tags and/or minimum importance (STM Filter operation)."""
        return self._store.query(tags=tags, min_importance=min_importance, limit=limit)

    # -- BM25 internals -------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, extract alphanumeric tokens."""
        tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
        return tokens

    def _bm25_score(
        self,
        query_terms: list[str],
        doc_terms: list[str],
        avg_dl: float,
        N: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """Compute BM25 score for a document given a query."""
        doc_len = len(doc_terms)
        doc_tf = Counter(doc_terms)

        score = 0.0
        for term in query_terms:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            df = sum(1 for e in self._store.get_all() if term in self._tokenize(e.content))
            if df == 0:
                df = 1

            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / max(avg_dl, 1))
            score += idf * numerator / denominator

        return score
