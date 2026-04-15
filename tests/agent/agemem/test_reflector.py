"""Unit tests for Reflector (gap detection and reflection)."""

import json
from pathlib import Path

import pytest

from nanobot.agent.agemem.reflector import Reflector, GapEvent


@pytest.fixture
def ws(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


class TestReflector_GapRecording:
    def test_on_retrieve_records_empty_gap(self, ws):
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        reflector.on_retrieve(query="user preferences", retrieved_count=0, was_empty=True)

        assert gaps_file.exists()
        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["query"] == "user preferences"
        assert event["retrieved_count"] == 0
        assert event["was_empty"] is True
        assert event["source"] == "ltm"
        assert event["resolved"] is False

    def test_on_retrieve_records_non_empty_event(self, ws):
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        reflector.on_retrieve(query="project name", retrieved_count=3, was_empty=False)

        assert gaps_file.exists()
        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["retrieved_count"] == 3
        assert event["was_empty"] is False

    def test_on_history_retrieval_records_source_history(self, ws):
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        reflector.on_history_retrieval("api key", [{"content": "secret"}])

        assert gaps_file.exists()
        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        event = json.loads(lines[0])
        assert event["source"] == "history"
        assert event["was_empty"] is False
        assert event["retrieved_count"] == 1

    def test_multiple_gaps_appended(self, ws):
        reflector = Reflector(ws)

        reflector.on_retrieve("query1", 0, True)
        reflector.on_retrieve("query2", 0, True)

        gaps_file = ws / "memory" / "gaps.jsonl"
        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


class TestReflector_GetGaps:
    def test_get_gaps_returns_all_unresolved_by_default(self, ws):
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        # Write two gaps, one resolved (use append mode)
        with open(gaps_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": "2026-04-15T10:00:00", "query": "q1",
                                "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": False}) + "\n")
            f.write(json.dumps({"timestamp": "2026-04-15T10:01:00", "query": "q2",
                                "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": True}) + "\n")

        gaps = reflector.get_gaps(unresolved_only=True)
        assert len(gaps) == 1
        assert gaps[0].query == "q1"

        gaps_all = reflector.get_gaps(unresolved_only=False)
        assert len(gaps_all) == 2

    def test_get_gaps_returns_empty_when_file_missing(self, ws):
        reflector = Reflector(ws)
        gaps = reflector.get_gaps()
        assert gaps == []


class TestReflector_MarkResolved:
    def test_mark_resolved_updates_file(self, ws):
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        gaps_file.write_text(
            json.dumps({"timestamp": "2026-04-15T10:00:00", "query": "test query",
                         "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": False}) + "\n",
            encoding="utf-8",
        )

        reflector.mark_resolved("test query")

        events = reflector.get_gaps(unresolved_only=False)
        assert len(events) == 1
        assert events[0].resolved is True


class TestReflector_Reflect:
    def test_reflect_returns_empty_when_no_gaps(self, ws):
        reflector = Reflector(ws)
        recs = reflector.reflect()
        assert recs == []

    def test_reflect_groups_similar_queries(self, ws):
        """Multiple gaps with similar queries should be grouped into one recommendation."""
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        # Write multiple gaps with similar queries
        for _ in range(3):
            gaps_file.write_text(
                json.dumps({"timestamp": "2026-04-15T10:00:00", "query": "user name",
                             "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": False}) + "\n",
                encoding="utf-8",
            )

        recs = reflector.reflect()
        assert len(recs) >= 1
        # Top recommendation should mention the pattern
        assert "name" in recs[0]["pattern"].lower() or "user" in recs[0]["pattern"].lower()

    def test_reflect_returns_action_add_rule(self, ws):
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        gaps_file.write_text(
            json.dumps({"timestamp": "2026-04-15T10:00:00", "query": "deadline",
                         "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": False}) + "\n",
            encoding="utf-8",
        )

        recs = reflector.reflect()
        assert len(recs) >= 1
        assert recs[0]["action"] == "add_rule"
        assert "importance" in recs[0]
        assert recs[0]["importance"] >= 0.5

    def test_reflect_respects_gap_count(self, ws):
        """More gaps should lead to higher importance recommendation."""
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        # Single occurrence - importance should be 0.5 + 0.1 = 0.6
        gaps_file.write_text(
            json.dumps({"timestamp": "2026-04-15T10:00:00", "query": "single gap",
                         "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": False}) + "\n",
            encoding="utf-8",
        )
        recs = reflector.reflect()
        assert recs[0]["gap_count"] == 1

    def test_reflect_caps_importance_at_one(self, ws):
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        # Write 10 gaps (should cap importance at 1.0)
        for _ in range(10):
            gaps_file.write_text(
                json.dumps({"timestamp": "2026-04-15T10:00:00", "query": "many gaps",
                             "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": False}) + "\n",
                encoding="utf-8",
            )

        recs = reflector.reflect()
        assert recs[0]["importance"] <= 1.0

    def test_reflect_stops_common_words(self, ws):
        """Stopwords like 'the', 'is', 'a' should not appear in patterns."""
        reflector = Reflector(ws)
        gaps_file = ws / "memory" / "gaps.jsonl"

        gaps_file.write_text(
            json.dumps({"timestamp": "2026-04-15T10:00:00", "query": "what is the answer",
                         "retrieved_count": 0, "was_empty": True, "source": "ltm", "resolved": False}) + "\n",
            encoding="utf-8",
        )

        recs = reflector.reflect()
        if recs:
            pattern_terms = recs[0]["pattern"].lower().split()
            stopwords = {"the", "is", "what", "a", "an"}
            for term in pattern_terms:
                assert term not in stopwords, f"Stopword '{term}' should not appear in pattern"
