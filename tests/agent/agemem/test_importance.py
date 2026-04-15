"""Unit tests for ImportanceScorer."""

import pytest

from nanobot.agent.agemem.importance import ImportanceScorer


class TestExtractImportanceSignals:
    """Test keyword-based importance signal extraction."""

    def test_critical_signals_return_095(self):
        scorer = ImportanceScorer()
        cases = [
            "I am allergic to penicillin",
            "Critical: health condition",
            "Medical emergency: blood type O negative",
            "Password: my secret key is abc123",
        ]
        for content in cases:
            result = scorer.extract_importance_signals(content)
            assert result == 0.95, f"Expected 0.95 for '{content}', got {result}"

    def test_high_signals_return_070(self):
        scorer = ImportanceScorer()
        cases = [
            "This is IMPORTANT: check the report",
            "Please remember this deadline",
            "Never forget to call back",
            "My preference is dark mode",
            "I always do morning exercise",
        ]
        for content in cases:
            result = scorer.extract_importance_signals(content)
            assert result == 0.7, f"Expected 0.7 for '{content}', got {result}"

    def test_medium_signals_return_050(self):
        scorer = ImportanceScorer()
        cases = [
            "Noted: user likes coffee",
            "Keep in mind the budget",
            "I like tea over coffee",
        ]
        for content in cases:
            result = scorer.extract_importance_signals(content)
            assert result == 0.5, f"Expected 0.5 for '{content}', got {result}"

    def test_low_signals_return_020(self):
        scorer = ImportanceScorer()
        cases = [
            "btw I was late today",
            "maybe we should check tomorrow",
            "fyi the meeting is at 3pm",
        ]
        for content in cases:
            result = scorer.extract_importance_signals(content)
            assert result == 0.2, f"Expected 0.2 for '{content}', got {result}"

    def test_no_signal_returns_none(self):
        scorer = ImportanceScorer()
        cases = [
            "Just a regular sentence with nothing special",
            "The weather is nice today",
            "Hello, how are you?",
        ]
        for content in cases:
            result = scorer.extract_importance_signals(content)
            assert result is None, f"Expected None for '{content}', got {result}"

    def test_mixed_signals_returns_highest(self):
        scorer = ImportanceScorer()
        # "allergic" should override "btw"
        result = scorer.extract_importance_signals("btw I am allergic to penicillin")
        assert result == 0.95


class TestScoreSync:
    """Test synchronous heuristic scoring."""

    def test_base_score_is_05(self):
        scorer = ImportanceScorer()
        result = scorer.score_sync("Any content here")
        assert result == 0.5

    def test_high_access_count_boosts(self):
        scorer = ImportanceScorer()
        # access_count > 10 adds 0.2
        result = scorer.score_sync("Frequent content", access_count=15)
        assert result == 0.7

    def test_medium_access_count_small_boost(self):
        scorer = ImportanceScorer()
        # access_count > 5 adds 0.1
        result = scorer.score_sync("Moderate content", access_count=7)
        assert result == 0.6

    def test_low_access_count_minimal_boost(self):
        scorer = ImportanceScorer()
        # access_count > 2 adds 0.05
        result = scorer.score_sync("Slight access", access_count=3)
        assert result == 0.55

    def test_score_clamped_to_one(self):
        scorer = ImportanceScorer()
        # Max boost: 0.5 base + 0.2 access + 0.1 recency = 0.8
        result = scorer.score_sync("Frequently accessed recent item", access_count=20)
        assert result <= 1.0

    def test_score_never_negative(self):
        scorer = ImportanceScorer()
        # Even with old content and high access, should not go below 0
        result = scorer.score_sync("Old content", access_count=100, created_at="2020-01-01T00:00:00")
        assert result >= 0.0
