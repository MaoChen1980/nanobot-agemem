"""Unit tests for TimestampedFact dataclass."""

from datetime import datetime

import pytest

from nanobot.agent.agemem.fact import TimestampedFact


class TestTimestampedFact:
    def test_to_dict_roundtrip(self):
        """Serializing and deserializing preserves all fields."""
        fact = TimestampedFact(
            id="test-id",
            timestamp="2026-04-18T10:00:00",
            type="action",
            content={"tool": "write_file", "path": "x.txt"},
            importance=0.8,
            causes=["cause-1"],
            effects=["effect-1"],
            tags=["test"],
            confidence=0.9,
        )
        d = fact.to_dict()
        restored = TimestampedFact.from_dict(d)

        assert restored.id == fact.id
        assert restored.timestamp == fact.timestamp
        assert restored.type == fact.type
        assert restored.content == fact.content
        assert restored.importance == fact.importance
        assert restored.causes == fact.causes
        assert restored.effects == fact.effects
        assert restored.tags == fact.tags
        assert restored.confidence == fact.confidence

    def test_from_dict_with_defaults(self):
        """Missing fields get sensible defaults."""
        d = {
            "id": "minimal",
            "content": {"key": "value"},
        }
        fact = TimestampedFact.from_dict(d)

        assert fact.id == "minimal"
        assert fact.content == {"key": "value"}
        assert fact.type == "unknown"
        assert fact.importance == 0.5
        assert fact.causes == []
        assert fact.effects == []
        assert fact.tags == []
        assert fact.confidence == 1.0

    def test_add_cause(self):
        """add_cause links a cause ID without duplicates."""
        fact = TimestampedFact(
            id="effect",
            timestamp=datetime.now().isoformat(),
            type="event",
            content={"result": "done"},
        )
        assert fact.causes == []

        fact.add_cause("cause-1")
        assert fact.causes == ["cause-1"]

        fact.add_cause("cause-2")
        assert fact.causes == ["cause-1", "cause-2"]

        # No duplicates
        fact.add_cause("cause-1")
        assert fact.causes == ["cause-1", "cause-2"]

    def test_add_effect(self):
        """add_effect links an effect ID without duplicates."""
        fact = TimestampedFact(
            id="cause",
            timestamp=datetime.now().isoformat(),
            type="action",
            content={"tool": "read_file"},
        )
        assert fact.effects == []

        fact.add_effect("effect-1")
        assert fact.effects == ["effect-1"]

        fact.add_effect("effect-1")  # no-op
        assert fact.effects == ["effect-1"]

    def test_content_is_dict(self):
        """content field must be a dict."""
        fact = TimestampedFact(
            id="f1",
            timestamp=datetime.now().isoformat(),
            type="action",
            content={"action": "write", "path": "a.txt"},
        )
        assert isinstance(fact.content, dict)
        assert fact.content["action"] == "write"
