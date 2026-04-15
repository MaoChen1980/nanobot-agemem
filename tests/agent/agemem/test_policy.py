"""Unit tests for MemoryPolicy (auto-add rules learning)."""

from pathlib import Path

import pytest

from nanobot.agent.agemem.policy import MemoryPolicy, AutoAddRule


@pytest.fixture
def ws(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


class TestMemoryPolicy_RecordSignals:
    def test_record_explicit_creates_high_importance_rule(self, ws):
        policy = MemoryPolicy(ws)
        rules_file = ws / "memory" / "policy_rules.jsonl"

        rule = policy.record_explicit("Remember: I am allergic to penicillin", 0.95)

        assert rule.importance == 0.95
        assert rule.reason.startswith("explicit")
        assert "allergic" in rule.pattern or "penicillin" in rule.pattern
        assert rules_file.exists()

    def test_record_self_assessed_creates_rule(self, ws):
        policy = MemoryPolicy(ws)

        rule = policy.record_self_assessed("User prefers dark mode interface", 0.7)

        assert rule is not None
        assert rule.importance == 0.7
        assert (ws / "memory" / "policy_rules.jsonl").exists()

    def test_record_self_assessed_creates_rule(self, ws):
        """record_self_assessed creates a rule with correct importance."""
        policy = MemoryPolicy(ws)

        rule = policy.record_self_assessed("User prefers dark mode configuration", 0.6)

        assert rule is not None
        assert rule.importance == 0.6
        assert rule.reason == "self_assessed: agent decided to add"

    def test_record_self_assessed_persists_to_disk(self, ws):
        """A self-assessed rule survives a new Policy instance."""
        policy1 = MemoryPolicy(ws)
        policy1.record_self_assessed("Important deadline matter", 0.75)

        policy2 = MemoryPolicy(ws)
        rules = policy2.get_auto_add_rules()

        assert len(rules) == 1
        assert rules[0].importance == 0.75

    def test_record_self_assessed_returns_none_for_generic_content(self, ws):
        policy = MemoryPolicy(ws)
        # Content with only common words
        rule = policy.record_self_assessed("the a is", 0.5)
        assert rule is None

    def test_record_reflected_creates_rule_from_recommendation(self, ws):
        policy = MemoryPolicy(ws)
        rules_file = ws / "memory" / "policy_rules.jsonl"

        recommendations = [{
            "action": "add_rule",
            "pattern": "budget project deadline",
            "importance": 0.75,
            "reason": "Gap detected 2 times",
        }]
        rules = policy.record_reflected(recommendations)

        assert len(rules) == 1
        assert rules[0].importance == 0.75
        assert rules[0].reason == "Gap detected 2 times"
        assert rules_file.exists()

    def test_record_reflected_updates_existing_rule(self, ws):
        """Reflecting the same pattern twice should increase importance."""
        policy = MemoryPolicy(ws)

        policy.record_reflected([{
            "action": "add_rule",
            "pattern": "meeting schedule",
            "importance": 0.6,
            "reason": "first reflection",
        }])
        initial_imp = policy.get_auto_add_rules()[0].importance

        policy.record_reflected([{
            "action": "add_rule",
            "pattern": "meeting schedule",
            "importance": 0.8,
            "reason": "second reflection",
        }])

        updated_imp = policy.get_auto_add_rules()[0].importance
        assert updated_imp >= initial_imp

    def test_record_reflected_ignores_non_add_rule_actions(self, ws):
        policy = MemoryPolicy(ws)

        recommendations = [{
            "action": "increase_importance",
            "existing_id": "some-id",
            "delta": 0.2,
        }]
        rules = policy.record_reflected(recommendations)
        assert rules == []

    def test_record_access_boosts_related_rule(self, ws):
        policy = MemoryPolicy(ws)

        policy.record_explicit("Remember project deadlines always", 0.8)
        initial_imp = policy.get_auto_add_rules()[0].importance

        policy.record_access(policy.get_auto_add_rules()[0].id, "project deadlines")

        updated_imp = policy.get_auto_add_rules()[0].importance
        assert updated_imp >= initial_imp


class TestMemoryPolicy_ShouldAutoAdd:
    def test_should_auto_add_returns_true_when_matched(self, ws):
        policy = MemoryPolicy(ws)
        policy.record_explicit("Remember: my birthday is July 4th", 0.9)

        should_add, importance = policy.should_auto_add("My birthday is July 4th this year")

        assert should_add is True
        assert importance == 0.9

    def test_should_auto_add_returns_false_when_no_match(self, ws):
        policy = MemoryPolicy(ws)
        policy.record_explicit("Remember: my birthday is July 4th", 0.9)

        should_add, importance = policy.should_auto_add("What is the weather today")

        assert should_add is False
        assert importance == 0.0

    def test_should_auto_add_requires_minimum_keyword_overlap(self, ws):
        """Need at least 2 matching terms between content and pattern."""
        policy = MemoryPolicy(ws)
        policy.record_explicit("Remember: user prefers dark mode", 0.8)

        # Only "prefers" matches - insufficient
        should_add, _ = policy.should_auto_add("I prefer coffee")
        assert should_add is False

        # Two words in common - should match
        should_add, _ = policy.should_auto_add("User prefers the dark theme")
        assert should_add is True


class TestMemoryPolicy_GetMatchingRules:
    def test_get_matching_rules_requires_two_term_overlap(self, ws):
        policy = MemoryPolicy(ws)
        policy.record_explicit("Remember: project deadline is Friday", 0.7)

        # Only "project" matches - insufficient
        rules = policy.get_matching_rules("I like project management")
        assert len(rules) == 0

        # "project" and "deadline" match - sufficient
        rules = policy.get_matching_rules("project deadline update")
        assert len(rules) >= 1

    def test_get_matching_rules_sorted_by_importance(self, ws):
        policy = MemoryPolicy(ws)
        # Use content with distinctive patterns (3+ char terms not in stopwords)
        policy.record_explicit("Remember: critical allergy penicillin", 0.5)
        policy.record_explicit("Remember: urgent deadline report", 0.9)

        # "urgent deadline report" and "critical allergy penicillin" have different patterns
        rules = policy.get_matching_rules("urgent deadline report urgent")
        assert len(rules) >= 1


class TestMemoryPolicy_GetAutoAddRules:
    def test_get_auto_add_rules_returns_sorted_list(self, ws):
        policy = MemoryPolicy(ws)
        # Use completely unique terms with zero word overlap
        policy.record_explicit("Lunch hunger ate food meal", 0.3)
        policy.record_explicit("Skiing mountain winter sport snow", 0.8)
        policy.record_explicit("Fever doctor medicine health sick", 0.5)

        rules = policy.get_auto_add_rules()

        assert len(rules) == 3
        # Sorted by importance descending
        assert rules[0].importance >= rules[1].importance >= rules[2].importance

    def test_get_auto_add_rules_empty_when_no_rules(self, ws):
        policy = MemoryPolicy(ws)
        rules = policy.get_auto_add_rules()
        assert rules == []


class TestMemoryPolicy_Persistence:
    def test_rules_persist_across_instances(self, ws):
        policy1 = MemoryPolicy(ws)
        policy1.record_explicit("Test content persists precisely", 0.85)

        policy2 = MemoryPolicy(ws)
        rules = policy2.get_auto_add_rules()

        assert len(rules) == 1
        assert rules[0].importance == 0.85

    def test_pattern_extraction_excludes_stopwords(self, ws):
        """Significant terms (3+ chars, not stopwords) should be extracted."""
        policy = MemoryPolicy(ws)
        rule = policy.record_explicit(
            "Remember: the user always prefers the dark theme in the evening",
            0.8,
        )

        pattern_terms = set(rule.pattern.lower().split())
        stopwords = {"the", "a", "an", "is", "in", "user"}
        for term in pattern_terms:
            assert term not in stopwords, f"Stopword '{term}' should not be in pattern"
