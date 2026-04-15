"""Persistence tests for AgeMem components.

Verifies that Reflector, MemoryPolicy, and GRPOCreditAssignment
correctly persist data to disk.
"""

import json
from pathlib import Path

import pytest

from nanobot.agent.agemem.reflector import Reflector
from nanobot.agent.agemem.policy import MemoryPolicy
from nanobot.agent.agemem.grpo import GRPOCreditAssignment
from nanobot.agent.agemem.rewards import RewardFunctions


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with memory directory."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


class TestReflectorPersistence:
    """Test Reflector gap event persistence."""

    def test_on_retrieve_records_empty_gap(self, workspace):
        reflector = Reflector(workspace)
        gaps_file = workspace / "memory" / "gaps.jsonl"

        reflector.on_retrieve(query="user name", retrieved_count=0, was_empty=True)

        assert gaps_file.exists()
        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["query"] == "user name"
        assert event["retrieved_count"] == 0
        assert event["was_empty"] is True
        assert event["source"] == "ltm"

    def test_on_retrieve_records_non_empty_event(self, workspace):
        reflector = Reflector(workspace)
        gaps_file = workspace / "memory" / "gaps.jsonl"

        reflector.on_retrieve(query="project deadline", retrieved_count=3, was_empty=False)

        assert gaps_file.exists()
        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["retrieved_count"] == 3
        assert event["was_empty"] is False

    def test_on_history_retrieval_records_gap(self, workspace):
        reflector = Reflector(workspace)
        gaps_file = workspace / "memory" / "gaps.jsonl"

        reflector.on_history_retrieval("api key", [{"content": "secret"}])

        assert gaps_file.exists()
        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        event = json.loads(lines[0])
        assert event["source"] == "history"
        assert event["was_empty"] is False

    def test_multiple_gaps_appended_to_same_file(self, workspace):
        reflector = Reflector(workspace)
        gaps_file = workspace / "memory" / "gaps.jsonl"

        reflector.on_retrieve("query1", 0, True)
        reflector.on_retrieve("query2", 0, True)

        lines = gaps_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_get_gaps_loads_resolved_flag(self, workspace):
        reflector = Reflector(workspace)
        gaps_file = workspace / "memory" / "gaps.jsonl"

        # Write a gap with resolved=True directly
        gap = {
            "timestamp": "2026-04-15T10:00:00",
            "query": "test query",
            "retrieved_count": 0,
            "was_empty": True,
            "source": "ltm",
            "resolved": True,
        }
        gaps_file.write_text(json.dumps(gap) + "\n", encoding="utf-8")

        # unresolved_only=True should skip it
        gaps = reflector.get_gaps(unresolved_only=True)
        assert len(gaps) == 0

        # unresolved_only=False should return it
        gaps = reflector.get_gaps(unresolved_only=False)
        assert len(gaps) == 1
        assert gaps[0].resolved is True


class TestMemoryPolicyPersistence:
    """Test MemoryPolicy auto-add rule persistence."""

    def test_record_explicit_creates_rule(self, workspace):
        policy = MemoryPolicy(workspace)
        rules_file = workspace / "memory" / "policy_rules.jsonl"

        rule = policy.record_explicit("Remember: I am allergic to penicillin", 0.95)

        assert rule.importance == 0.95
        assert "allergic" in rule.pattern or "penicillin" in rule.pattern
        assert rules_file.exists()

        lines = rules_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        saved = json.loads(lines[0])
        assert saved["importance"] == 0.95

    def test_record_self_assessed_creates_rule(self, workspace):
        policy = MemoryPolicy(workspace)
        rules_file = workspace / "memory" / "policy_rules.jsonl"

        rule = policy.record_self_assessed("User prefers dark mode", 0.7)

        assert rule is not None
        assert rule.importance == 0.7
        assert rules_file.exists()

    def test_policy_reloads_rules_on_init(self, workspace):
        # Create policy and add a rule
        policy1 = MemoryPolicy(workspace)
        policy1.record_explicit("Test content", 0.8)

        # Create new policy instance - should reload
        policy2 = MemoryPolicy(workspace)
        rules = policy2.get_auto_add_rules()

        assert len(rules) == 1
        assert rules[0].importance == 0.8

    def test_record_reflected_creates_rule_from_recommendation(self, workspace):
        policy = MemoryPolicy(workspace)
        rules_file = workspace / "memory" / "policy_rules.jsonl"

        recommendations = [{
            "action": "add_rule",
            "pattern": "deadline project",
            "importance": 0.75,
            "reason": "Gap detected 2 times",
        }]
        rules = policy.record_reflected(recommendations)

        assert len(rules) == 1
        assert rules[0].importance == 0.75
        assert rules_file.exists()

    def test_should_auto_add_returns_true_when_matched(self, workspace):
        policy = MemoryPolicy(workspace)
        policy.record_explicit("Remember: my birthday is July 4th", 0.9)

        should_add, importance = policy.should_auto_add("My birthday is July 4th this year")

        assert should_add is True
        assert importance == 0.9

    def test_should_auto_add_returns_false_when_no_match(self, workspace):
        policy = MemoryPolicy(workspace)
        policy.record_explicit("Remember: my birthday is July 4th", 0.9)

        should_add, importance = policy.should_auto_add("What is the weather today")

        assert should_add is False
        assert importance == 0.0


class TestGRPOPersistence:
    """Test GRPOCreditAssignment trajectory persistence."""

    def test_start_and_complete_trajectory(self, workspace):
        grpo = GRPOCreditAssignment(workspace)
        traj_dir = workspace / "memory" / "trajectories"

        traj_id = grpo.start_trajectory("Test task: remember my name")
        assert traj_id != ""

        # Record some steps
        grpo.record_step("add", memory_id="mem-123", content="My name is Bob")
        grpo.record_step("retrieve", memory_id=None, content="Found 0 memories", query="name")

        # Complete trajectory
        from nanobot.agent.agemem.rewards import RewardFunctions
        rewards = RewardFunctions()
        credits = grpo.complete_trajectory(0.85, rewards)

        assert len(credits) == 2
        # Verify trajectory file was created
        traj_files = list(traj_dir.glob("*.json"))
        assert len(traj_files) == 1
        saved = json.loads(traj_files[0].read_text(encoding="utf-8"))
        assert saved["final_reward"] == 0.85
        assert len(saved["steps"]) == 2

    def test_complete_empty_trajectory(self, workspace):
        grpo = GRPOCreditAssignment(workspace)

        grpo.start_trajectory("Empty task")
        from nanobot.agent.agemem.rewards import RewardFunctions
        credits = grpo.complete_trajectory(0.5, RewardFunctions())

        assert credits == []

    def test_get_recent_trajectories(self, workspace):
        grpo = GRPOCreditAssignment(workspace)

        for i in range(3):
            grpo.start_trajectory(f"Task {i}")
            grpo.record_step("add", memory_id=f"mem-{i}", content=f"Content {i}")
            grpo.complete_trajectory(0.5 + i * 0.1, RewardFunctions())

        recent = grpo.get_recent_trajectories(limit=2)
        assert len(recent) == 2

    def test_cancel_trajectory_no_file(self, workspace):
        grpo = GRPOCreditAssignment(workspace)
        traj_dir = workspace / "memory" / "trajectories"

        grpo.start_trajectory("Cancelled task")
        grpo.record_step("add", memory_id="mem-1", content="Content")
        grpo.cancel_trajectory()

        # Should not create any trajectory file
        if traj_dir.exists():
            assert len(list(traj_dir.glob("*.json"))) == 0

    def test_gamma_discount_applied_in_credits(self, workspace):
        grpo = GRPOCreditAssignment(workspace, gamma=0.9)

        traj_id = grpo.start_trajectory("Discount test")
        grpo.record_step("add", memory_id="mem-1", content="First")
        grpo.record_step("add", memory_id="mem-2", content="Second")

        credits = grpo.complete_trajectory(1.0, RewardFunctions())

        # Last step (i=0) gets full reward: 1.0
        assert credits[0]["advantage"] == 1.0
        # First step (i=1) gets gamma^1 * 1.0 = 0.9
        assert credits[1]["advantage"] == 0.9
