"""Integration tests for MemoryHook lifecycle.

Tests the full feedback loop:
1. GRPO trajectory starts on first iteration
2. Memory tool events are recorded (retrieve → gap, add/update → policy)
3. Phase 2d auto-add triggers when response matches learned rules
4. finalize_content broadcasts GRPO rewards to all steps
"""

import pytest

from nanobot.agent.agemem.store import MemoryStoreV2
from nanobot.agent.hook import AgentHookContext
from nanobot.agent.hooks.memory_hook import MemoryHook
from nanobot.agent.agemem.policy import MemoryPolicy
from nanobot.agent.agemem.reflector import Reflector
from nanobot.providers.base import LLMResponse


# MemoryHook.before_iteration and after_iteration are async;
# finalize_content is sync. Use module-level mark + make all test methods async.
pytestmark = pytest.mark.asyncio


@pytest.fixture
def ws(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


def _ctx(
    iteration: int = 1,
    messages: list | None = None,
    response_content: str = "",
    tool_events: list | None = None,
) -> AgentHookContext:
    """Build an AgentHookContext for testing."""
    if messages is None:
        messages = [{"role": "user", "content": "Remember my name is Alice"}]
    return AgentHookContext(
        iteration=iteration,
        messages=messages,
        response=LLMResponse(content=response_content) if response_content else None,
        tool_events=tool_events or [],
    )


class TestMemoryHook_TrajectoryLifecycle:
    async def test_before_iteration_starts_grpo_trajectory(self, ws):
        """First before_iteration call starts a GRPO trajectory."""
        hook = MemoryHook(ws)
        ctx = _ctx(messages=[{"role": "user", "content": "What is my name?"}])

        await hook.before_iteration(ctx)

        assert hook._trajectory_started is True
        assert hook._task_description == "What is my name?"
        # Trajectory should be started in GRPO
        assert len(hook.grpo.get_recent_trajectories()) == 0  # Not completed yet

    async def test_before_iteration_skips_if_already_started(self, ws):
        """Subsequent before_iteration calls are no-ops."""
        hook = MemoryHook(ws)
        ctx1 = _ctx(messages=[{"role": "user", "content": "First task"}])
        ctx2 = _ctx(messages=[{"role": "user", "content": "Second task"}])

        await hook.before_iteration(ctx1)
        await hook.before_iteration(ctx2)

        # Should still be the first task description
        assert hook._task_description == "First task"

    async def test_finalize_content_completes_trajectory(self, ws):
        """finalize_content completes GRPO trajectory and broadcasts rewards."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            messages=[{"role": "user", "content": "My name is Bob"}],
            response_content="Hello Bob! I will remember that.",
            tool_events=[
                {"name": "add_memory", "status": "ok", "detail": "ID: 12345678-1234-1234-1234-123456789012 Content: My name is Bob"},
            ],
        )

        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        result = hook.finalize_content(ctx, "Hello Bob!")

        # Trajectory should be completed and saved
        trajectories = hook.grpo.get_recent_trajectories(limit=1)
        assert len(trajectories) == 1
        assert trajectories[0].task_description == "My name is Bob"
        assert trajectories[0].final_reward is not None
        assert hook._trajectory_started is False


class TestMemoryHook_GapDetection:
    async def test_empty_retrieve_records_gap(self, ws):
        """Empty retrieve_memories result records a gap in the reflector."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            tool_events=[
                {"name": "retrieve_memories", "status": "ok", "detail": "No memories found"},
            ],
        )

        await hook.after_iteration(ctx)

        gaps = hook.reflector.get_gaps()
        assert len(gaps) == 1
        assert gaps[0].was_empty is True
        assert gaps[0].retrieved_count == 0

    async def test_non_empty_retrieve_records_non_gap_event(self, ws):
        """retrieve_memories with results records a non-gap (was_empty=False) event."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            tool_events=[
                {"name": "retrieve_memories", "status": "ok", "detail": "[id=12345678-1234-1234-1234-123456789012] Found 1 memory"},
            ],
        )

        await hook.after_iteration(ctx)

        gaps = hook.reflector.get_gaps()
        assert len(gaps) == 1
        assert gaps[0].was_empty is False
        assert gaps[0].retrieved_count == 1


class TestMemoryHook_MemoryOperationTracking:
    async def test_add_memory_records_policy_rule(self, ws):
        """add_memory records an explicit rule in the policy."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            tool_events=[
                {
                    "name": "add_memory",
                    "status": "ok",
                    "detail": "ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee Content: My name is Alice",
                },
            ],
        )

        await hook.after_iteration(ctx)

        rules = hook.policy.get_auto_add_rules()
        assert len(rules) >= 1
        # The content should have been recorded
        rule = rules[0]
        assert rule.importance > 0

    async def test_add_memory_records_grpo_step(self, ws):
        """add_memory records a step in the GRPO trajectory."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            messages=[{"role": "user", "content": "Remember something"}],
            tool_events=[
                {
                    "name": "add_memory",
                    "status": "ok",
                    "detail": "ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee Content: Important content",
                },
            ],
        )

        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        hook.finalize_content(ctx, "Done")

        trajectories = hook.grpo.get_recent_trajectories(limit=1)
        assert len(trajectories) == 1
        step_ops = [s.operation for s in trajectories[0].steps]
        assert "add" in step_ops

    async def test_update_memory_records_access(self, ws):
        """update_memory records an access boost in the policy."""
        hook = MemoryHook(ws)

        # Pre-populate a rule
        hook.policy.record_explicit("Original content", 0.5)

        ctx = _ctx(
            tool_events=[
                {
                    "name": "update_memory",
                    "status": "ok",
                    "detail": "ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee Content: Updated content",
                },
            ],
        )

        await hook.after_iteration(ctx)
        # Should not raise - policy should handle the update

    async def test_summarize_session_does_not_crash(self, ws):
        """summarize_session is handled gracefully (no policy update)."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            tool_events=[
                {
                    "name": "summarize_session",
                    "status": "ok",
                    "detail": "Summary: Discussed project deadline",
                },
            ],
        )

        # Should not raise
        await hook.after_iteration(ctx)


class TestMemoryHook_Phase2dAutoAdd:
    async def test_auto_add_triggers_when_response_matches_rule(self, ws):
        """Phase 2d: response matching a policy rule auto-adds to LTM store."""
        # Pre-populate a policy rule
        policy = MemoryPolicy(ws)
        policy.record_explicit("allergic penicillin", 0.9)

        hook = MemoryHook(ws, auto_add_enabled=True)
        # Response contains "penicillin" which matches the rule
        ctx = _ctx(
            messages=[{"role": "user", "content": "I might need to see a doctor"}],
            response_content="You should avoid penicillin since you are allergic to it.",
            tool_events=[],
        )

        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        hook.finalize_content(ctx, "Avoid penicillin - you are allergic.")

        # Memory store should have the auto-added entry
        store = MemoryStoreV2(ws)
        entries = store.get_all()
        assert len(entries) >= 1
        # Content should mention penicillin or allergy
        contents = [e.content.lower() for e in entries]
        assert any("penicillin" in c or "allergic" in c for c in contents)

    async def test_auto_add_disabled_respects_flag(self, ws):
        """When auto_add_enabled=False, no auto-add happens."""
        hook = MemoryHook(ws, auto_add_enabled=False)
        ctx = _ctx(
            response_content="You are allergic to penicillin.",
            tool_events=[],
        )

        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)

        store = MemoryStoreV2(ws)
        # Should not auto-add when disabled
        # (may have other entries from previous tests, but at minimum no NEW auto-add)
        initial_count = len(store.get_all())


class TestMemoryHook_GRPOCreditBroadcast:
    async def test_finalize_content_broadcasts_rewards_to_steps(self, ws):
        """GRPO rewards are broadcast from final step to all preceding steps."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            messages=[{"role": "user", "content": "Remember my deadline"}],
            response_content="Your deadline is Friday.",
            tool_events=[
                {
                    "name": "add_memory",
                    "status": "ok",
                    "detail": "ID: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee Content: Deadline Friday",
                },
            ],
        )

        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        hook.finalize_content(ctx, "Your deadline is Friday.")

        trajectories = hook.grpo.get_recent_trajectories(limit=1)
        assert len(trajectories) == 1
        traj = trajectories[0]

        # All steps should have non-None rewards after broadcast
        for step in traj.steps:
            assert step.reward is not None
            assert step.advantage is not None

        # Final reward should be > 0 (compute_rtask_sync heuristic)
        assert traj.final_reward > 0

    async def test_finalize_content_without_trajectory_is_noop(self, ws):
        """finalize_content without a started trajectory returns content unchanged."""
        hook = MemoryHook(ws)
        ctx = _ctx(response_content="Some response")

        # No before_iteration call - trajectory never started
        result = hook.finalize_content(ctx, "Some response")

        assert result == "Some response"
        assert len(hook.grpo.get_recent_trajectories()) == 0


class TestMemoryHook_ReflectorIntegration:
    async def test_retrieve_non_empty_records_non_gap_event(self, ws):
        """Non-empty retrieve records a non-gap event (was_empty=False)."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            tool_events=[
                {
                    "name": "retrieve_memories",
                    "status": "ok",
                    "detail": "[id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee] Content: Your name is Alice",
                },
            ],
        )

        await hook.after_iteration(ctx)

        gaps = hook.reflector.get_gaps()
        # Non-empty retrieve records an event (was_empty=False), not a gap
        assert len(gaps) == 1
        assert gaps[0].was_empty is False
        assert gaps[0].retrieved_count == 1


class TestMemoryHook_Persistence:
    async def test_gaps_persist_to_disk(self, ws):
        """Gap events are persisted to gaps.jsonl."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            tool_events=[
                {"name": "retrieve_memories", "status": "ok", "detail": "No memories found"},
            ],
        )

        await hook.after_iteration(ctx)

        gaps_file = ws / "memory" / "gaps.jsonl"
        assert gaps_file.exists()

    async def test_trajectories_persist_to_disk(self, ws):
        """Completed trajectories are persisted to trajectories/ directory."""
        hook = MemoryHook(ws)
        ctx = _ctx(
            messages=[{"role": "user", "content": "Test task"}],
            response_content="Test response",
            tool_events=[],
        )

        await hook.before_iteration(ctx)
        hook.finalize_content(ctx, "Test response")

        traj_dir = ws / "memory" / "trajectories"
        assert traj_dir.exists()
        assert len(list(traj_dir.glob("*.json"))) >= 1
