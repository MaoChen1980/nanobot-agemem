"""GRPO credit assignment for AgeMem memory policy optimization.

Implements step-wise reward broadcast from the AgeMem paper:
- Track memory operation trajectories
- Compute advantages via reward broadcast
- Update MemoryPolicy based on observed rewards

This is a simplified GRPO focusing on credit assignment, not full policy gradient.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class MemoryStep:
    """A single step in a memory operation trajectory."""
    step_id: str
    timestamp: str
    operation: str  # "retrieve", "add", "update", "delete", "summarize"
    memory_id: str | None  # None for summarize/retrieve without results
    content: str | None
    importance: float | None
    query: str | None  # For retrieve operations
    reward: float | None = None  # Assigned after task completion
    advantage: float | None = None


@dataclass
class Trajectory:
    """A complete trajectory from task start to completion."""
    trajectory_id: str
    task_description: str
    created_at: str
    completed_at: str | None = None
    final_reward: float | None = None
    steps: list[MemoryStep] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(s) if isinstance(s, MemoryStep) else s for s in self.steps]
        return d


class GRPOCreditAssignment:
    """Step-wise GRPO credit assignment for memory operations.

    The key insight from AgeMem paper:
    "Terminal rewards are broadcast to all preceding steps"

    This means: when a task completes (Rtask), we propagate that reward
    backward through all memory operations that contributed.
    """

    def __init__(self, workspace: Path, gamma: float = 0.95):
        """
        Args:
            workspace: Working directory for trajectory storage
            gamma: Discount factor for reward propagation
        """
        self.workspace = workspace
        self.gamma = gamma
        self.trajectories_dir = workspace / "memory" / "trajectories"
        self._ensure_dir()

        self._current_trajectory: Trajectory | None = None
        self._step_counter = 0

    def _ensure_dir(self) -> None:
        if not self.trajectories_dir.exists():
            self.trajectories_dir.mkdir(parents=True, exist_ok=True)

    def start_trajectory(self, task_description: str) -> str:
        """Begin a new trajectory for a task.

        Returns the trajectory_id.
        """
        import uuid
        self._step_counter = 0
        self._current_trajectory = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            task_description=task_description,
            created_at=datetime.now().isoformat(),
        )
        logger.debug("GRPO: started trajectory {}", self._current_trajectory.trajectory_id)
        return self._current_trajectory.trajectory_id

    def record_step(
        self,
        operation: str,
        memory_id: str | None = None,
        content: str | None = None,
        importance: float | None = None,
        query: str | None = None,
    ) -> str:
        """Record a memory operation step in the current trajectory.

        Returns the step_id.
        """
        if self._current_trajectory is None:
            return ""

        import uuid
        self._step_counter += 1
        step_id = f"{self._current_trajectory.trajectory_id}_{self._step_counter}"

        step = MemoryStep(
            step_id=step_id,
            timestamp=datetime.now().isoformat(),
            operation=operation,
            memory_id=memory_id,
            content=content,
            importance=importance,
            query=query,
        )
        self._current_trajectory.steps.append(step)
        logger.debug("GRPO: recorded step {} operation={}", step_id, operation)
        return step_id

    def complete_trajectory(
        self,
        final_reward: float,
        rewards: Any,  # RewardFunctions instance for detailed breakdown
    ) -> list[dict[str, Any]]:
        """Complete the current trajectory with final reward.

        Broadcasts the reward to all preceding steps using step-wise GRPO.
        Returns list of credit assignments for each step.
        """
        if self._current_trajectory is None:
            return []

        traj = self._current_trajectory
        traj.completed_at = datetime.now().isoformat()
        traj.final_reward = final_reward

        # Step-wise reward broadcast: propagate reward backward
        # The last step gets full credit, earlier steps get discounted credit
        credits: list[dict[str, Any]] = []
        n_steps = len(traj.steps)

        if n_steps == 0:
            self._save_trajectory(traj)
            self._current_trajectory = None
            return []

        # Broadcast rewards from end to start
        accumulated = 0.0
        for i, step in enumerate(reversed(traj.steps)):
            idx = n_steps - 1 - i
            # Broadcast: each step gets its immediate reward plus discounted future rewards
            if i == 0:
                # Terminal step: full reward
                advantage = final_reward
            else:
                # Earlier steps: gamma^i * final_reward
                advantage = (self.gamma ** i) * final_reward

            step.reward = advantage
            step.advantage = advantage
            accumulated += advantage

            credits.append({
                "step_id": step.step_id,
                "operation": step.operation,
                "memory_id": step.memory_id,
                "advantage": round(advantage, 4),
                "position": idx + 1,
                "total_steps": n_steps,
            })

        logger.info(
            "GRPO: completed trajectory {} with {} steps, final_reward={:.3f}",
            traj.trajectory_id, n_steps, final_reward,
        )

        self._save_trajectory(traj)
        self._current_trajectory = None

        return credits

    def cancel_trajectory(self) -> None:
        """Cancel the current trajectory without reward."""
        if self._current_trajectory:
            logger.debug("GRPO: cancelled trajectory {}", self._current_trajectory.trajectory_id)
        self._current_trajectory = None

    def _save_trajectory(self, traj: Trajectory) -> None:
        """Persist trajectory to disk."""
        filepath = self.trajectories_dir / f"{traj.trajectory_id}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(traj.to_dict(), f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("GRPO: failed to save trajectory: {}", e)

    def get_recent_trajectories(self, limit: int = 10) -> list[Trajectory]:
        """Load recent completed trajectories."""
        if not self.trajectories_dir.exists():
            return []

        trajectories: list[Trajectory] = []
        try:
            files = sorted(
                self.trajectories_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:limit]

            for filepath in files:
                with open(filepath, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    # Reconstruct MemoryStep objects
                    if "steps" in d:
                        d["steps"] = [MemoryStep(**s) for s in d["steps"]]
                    trajectories.append(Trajectory(**d))
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("GRPO: failed to load trajectories: {}", e)

        return trajectories

    def compute_policy_updates(
        self,
        credits: list[dict[str, Any]],
        policy: Any,  # MemoryPolicy
    ) -> list[Any]:
        """Apply GRPO credits to update the MemoryPolicy.

        Uses the advantage values to adjust memory entry importances
        and create/update auto-add rules.
        """
        if not credits:
            return []

        updates: list[Any] = []
        memory_ids_seen: set[str] = set()

        # Sort by advantage descending to prioritize high-impact operations
        sorted_credits = sorted(credits, key=lambda c: c["advantage"], reverse=True)

        for credit in sorted_credits:
            op = credit["operation"]
            mem_id = credit.get("memory_id")
            advantage = credit["advantage"]
            content = credit.get("content", "")

            # Skip operations without memory IDs
            if not mem_id and op not in ("summarize", "retrieve"):
                continue

            if advantage <= 0:
                continue  # No positive credit, no update needed

            # Compute importance delta based on advantage
            # Higher advantage -> higher importance boost
            importance_delta = min(advantage * 0.2, 0.3)  # Cap at 0.3 per step

            if op == "add" and mem_id and advantage > 0.2:
                # Positive credit for successful add
                # Boost related rules
                if content:
                    rule = policy.record_self_assessed(
                        content,
                        importance=0.5 + advantage * 0.3,
                    )
                    if rule:
                        updates.append(f"boosted rule: {rule.pattern} (+{importance_delta:.2f})")
                    memory_ids_seen.add(mem_id)

            elif op == "retrieve" and advantage > 0.3:
                # Successful retrieval (led to task completion)
                # Increase importance of retrieved memories
                if mem_id and mem_id not in memory_ids_seen:
                    policy.record_access(mem_id, content or "")
                    updates.append(f"accessed: {mem_id} (advantage={advantage:.2f})")
                    memory_ids_seen.add(mem_id)

            elif op == "update" and mem_id and advantage > 0.1:
                # Update was beneficial
                if mem_id not in memory_ids_seen:
                    policy.record_access(mem_id, content or "")
                    memory_ids_seen.add(mem_id)

            elif op == "auto_add" and mem_id and advantage > 0.2:
                # Policy-initiated auto-add was beneficial
                if content:
                    rule = policy.record_self_assessed(
                        content,
                        importance=0.5 + advantage * 0.3,
                    )
                    if rule:
                        updates.append(f"auto_add rule: {rule.pattern} (+{importance_delta:.2f})")
                    memory_ids_seen.add(mem_id)

        return updates
