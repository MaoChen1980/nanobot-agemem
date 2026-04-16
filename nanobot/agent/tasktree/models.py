"""Data models for TaskTree."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Status of a TaskNode."""

    PENDING = "pending"      # Not yet started
    RUNNING = "running"     # Currently executing
    DONE = "done"           # Completed successfully
    FAILED = "failed"       # Failed (after replan attempts exhausted)
    BLOCKED = "blocked"     # Blocked by constraint veto


class RootCause(str, Enum):
    """Enumerated root cause for node failures."""

    API_TIMEOUT = "api_timeout"
    FILE_NOT_FOUND = "file_not_found"
    CONSTRAINT_VETO = "constraint_veto"
    NO_REMAINING_OPTIONS = "no_remaining_options"
    MAX_REPLAN_REACHED = "max_replan_reached"
    UNKNOWN = "unknown"


class WorkspaceState(str, Enum):
    """Impact of node execution on workspace."""

    CLEAN = "clean"    # No files modified
    PARTIAL = "partial"  # Some modifications, can be ignored
    DIRTY = "dirty"    # Modifications may affect other nodes


@dataclass
class TaskNode:
    """A single node in the task tree.

    Nodes are created by a parent node's execution and form a tree structure.
    The tree is never mutated by deleting nodes — only by adding children.
    """

    id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    result: NodeResult | None = None
    failure: FailureReport | None = None
    replan_count: int = 0
    verification_failure: str | None = None  # Populated when this node failed verification
    workspace_state: WorkspaceState = WorkspaceState.CLEAN  # Tracks modifications to workspace
    created_at: float = field(default_factory=time.time)

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """Return True if this node is in a terminal state."""
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED)


@dataclass
class Artifact:
    """A structured output produced by a node."""

    type: str                           # e.g. "file_written", "file_modified", "api_call"
    description: str                   # Human-readable description
    path: str | None = None            # Relevant file or resource path

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "path": self.path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
        return cls(
            type=d["type"],
            path=d.get("path"),
            description=d["description"],
        )


@dataclass
class NodeResult:
    """Result returned when a TaskNode completes successfully."""

    node_id: str
    status: TaskStatus = TaskStatus.DONE
    summary: str = ""
    artifacts: list[Artifact] = field(default_factory=list)
    constraints_respected: bool = True
    token_spent: int = 0
    workspace_state: WorkspaceState = WorkspaceState.CLEAN
    user_input_question: str | None = None  # If set, node needs user input before proceeding
    user_input_answer: str | None = None   # User's response to the question

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "summary": self.summary,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "constraints_respected": self.constraints_respected,
            "token_spent": self.token_spent,
            "workspace_state": self.workspace_state.value,
            "user_input_question": self.user_input_question,
            "user_input_answer": self.user_input_answer,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeResult:
        return cls(
            node_id=d["node_id"],
            status=TaskStatus(d.get("status", "done")),
            summary=d.get("summary", ""),
            artifacts=[Artifact.from_dict(a) for a in d.get("artifacts", [])],
            constraints_respected=d.get("constraints_respected", True),
            token_spent=d.get("token_spent", 0),
            workspace_state=WorkspaceState(d.get("workspace_state", "clean")),
            user_input_question=d.get("user_input_question"),
            user_input_answer=d.get("user_input_answer"),
        )


@dataclass
class FailureReport:
    """Structured failure report when a TaskNode fails.

    Used by the parent node to decide whether to replan or propagate failure upward.
    """

    node_id: str
    status: TaskStatus = TaskStatus.FAILED
    root_cause: RootCause = RootCause.UNKNOWN
    summary: str = ""
    tried: list[str] = field(default_factory=list)
    remaining_options: list[str] = field(default_factory=list)
    impact_on_parent: str = ""
    constraint_veto: bool = False
    workspace_state: WorkspaceState = WorkspaceState.CLEAN
    token_spent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "root_cause": self.root_cause.value,
            "summary": self.summary,
            "tried": self.tried,
            "remaining_options": self.remaining_options,
            "impact_on_parent": self.impact_on_parent,
            "constraint_veto": self.constraint_veto,
            "workspace_state": self.workspace_state.value,
            "token_spent": self.token_spent,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FailureReport:
        return cls(
            node_id=d["node_id"],
            status=TaskStatus(d.get("status", "failed")),
            root_cause=RootCause(d.get("root_cause", "unknown")),
            summary=d.get("summary", ""),
            tried=d.get("tried", []),
            remaining_options=d.get("remaining_options", []),
            impact_on_parent=d.get("impact_on_parent", ""),
            constraint_veto=d.get("constraint_veto", False),
            workspace_state=WorkspaceState(d.get("workspace_state", "clean")),
            token_spent=d.get("token_spent", 0),
        )


@dataclass
class ConstraintSet:
    """Boundary constraints for a node's execution.

    Produced by ConstraintAgent before a node executes.
    All constraints are hard constraints — violation triggers a constraint_veto.
    """

    max_depth: int = 10                     # Maximum tree depth from root
    forbidden_actions: list[str] = field(default_factory=list)  # e.g. ["delete_file", "rm_rf"]
    failure_count_limit: int = 2            # Same root cause retry limit before veto

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "forbidden_actions": self.forbidden_actions,
            "failure_count_limit": self.failure_count_limit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConstraintSet:
        return cls(
            max_depth=d.get("max_depth", 10),
            forbidden_actions=d.get("forbidden_actions", []),
            failure_count_limit=d.get("failure_count_limit", 2),
        )
