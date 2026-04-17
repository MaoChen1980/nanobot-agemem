"""SubgoalParser: extracts child goals from a successful NodeResult."""

from __future__ import annotations

import json
import re
from typing import Protocol

from nanobot.agent.tasktree.models import NodeResult


class SubgoalParser(Protocol):
    """Parse a NodeResult to extract a list of child goal strings."""

    def parse(self, result: NodeResult) -> list[str]:
        """Return list of child goal descriptions."""
        ...


class LLMSubgoalParser:
    """SubgoalParser backed by structured extraction.

    Primary strategy: look for ## TASKS block (Claude Code style).
    Fallback: numbered lists, markdown lists.
    """

    def parse(self, result: NodeResult) -> list[str]:
        """Extract child goals from result content."""
        content = result.summary

        # Strategy 1: Extract ## TASKS block (strict, primary — Claude Code style)
        goals = _try_parse_tasks_block(content)
        if goals:
            return goals

        # Strategy 2: Look for bare JSON array (fallback — common in some models)
        goals = _try_parse_json_goals(content)
        if goals:
            return goals

        # Strategy 3: Look for numbered list (fallback)
        goals = _try_parse_numbered_goals(content)
        if goals:
            return goals

        # Strategy 4: Look for markdown list (fallback)
        goals = _try_parse_markdown_goals(content)
        if goals:
            return goals

        return []


def _try_parse_tasks_block(content: str) -> list[str] | None:
    """Extract goals from ##[TASKS]...##[/TASKS] block.

    Format:
        ##[TASKS]
        ["goal 1", "goal 2", "goal 3"]
        ##[/TASKS]

    Also supports the legacy format:
        ## TASKS
        [...]
        ## TASKS

    Everything outside the markers is ignored — including <think> blocks.
    """
    # Match ##[TASKS] ... ##[/TASKS] (non-greedy, multiline)
    match = re.search(r'##\[TASKS\]\s*([\s\S]*?)\s*##\[/TASKS\]', content)
    if match:
        json_str = match.group(1).strip()
        if not json_str:
            return None
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    return []  # explicit empty = leaf node
                if all(isinstance(x, str) for x in parsed):
                    goals = [g.strip() for g in parsed if g.strip()]
                    if goals:
                        return goals
                if all(isinstance(x, dict) for x in parsed):
                    goals = [item.get("goal", "") for item in parsed]
                    goals = [g.strip() for g in goals if g.strip()]
                    if goals:
                        return goals
        except json.JSONDecodeError:
            pass

    # Fallback: try legacy ## TASKS ... ## TASKS format
    match = re.search(r'## TASKS\s*([\s\S]*?)\s*## TASKS', content)
    if not match:
        return None
    json_str = match.group(1).strip()
    if not json_str:
        return None
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            # Explicit empty array: ## TASKS [] ## TASKS means "no children (leaf node)"
            if len(parsed) == 0:
                return []  # Empty list = explicit "no subtasks"
            # String array format: ["goal 1", "goal 2"]
            if all(isinstance(x, str) for x in parsed):
                goals = [g.strip() for g in parsed if g.strip()]
                if goals:
                    return goals
            # Legacy dict format: [{"goal": "...", "description": "..."}]
            if all(isinstance(x, dict) for x in parsed):
                goals = [item.get("goal", "") for item in parsed]
                goals = [g.strip() for g in goals if g.strip()]
                if goals:
                    return goals
    except json.JSONDecodeError:
        pass
    return None


def _try_parse_json_goals(content: str) -> list[str] | None:
    """Try to extract a JSON array of goals from content."""
    # Look for JSON array patterns
    patterns = [
        r'\[\s*\{.*?"goal".*?\}\s*\]',  # non-greedy to handle multiple properties
        r'\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    return [g.strip() for g in parsed if g.strip()]
                if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                    goals = [g.get("goal", g.get("name", "")) for g in parsed]
                    return [g.strip() for g in goals if g.strip()]
            except json.JSONDecodeError:
                pass
    return None


def _try_parse_numbered_goals(content: str) -> list[str] | None:
    """Try to extract goals from numbered list (e.g. "1. First goal\n2. Second goal")."""
    lines = content.split("\n")
    goals = []
    for line in lines:
        line = line.strip()
        # Match "1. Goal text", "1) Goal text", or Chinese "1、 Goal text" patterns
        # Also supports fullwidth digits (１.) used by some models
        m = re.match(r'^\d+[.)．、]\s*(.+)', line)
        if m:
            goal = m.group(1).strip()
            if goal:
                goals.append(goal)
    if len(goals) >= 2:
        return goals
    return None


def _try_parse_markdown_goals(content: str) -> list[str] | None:
    """Try to extract goals from markdown list (e.g. "- First goal\n- Second goal")."""
    lines = content.split("\n")
    goals = []
    for line in lines:
        line = line.strip()
        # Match bullet list items (including Chinese "、" style)
        m = re.match(r'^[-*+●]\s*(.+)', line)
        if m:
            goal = m.group(1).strip()
            if goal:
                goals.append(goal)
    if len(goals) >= 2:
        return goals
    return None
