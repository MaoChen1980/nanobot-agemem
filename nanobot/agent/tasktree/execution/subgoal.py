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
    """SubgoalParser backed by simple pattern matching + LLM fallback.

    Primary strategy: look for structured markers like "[Subgoals]", numbered lists,
    or JSON arrays in the result content.
    """

    def parse(self, result: NodeResult) -> list[str]:
        """Extract child goals from result content."""
        content = result.summary

        # Strategy 1: Look for JSON array
        goals = _try_parse_json_goals(content)
        if goals:
            return goals

        # Strategy 2: Look for numbered list
        goals = _try_parse_numbered_goals(content)
        if goals:
            return goals

        # Strategy 3: Look for markdown list
        goals = _try_parse_markdown_goals(content)
        if goals:
            return goals

        return []


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
