"""MemoryPolicy: learns when to proactively store memories based on importance signals.

Implements the "有意识记忆" (conscious memory) policy:
- Records importance signals from various sources
- Maintains auto-add rules derived from reflection
- Updates memory importance based on observed utility
"""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class AutoAddRule:
    """A learned rule for proactive memory storage."""
    id: str
    pattern: str  # keyword pattern to match
    importance: float
    created_at: str
    hit_count: int = 0
    last_hit: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AutoAddRule":
        return cls(**d)


class MemoryPolicy:
    """Learns proactive memory storage policy from importance signals.

    The policy maintains auto-add rules that tell the agent when to
    proactively store information before being asked.

    Signal types:
    - explicit: user said "remember this" (weight=1.0, creates rule with max importance)
    - self_assessed: agent decided to add (weight=0.7)
    - reflected: learned from gap (weight=0.9, creates/updates rule)
    - accessed: memory was retrieved and used successfully (weight=0.4, boosts importance)
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.rules_file = workspace / "memory" / "policy_rules.jsonl"
        self._rules: dict[str, AutoAddRule] = {}
        self._ensure_file()
        self._load()

    def _ensure_file(self) -> None:
        if not self.rules_file.exists():
            self.rules_file.parent.mkdir(parents=True, exist_ok=True)
            self.rules_file.write_text("", encoding="utf-8")

    def _load(self) -> None:
        """Load rules from JSONL file."""
        self._rules.clear()
        if not self.rules_file.exists():
            return
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rule = AutoAddRule.from_dict(d)
                        self._rules[rule.id] = rule
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            pass

    def _save(self) -> None:
        """Persist all rules to JSONL."""
        with open(self.rules_file, "w", encoding="utf-8") as f:
            for rule in self._rules.values():
                f.write(json.dumps(rule.to_dict(), ensure_ascii=False) + "\n")

    def record_explicit(self, content: str, importance: float = 1.0) -> AutoAddRule:
        """User explicitly asked to remember something.

        Creates a high-priority auto-add rule.
        """
        pattern = self._extract_pattern(content)
        now = datetime.now()
        rule = AutoAddRule(
            id=f"explicit_{now.strftime('%Y%m%d%H%M%S')}{now.microsecond:06d}",
            pattern=pattern,
            importance=importance,
            created_at=now.isoformat(),
            reason="explicit: user said remember this",
        )
        self._rules[rule.id] = rule
        self._save()
        logger.debug("Policy: explicit rule created: pattern='{}'", pattern)
        return rule

    def record_self_assessed(self, content: str, importance: float = 0.7) -> AutoAddRule | None:
        """Agent self-assessed that content is important.

        Creates or updates an auto-add rule.
        """
        pattern = self._extract_pattern(content)
        if not pattern:
            return None
        # Check if similar rule exists
        existing = self._find_similar_rule(pattern)
        if existing:
            # Boost importance slightly
            existing.importance = min(existing.importance + 0.1, 1.0)
            existing.hit_count += 1
            existing.last_hit = datetime.now().isoformat()
            self._save()
            logger.debug("Policy: self-assessed rule boosted: pattern='{}'", existing.pattern)
            return existing

        rule = AutoAddRule(
            id=f"selfassess_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            pattern=pattern,
            importance=importance,
            created_at=datetime.now().isoformat(),
            reason="self_assessed: agent decided to add",
        )
        self._rules[rule.id] = rule
        self._save()
        logger.debug("Policy: self-assessed rule created: pattern='{}'", pattern)
        return rule

    def record_reflected(self, recommendations: list[dict[str, Any]]) -> list[AutoAddRule]:
        """Record policy updates derived from reflection on memory gaps.

        Called by Dream after reflect() generates recommendations.
        """
        rules = []
        for rec in recommendations:
            if rec.get("action") != "add_rule":
                continue
            pattern = rec.get("pattern", "")
            if not pattern:
                continue

            # Check if rule for this pattern already exists
            existing = self._find_similar_rule(pattern)
            if existing:
                # Increase importance
                delta = rec.get("importance", 0.5) - existing.importance
                existing.importance = min(existing.importance + delta * 0.5, 1.0)
                existing.hit_count += 1
                existing.last_hit = datetime.now().isoformat()
                existing.reason = rec.get("reason", existing.reason)
                self._save()
                rules.append(existing)
                logger.debug("Policy: reflected rule updated: pattern='{}'", existing.pattern)
            else:
                rule = AutoAddRule(
                    id=f"reflected_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    pattern=pattern,
                    importance=rec.get("importance", 0.5),
                    created_at=datetime.now().isoformat(),
                    reason=rec.get("reason", "reflected from gap analysis"),
                )
                self._rules[rule.id] = rule
                self._save()
                rules.append(rule)
                logger.info("Policy: reflected rule created: pattern='{}', importance={}",
                           pattern, rule.importance)
        return rules

    def record_access(self, memory_id: str, content: str) -> None:
        """Record that a memory was successfully accessed and used.

        This is a positive signal that should slightly boost related rules.
        """
        pattern = self._extract_pattern(content)
        if not pattern:
            return
        related = self._find_similar_rule(pattern)
        if related:
            # Slight boost when accessed memory is useful
            related.importance = min(related.importance + 0.05, 1.0)
            related.hit_count += 1
            related.last_hit = datetime.now().isoformat()
            self._save()

    def get_matching_rules(self, content: str) -> list[AutoAddRule]:
        """Return all rules whose patterns match the given content.

        Uses keyword overlap scoring: if at least 2 pattern terms appear
        in the content, consider it a match.
        """
        content_pattern = self._extract_pattern(content)
        if not content_pattern:
            return []
        content_terms = set(content_pattern.split())

        matched = []
        for rule in self._rules.values():
            rule_terms = set(rule.pattern.lower().split())
            overlap = len(content_terms & rule_terms)
            if overlap >= 2:  # require at least 2 matching terms
                matched.append(rule)
        # Sort by importance descending
        matched.sort(key=lambda r: r.importance, reverse=True)
        return matched

    def should_auto_add(self, content: str) -> tuple[bool, float]:
        """Check if content matches any auto-add rules.

        Returns (should_add, recommended_importance).
        """
        matching = self.get_matching_rules(content)
        if not matching:
            return False, 0.0
        top = matching[0]
        return True, top.importance

    def get_auto_add_rules(self) -> list[AutoAddRule]:
        """Return all active auto-add rules sorted by importance."""
        return sorted(self._rules.values(), key=lambda r: r.importance, reverse=True)

    def _extract_pattern(self, content: str) -> str:
        """Extract a distinctive keyword pattern from content.

        Returns a short comma-separated list of key terms.
        """
        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "or", "if", "because", "until", "while", "about",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "it", "its", "they", "them", "their", "we", "us", "our",
            "you", "your", "he", "him", "his", "she", "her",
            "i", "me", "my", "myself", "any", "both", "either", "neither",
        }
        terms = re.findall(r"[a-z0-9]{3,}", content.lower())
        significant = [t for t in terms if t not in stopwords]
        # Return top 5 most distinctive (prefer longer terms)
        significant.sort(key=len, reverse=True)
        return " ".join(significant[:5])

    def _find_similar_rule(self, pattern: str) -> AutoAddRule | None:
        """Find a rule with high pattern overlap."""
        pattern_terms = set(pattern.lower().split())
        if not pattern_terms:
            return None
        best: AutoAddRule | None = None
        best_score = 0.0
        for rule in self._rules.values():
            rule_terms = set(rule.pattern.lower().split())
            if not rule_terms:
                continue
            overlap = len(pattern_terms & rule_terms)
            union = len(pattern_terms | rule_terms)
            score = overlap / union if union > 0 else 0.0
            if score > best_score and score > 0.3:
                best_score = score
                best = rule
        return best
