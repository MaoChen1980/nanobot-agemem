"""MemoryHook: integrates Conscious Memory (Reflector + MemoryPolicy) into the agent loop.

Monitors tool execution and:
- Records retrieval gaps via Reflector
- Tracks memory access for policy learning
- Computes reward signals (Rtask, Rcontext, Rmemory)
- GRPO credit assignment for memory policy optimization
- Triggers proactive memory storage based on learned rules
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.agemem.reflector import Reflector
from nanobot.agent.agemem.policy import MemoryPolicy
from nanobot.agent.agemem.importance import ImportanceScorer
from nanobot.agent.agemem.rewards import RewardFunctions
from nanobot.agent.agemem.grpo import GRPOCreditAssignment
from nanobot.agent.agemem.store import MemoryStoreV2

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


MEMORY_TOOLS = frozenset({
    "add_memory",
    "update_memory",
    "delete_memory",
    "retrieve_memories",
    "filter_memories",
    "summarize_session",
})


class MemoryHook(AgentHook):
    """Hook that implements conscious memory feedback loops.

    Integrates:
    - Reflector: gap detection
    - MemoryPolicy: importance learning
    - ImportanceScorer: LLM-as-judge scoring
    - RewardFunctions: Rtask, Rcontext, Rmemory
    - GRPOCreditAssignment: step-wise reward broadcast
    """

    def __init__(
        self,
        workspace: Path,
        provider: "LLMProvider | None" = None,
        model: str = "gpt-4o-mini",
        gamma: float = 0.95,
        auto_add_enabled: bool = True,
    ):
        super().__init__()
        self._workspace = workspace
        self._reflector = Reflector(workspace)
        self._policy = MemoryPolicy(workspace)
        self._scorer = ImportanceScorer()
        self._rewards = RewardFunctions()
        self._grpo = GRPOCreditAssignment(workspace, gamma=gamma)
        self._model = model
        self._provider = provider
        self._auto_add_enabled = auto_add_enabled
        self._trajectory_started = False
        self._task_description = ""

        if provider:
            self._scorer.set_provider(provider, model)
            self._rewards.set_judge(provider, model)

    def set_provider(self, provider: "LLMProvider", model: str | None = None) -> None:
        """Set LLM provider after construction (called by AgentLoop)."""
        self._provider = provider
        if model:
            self._model = model
        self._scorer.set_provider(provider, model)
        self._rewards.set_judge(provider, model)

    @property
    def _memory_store(self) -> MemoryStoreV2:
        """Lazy-access MemoryStoreV2 for auto-add operations."""
        if not hasattr(self, "_store"):
            self._store = MemoryStoreV2(self._workspace)
        return self._store

    @property
    def reflector(self) -> Reflector:
        return self._reflector

    @property
    def policy(self) -> MemoryPolicy:
        return self._policy

    @property
    def scorer(self) -> ImportanceScorer:
        return self._scorer

    @property
    def rewards(self) -> RewardFunctions:
        return self._rewards

    @property
    def grpo(self) -> GRPOCreditAssignment:
        return self._grpo

    async def before_iteration(self, context: AgentHookContext) -> None:
        """Start a GRPO trajectory on the first iteration.

        Extracts task description from user messages.
        """
        if self._trajectory_started:
            return

        # Extract task description from messages
        messages = context.messages or []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    # Strip the Runtime Context block so GRPO task description
                    # contains only the actual user message, not metadata.
                    text = content.strip()
                    ctx_start = "[Runtime Context"
                    ctx_end = "[/Runtime Context]"
                    if ctx_start in text:
                        end_idx = text.find(ctx_end)
                        if end_idx != -1:
                            text = text[end_idx + len(ctx_end):].strip()
                    self._task_description = text[:200]  # Cap at 200 chars
                    break

        # Start GRPO trajectory
        if self._task_description:
            self._grpo.start_trajectory(self._task_description)
            self._trajectory_started = True
            logger.debug("MemoryHook: GRPO trajectory started for: {}", self._task_description[:50])

    async def after_iteration(self, context: AgentHookContext) -> None:
        """Process memory tool events after each iteration.

        - retrieve_memories: detect gaps (empty/irrelevant results)
        - add_memory/update_memory: record importance, compute Rmemory
        - summarize_session: compute Rcontext
        - Track operations in GRPO trajectory
        - Auto-add: proactively store important content based on learned policy
        """
        for event in context.tool_events:
            name = event.get("name", "")
            status = event.get("status", "")
            detail = event.get("detail", "")

            if name == "retrieve_memories":
                self._handle_retrieve(status, detail, context)
            elif name in MEMORY_TOOLS and status == "ok":
                self._handle_memory_operation(name, detail, context)

        # Phase 2d: proactive auto-add based on learned policy
        if self._auto_add_enabled and self._trajectory_started:
            await self._try_auto_add(context)

    def finalize_content(
        self,
        context: AgentHookContext,
        content: str | None,
    ) -> str | None:
        """Complete the GRPO trajectory with Rtask reward.

        Called when the agent loop finishes. Broadcasts rewards to all
        memory operations in the trajectory.
        """
        if not self._trajectory_started:
            self._trajectory_started = False
            return content

        self._trajectory_started = False

        # Compute Rtask
        final_content = content or ""
        if self._task_description:
            rtask = self._rewards.compute_rtask_sync(final_content, self._task_description)

            # Complete GRPO trajectory with reward broadcast
            credits = self._grpo.complete_trajectory(rtask, self._rewards)

            if credits:
                # Apply credit assignment to MemoryPolicy
                updates = self._grpo.compute_policy_updates(credits, self._policy)
                logger.info(
                    "MemoryHook: GRPO completed, Rtask={:.3f}, {} credits, {} policy updates",
                    rtask, len(credits), len(updates),
                )
                for update in updates[:5]:
                    logger.debug("  Policy update: {}", update)

        return content

    def _handle_retrieve(self, status: str, detail: str, context: AgentHookContext) -> None:
        """Handle retrieve_memories tool result.

        Detects if the retrieval was empty (gap) or had results.
        Records the operation in GRPO trajectory.
        """
        if status != "ok":
            return

        # Parse detail to check if empty
        was_empty = "No memories found" in detail
        count = 0
        if not was_empty:
            count = len(re.findall(r"\[id=", detail))

        self._reflector.on_retrieve(query="", retrieved_count=count, was_empty=was_empty)

        # Record in GRPO trajectory
        if self._trajectory_started:
            # Extract first memory ID if available
            memory_id_match = re.search(r"\[id=([a-f0-9-]+)", detail)
            memory_id = memory_id_match.group(1) if memory_id_match else None
            self._grpo.record_step(
                operation="retrieve",
                memory_id=memory_id,
                content=f"Found {count} memories" if count > 0 else "No memories found",
                query="",
            )

        if was_empty:
            logger.debug("MemoryHook: retrieve_memories returned empty (gap)")

    def _handle_memory_operation(
        self,
        tool_name: str,
        detail: str,
        context: AgentHookContext,
    ) -> None:
        """Handle successful memory tool execution.

        Records importance signals and computes reward functions.
        Tracks operations in GRPO trajectory.
        """
        if tool_name == "retrieve_memories":
            return  # Already handled

        # Extract memory ID and content from detail
        memory_id = self._extract_id_from_detail(detail)
        content = self._extract_content_from_detail(detail)

        # Record in GRPO trajectory
        if self._trajectory_started and tool_name in ("add_memory", "update_memory", "delete_memory"):
            self._grpo.record_step(
                operation=tool_name.replace("_memory", ""),
                memory_id=memory_id,
                content=content,
            )

        if tool_name == "add_memory":
            self._handle_add_memory(content, detail)
        elif tool_name == "update_memory":
            self._handle_update_memory(memory_id, content)
        elif tool_name == "delete_memory":
            self._handle_delete_memory(memory_id)
        elif tool_name == "summarize_session":
            self._handle_summarize_session(content, context)

    def _handle_add_memory(self, content: str, detail: str) -> None:
        """Handle successful add_memory.

        Records explicit importance signal.
        """
        if not content:
            return

        # Check for explicit importance signals in content
        explicit_score = self._scorer.extract_importance_signals(content)
        if explicit_score is not None:
            importance = explicit_score
            self._policy.record_explicit(content, importance)
            logger.debug("MemoryHook: explicit importance recorded: {:.2f}", importance)
        else:
            importance = self._scorer.score_sync(content)
            self._policy.record_self_assessed(content, importance)
            logger.debug("MemoryHook: self-assessed importance: {:.2f}", importance)

    def _handle_update_memory(self, memory_id: str, content: str) -> None:
        """Handle successful update_memory."""
        if memory_id and content:
            self._policy.record_access(memory_id, content)

    def _handle_delete_memory(self, memory_id: str) -> None:
        """Handle successful delete_memory."""
        logger.debug("MemoryHook: memory deleted: {}", memory_id)

    def _handle_summarize_session(self, summary: str, context: AgentHookContext) -> None:
        """Handle successful summarize_session.

        Computes Rcontext compression reward.
        """
        if not summary:
            return
        logger.debug("MemoryHook: session summarized, {} chars", len(summary))

    async def _try_auto_add(self, context: AgentHookContext) -> None:
        """Check if recent LLM response contains important content to auto-add.

        Called after each iteration when auto_add_enabled is True and a
        trajectory is running. Extracts content from the LLM response,
        checks against learned auto_add rules, and writes to LTM if matched.
        """
        response = context.response
        if not response or not response.content:
            return

        content = response.content.strip()
        if not content or len(content) < 10:
            return

        # Extract potentially important content from the response
        important_content = self._extract_important_content(content)
        if not important_content:
            return

        # Check against learned policy rules
        should_add, importance = self._policy.should_auto_add(important_content)
        if not should_add:
            return

        # Trigger auto-add: write directly to LTM store
        # Extract tags from importance signals
        tags = self._extract_tags_from_content(important_content)
        entry_id = self._memory_store.add(
            content={"text": important_content},
            importance=importance,
            tags=tags,
        )

        # Record in GRPO trajectory
        self._grpo.record_step(
            operation="auto_add",
            memory_id=entry_id,
            content=important_content[:100],
        )
        # Also record self-assessed importance in policy for rule learning
        self._policy.record_self_assessed(important_content, importance)

        logger.info(
            "MemoryHook: auto-added memory id={}, importance={:.2f}, content='{}'",
            entry_id, importance, important_content[:80],
        )

    def _extract_important_content(self, text: str) -> str | None:
        """Extract potentially important factual content from LLM response text.

        Looks for sentences that contain:
        - Specific facts (names, dates, numbers)
        - Explicit user signals ("you said", "your", "I learned")
        - Direct answers to user questions
        """
        if not text:
            return None

        # Skip very short responses
        if len(text) < 15:
            return None

        # Check for explicit importance signals first
        signal = self._scorer.extract_importance_signals(text)
        if signal is not None and signal >= 0.5:
            # Return the full meaningful content (up to 200 chars)
            return text[:200].strip()

        # Skip responses that are mostly questions (agent asking for clarification)
        question_count = text.count("?")
        if question_count > len(text) / 50:  # More than 1 question per 50 chars
            return None

        # Skip generic/filler responses
        generic_patterns = ["i'm not sure", "i don't know", "cannot answer", "no information"]
        text_lower = text.lower()
        if any(p in text_lower for p in generic_patterns):
            return None

        # Look for sentences with potentially important content
        # (contains names, specific terms, or factual statements)
        sentences = re.split(r"[.。!！\n]+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            # Check for importance signals
            if self._scorer.extract_importance_signals(sentence) is not None:
                return sentence[:200]
            # Check for specific patterns indicating facts
            if any(kw in sentence.lower() for kw in [
                "your name", "you are", "you said", "remember",
                "preference", "always", "never", "deadline", "project",
                "allergic", "important",
            ]):
                return sentence[:200]

        return None

    def _extract_tags_from_content(self, content: str) -> list[str]:
        """Extract tags from content based on keywords."""
        tags = []
        content_lower = content.lower()
        if any(w in content_lower for w in ["preference", "like", "prefer"]):
            tags.append("preference")
        if any(w in content_lower for w in ["name", "called"]):
            tags.append("identity")
        if any(w in content_lower for w in ["allergic", "health", "medical"]):
            tags.append("health")
        if any(w in content_lower for w in ["deadline", "schedule", "date"]):
            tags.append("schedule")
        if any(w in content_lower for w in ["project", "task", "work"]):
            tags.append("project")
        return tags if tags else ["auto-added"]

    def _extract_id_from_detail(self, detail: str) -> str | None:
        """Extract memory ID from tool detail string."""
        match = re.search(r"ID: ([a-f0-9-]{36})", detail)
        return match.group(1) if match else None

    def _extract_content_from_detail(self, detail: str) -> str | None:
        """Extract memory content preview from tool detail string."""
        match = re.search(r"Content: (.{0,100})", detail)
        if match:
            return match.group(1).strip()
        # Fallback: try to get Added/Updated content
        match2 = re.search(r"(?:Added|Updated): (.{0,100})", detail)
        if match2:
            return match2.group(1).strip()
        return None
