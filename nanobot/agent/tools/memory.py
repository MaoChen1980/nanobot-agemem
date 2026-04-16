"""AgeMem memory tools: LTM (Add/Update/Delete) and STM (Retrieve/Summary/Filter).

These tools expose structured memory operations as AgentRunner tools, following
the AgeMem paper's tool-based memory management paradigm.
"""

from pathlib import Path
from typing import Any

from nanobot.agent.agemem.retriever import MemoryRetriever
from nanobot.agent.agemem.store import MemoryStoreV2
from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    IntegerSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)


class _MemoryTool(Tool):
    """Base for memory tools — provides shared store and retriever initialization."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._store: MemoryStoreV2 | None = None
        self._retriever: MemoryRetriever | None = None

    @property
    def store(self) -> MemoryStoreV2:
        if self._store is None:
            self._store = MemoryStoreV2(self._workspace)
        return self._store

    @property
    def retriever(self) -> MemoryRetriever:
        if self._retriever is None:
            self._retriever = MemoryRetriever(self.store)
        return self._retriever


# ---------------------------------------------------------------------------
# LTM Tools (Long-Term Memory)
# ---------------------------------------------------------------------------


@tool_parameters(
    tool_parameters_schema(
        required=["content"],
        content=StringSchema("The memory content to store"),
        importance=NumberSchema(
            0.5,
            description="Importance score from 0.0 (trivial) to 1.0 (critical)",
            minimum=0.0,
            maximum=1.0,
        ),
        tags=ArraySchema(
            StringSchema(""),
            description="Optional tags for categorizing this memory",
        ),
    )
)
class AddMemoryTool(_MemoryTool):
    """Add a new entry to long-term memory (AgeMem LTM Add operation)."""

    name = "add_memory"
    description = "Store a new fact, preference, or piece of knowledge in long-term memory. Use this when the user shares information that should be remembered across sessions."
    read_only = False

    async def execute(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str:
        entry = self.store.add(content=content, importance=importance, tags=tags or [])
        return (
            f"Added memory entry:\n"
            f"  ID: {entry.id}\n"
            f"  Importance: {entry.importance}\n"
            f"  Tags: {', '.join(entry.tags) or '(none)'}\n"
            f"  Created: {entry.created_at}"
        )


@tool_parameters(
    tool_parameters_schema(
        required=["id"],
        id=StringSchema("The memory entry ID to update"),
        content=StringSchema("Updated content", nullable=True),
        importance=NumberSchema(
            description="New importance score (0.0-1.0)",
            minimum=0.0,
            maximum=1.0,
            nullable=True,
        ),
        tags=ArraySchema(
            StringSchema(""),
            description="New tags list (replaces existing)",
            nullable=True,
        ),
    )
)
class UpdateMemoryTool(_MemoryTool):
    """Update an existing long-term memory entry (AgeMem LTM Update operation)."""

    name = "update_memory"
    description = "Update the content, importance, or tags of an existing memory entry. Use when information changes or needs correction."
    read_only = False

    async def execute(
        self,
        id: str,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> str:
        entry = self.store.update(id=id, content=content, importance=importance, tags=tags)
        if entry is None:
            return f"Memory entry '{id}' not found."
        return (
            f"Updated memory entry:\n"
            f"  ID: {entry.id}\n"
            f"  Content: {entry.content[:100]}{'...' if len(entry.content) > 100 else ''}\n"
            f"  Importance: {entry.importance}\n"
            f"  Tags: {', '.join(entry.tags) or '(none)'}\n"
            f"  Updated: {entry.updated_at}"
        )


@tool_parameters(
    tool_parameters_schema(
        required=["id"],
        id=StringSchema("The memory entry ID to delete"),
    )
)
class DeleteMemoryTool(_MemoryTool):
    """Soft-delete a long-term memory entry (AgeMem LTM Delete operation)."""

    name = "delete_memory"
    description = "Remove an obsolete or incorrect memory entry. The entry is soft-deleted and no longer retrieved."
    read_only = False

    async def execute(self, id: str) -> str:
        success = self.store.delete(id=id)
        if not success:
            return f"Memory entry '{id}' not found."
        return f"Deleted memory entry '{id}'."


# ---------------------------------------------------------------------------
# STM Tools (Short-Term Memory)
# ---------------------------------------------------------------------------


@tool_parameters(
    tool_parameters_schema(
        required=["query"],
        query=StringSchema("The query to search memories for"),
        top_k=IntegerSchema(5, description="Maximum number of results to return"),
    )
)
class RetrieveMemoriesTool(_MemoryTool):
    """Retrieve relevant memories from long-term storage (AgeMem STM Retrieve operation).

    This tool performs BM25-based semantic search over stored memories.
    """

    name = "retrieve_memories"
    description = "Search long-term memory for entries relevant to the current task or question. Returns top-k most relevant memories with their relevance scores."
    read_only = True

    async def execute(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        scored_entries = self.retriever.retrieve(query, top_k=top_k)
        if not scored_entries:
            return "No memories found."
        lines = []
        for se in scored_entries:
            e = se.entry
            freshness = f" {se.freshness_label}" if se.freshness_label else ""
            ts = e.created_at[:16] if e.created_at else ""
            lines.append(
                f"[id={e.id}] relevance={se.score:.3f}{freshness} created={ts} tags={e.tags}\n"
                f"  {e.content[:200]}{'...' if len(e.content) > 200 else ''}"
            )
        return "Retrieved memories:\n" + "\n".join(lines)


@tool_parameters(
    tool_parameters_schema(
        tags=ArraySchema(
            StringSchema(""),
            description="Filter by these tags (any match)",
        ),
        min_importance=NumberSchema(
            0.0,
            description="Minimum importance score (0.0-1.0)",
            minimum=0.0,
            maximum=1.0,
        ),
    )
)
class FilterMemoriesTool(_MemoryTool):
    """Filter long-term memories by tags and importance (AgeMem STM Filter operation)."""

    name = "filter_memories"
    description = "Filter stored memories by tags and/or minimum importance threshold. Use to find memories about specific topics or of minimum significance."
    read_only = True

    async def execute(
        self,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
    ) -> str:
        entries = self.retriever.filter(tags=tags, min_importance=min_importance)
        if not entries:
            return "No memories match the filter."
        lines = []
        for e in entries:
            lines.append(
                f"[id={e.id}] importance={e.importance} tags={e.tags}\n"
                f"  {e.content[:200]}{'...' if len(e.content) > 200 else ''}"
            )
        return "Filtered memories:\n" + "\n".join(lines)


@tool_parameters(
    tool_parameters_schema(
        session_description=StringSchema("Optional description of what to summarize from recent conversation"),
    )
)
class SummarizeSessionTool(_MemoryTool):
    """Summarize recent conversation and store highlights in long-term memory (AgeMem STM Summary operation).

    This consolidates the current session's key points into persistent memory.
    """

    name = "summarize_session"
    description = "Summarize key information from the current conversation and store it in long-term memory. Use when important facts emerge that should be remembered."
    read_only = False

    async def execute(self, description: str = "") -> str:
        # Phase 2: actually call LLM to summarize session messages
        # Phase 1: record the description as a memory
        if description:
            entry = self.store.add(
                content=f"Session note: {description}",
                importance=0.5,
                tags=["session-summary"],
            )
            return (
                f"Summarized session into memory:\n"
                f"  ID: {entry.id}\n"
                f"  Content: {description[:100]}{'...' if len(description) > 100 else ''}"
            )
        return "No description provided. Provide a summary description of the key points to store."
