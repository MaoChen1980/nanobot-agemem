"""Unit tests for the 6 AgeMem memory tools."""

import re
from pathlib import Path

import pytest

from nanobot.agent.tools.memory import (
    AddMemoryTool,
    UpdateMemoryTool,
    DeleteMemoryTool,
    RetrieveMemoriesTool,
    FilterMemoriesTool,
    SummarizeSessionTool,
)

# Full UUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
_UUID_RE = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"


def _extract_id(result: str) -> str:
    """Extract full UUID from tool result string."""
    match = re.search(_UUID_RE, result, re.IGNORECASE)
    if not match:
        raise AssertionError(f"No UUID found in: {result}")
    return match.group()


@pytest.fixture
def ws(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return tmp_path


class TestAddMemoryTool:
    @pytest.mark.asyncio
    async def test_add_returns_id_and_confirms(self, ws):
        tool = AddMemoryTool(ws)
        result = await tool.execute("Remember my name is Bob", importance=0.8, tags=["name"])

        mem_id = _extract_id(result)
        assert len(mem_id) == 36
        assert "name" in result.lower()
        assert "0.8" in result

    @pytest.mark.asyncio
    async def test_add_default_importance(self, ws):
        tool = AddMemoryTool(ws)
        result = await tool.execute("Some content")
        assert "0.5" in result  # default importance

    @pytest.mark.asyncio
    async def test_add_with_tags(self, ws):
        tool = AddMemoryTool(ws)
        result = await tool.execute("Test", tags=["tag1", "tag2"])
        assert "tag1" in result.lower() and "tag2" in result.lower()


class TestUpdateMemoryTool:
    @pytest.mark.asyncio
    async def test_update_existing_changes_importance(self, ws):
        add_tool = AddMemoryTool(ws)
        add_result = await add_tool.execute("Original content", importance=0.5)
        mem_id = _extract_id(add_result)

        update_tool = UpdateMemoryTool(ws)
        updated = await update_tool.execute(id=mem_id, importance=0.9)
        assert "updated" in updated.lower() or "0.9" in updated

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_not_found(self, ws):
        tool = UpdateMemoryTool(ws)
        result = await tool.execute(id="00000000-0000-0000-0000-000000000000", importance=0.9)
        assert "not found" in result.lower() or "does not exist" in result.lower()

    @pytest.mark.asyncio
    async def test_update_content_only(self, ws):
        add_tool = AddMemoryTool(ws)
        add_result = await add_tool.execute("Original content")
        mem_id = _extract_id(add_result)

        update_tool = UpdateMemoryTool(ws)
        updated = await update_tool.execute(id=mem_id, content="Updated content")
        assert "updated" in updated.lower()


class TestDeleteMemoryTool:
    @pytest.mark.asyncio
    async def test_delete_marks_as_deleted(self, ws):
        add_tool = AddMemoryTool(ws)
        add_result = await add_tool.execute("To be deleted")
        mem_id = _extract_id(add_result)

        delete_tool = DeleteMemoryTool(ws)
        result = await delete_tool.execute(mem_id)
        assert "deleted" in result.lower() or "removed" in result.lower()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_not_found(self, ws):
        tool = DeleteMemoryTool(ws)
        result = await tool.execute(id="00000000-0000-0000-0000-000000000000")
        assert "not found" in result.lower() or "does not exist" in result.lower()


class TestRetrieveMemoriesTool:
    @pytest.mark.asyncio
    async def test_retrieve_empty_query_returns_all(self, ws):
        add_tool = AddMemoryTool(ws)
        await add_tool.execute("Remember the deadline is Friday", importance=0.7, tags=["deadline"])

        retrieve_tool = RetrieveMemoriesTool(ws)
        result = await retrieve_tool.execute("")

        assert "deadline" in result.lower() or "friday" in result.lower()

    @pytest.mark.asyncio
    async def test_retrieve_with_query_returns_relevant(self, ws):
        add_tool = AddMemoryTool(ws)
        await add_tool.execute("Remember the deadline is Friday", importance=0.7)
        await add_tool.execute("Remember my name is Alice", importance=0.8)

        retrieve_tool = RetrieveMemoriesTool(ws)
        result = await retrieve_tool.execute("deadline")

        assert "deadline" in result.lower() or "friday" in result.lower()

    @pytest.mark.asyncio
    async def test_retrieve_top_k_limits_results(self, ws):
        add_tool = AddMemoryTool(ws)
        for i in range(5):
            await add_tool.execute(f"Memory item {i}", importance=0.5)

        retrieve_tool = RetrieveMemoriesTool(ws)
        result = await retrieve_tool.execute("memory", top_k=2)

        # top_k=2 should return at most 2 results
        # Count entry lines (each entry has 2-3 lines)
        id_count = len(re.findall(r"\[id=", result))
        assert id_count <= 2


class TestFilterMemoriesTool:
    @pytest.mark.asyncio
    async def test_filter_by_tag(self, ws):
        add_tool = AddMemoryTool(ws)
        await add_tool.execute("I prefer dark mode", importance=0.6, tags=["preference"])
        await add_tool.execute("Remember the deadline", importance=0.7, tags=["deadline"])

        filter_tool = FilterMemoriesTool(ws)
        result = await filter_tool.execute(tags=["preference"])

        assert "preference" in result.lower()
        assert "deadline" not in result.lower()

    @pytest.mark.asyncio
    async def test_filter_by_importance(self, ws):
        add_tool = AddMemoryTool(ws)
        await add_tool.execute("Low importance item", importance=0.2)
        await add_tool.execute("High importance item", importance=0.9)

        filter_tool = FilterMemoriesTool(ws)
        result = await filter_tool.execute(min_importance=0.8)

        assert "high" in result.lower() or "0.9" in result
        assert "low" not in result.lower()


class TestSummarizeSessionTool:
    @pytest.mark.asyncio
    async def test_summarize_returns_content(self, ws):
        tool = SummarizeSessionTool(ws)
        result = await tool.execute("Discussed project deadline and budget with the team")

        assert "deadline" in result.lower() or "budget" in result.lower()
        assert len(result) > 10

    @pytest.mark.asyncio
    async def test_summarize_empty_description(self, ws):
        tool = SummarizeSessionTool(ws)
        result = await tool.execute("")
        assert isinstance(result, str)
