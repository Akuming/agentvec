"""
Tests for agentvec-mcp server.

These tests verify the MCP server tools for memory operations.
"""

import tempfile
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

# Import server components
from agentvec_mcp.server import (
    server,
    list_tools,
    call_tool,
    get_memory,
)
from agentvec_memory import MemoryTier


# --- Fixtures ---

@pytest.fixture
def mock_project_memory():
    """Create a mock ProjectMemory instance."""
    mock_memory = Mock()
    mock_memory.remember.return_value = "test-memory-id-1234"
    mock_memory.recall.return_value = []
    mock_memory.forget.return_value = 0
    mock_memory.get_stats.return_value = {
        "path": "/tmp/test",
        "tiers": {
            "working": 0,
            "session": 1,
            "project": 5,
            "user": 2,
        },
        "total_memories": 8,
    }
    return mock_memory


@pytest.fixture
def patched_memory(mock_project_memory):
    """Patch get_memory to return mock."""
    import agentvec_mcp.server as server_module
    original = server_module._memory
    server_module._memory = mock_project_memory
    yield mock_project_memory
    server_module._memory = original


# --- Tool Listing Tests ---

class TestListTools:
    """Tests for list_tools()."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_four_tools(self):
        """Test that list_tools returns all four memory tools."""
        tools = await list_tools()

        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "memory_remember" in tool_names
        assert "memory_recall" in tool_names
        assert "memory_forget" in tool_names
        assert "memory_stats" in tool_names

    @pytest.mark.asyncio
    async def test_memory_remember_schema(self):
        """Test memory_remember tool has correct schema."""
        tools = await list_tools()
        remember_tool = next(t for t in tools if t.name == "memory_remember")

        schema = remember_tool.inputSchema
        assert schema["type"] == "object"
        assert "content" in schema["properties"]
        assert "tier" in schema["properties"]
        assert "ttl" in schema["properties"]
        assert "content" in schema["required"]

    @pytest.mark.asyncio
    async def test_memory_recall_schema(self):
        """Test memory_recall tool has correct schema."""
        tools = await list_tools()
        recall_tool = next(t for t in tools if t.name == "memory_recall")

        schema = recall_tool.inputSchema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "k" in schema["properties"]
        assert "threshold" in schema["properties"]
        assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_memory_forget_schema(self):
        """Test memory_forget tool has correct schema."""
        tools = await list_tools()
        forget_tool = next(t for t in tools if t.name == "memory_forget")

        schema = forget_tool.inputSchema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "threshold" in schema["properties"]
        assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        tools = await list_tools()

        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 0


# --- Tool Call Tests ---

class TestMemoryRememberTool:
    """Tests for memory_remember tool."""

    @pytest.mark.asyncio
    async def test_remember_basic(self, patched_memory):
        """Test basic remember call."""
        result = await call_tool("memory_remember", {
            "content": "User prefers dark mode"
        })

        assert len(result) == 1
        assert "Remembered" in result[0].text
        assert "dark mode" in result[0].text

        patched_memory.remember.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_with_tier(self, patched_memory):
        """Test remember with explicit tier."""
        await call_tool("memory_remember", {
            "content": "Test fact",
            "tier": "user"
        })

        call_args = patched_memory.remember.call_args
        assert call_args.kwargs.get("tier") == MemoryTier.USER

    @pytest.mark.asyncio
    async def test_remember_with_ttl(self, patched_memory):
        """Test remember with custom TTL."""
        await call_tool("memory_remember", {
            "content": "Short-lived fact",
            "ttl": 60
        })

        call_args = patched_memory.remember.call_args
        assert call_args.kwargs.get("ttl") == 60

    @pytest.mark.asyncio
    async def test_remember_truncates_long_content(self, patched_memory):
        """Test that long content is truncated in response."""
        long_content = "A" * 200

        result = await call_tool("memory_remember", {
            "content": long_content
        })

        # Should truncate to 100 chars + "..."
        assert "..." in result[0].text


class TestMemoryRecallTool:
    """Tests for memory_recall tool."""

    @pytest.mark.asyncio
    async def test_recall_no_results(self, patched_memory):
        """Test recall when no memories found."""
        patched_memory.recall.return_value = []

        result = await call_tool("memory_recall", {
            "query": "nonexistent topic"
        })

        assert "No memories found" in result[0].text

    @pytest.mark.asyncio
    async def test_recall_with_results(self, patched_memory):
        """Test recall when memories are found."""
        mock_memory = Mock()
        mock_memory.id = "test-id"
        mock_memory.content = "User prefers dark mode"
        mock_memory.tier = MemoryTier.USER
        mock_memory.score = 0.85

        patched_memory.recall.return_value = [mock_memory]

        result = await call_tool("memory_recall", {
            "query": "user preferences"
        })

        assert "Found 1 memories" in result[0].text
        assert "dark mode" in result[0].text
        assert "0.850" in result[0].text

    @pytest.mark.asyncio
    async def test_recall_with_k_parameter(self, patched_memory):
        """Test recall respects k parameter."""
        await call_tool("memory_recall", {
            "query": "test",
            "k": 10
        })

        call_args = patched_memory.recall.call_args
        assert call_args.kwargs.get("k") == 10

    @pytest.mark.asyncio
    async def test_recall_with_threshold(self, patched_memory):
        """Test recall respects threshold parameter."""
        await call_tool("memory_recall", {
            "query": "test",
            "threshold": 0.5
        })

        call_args = patched_memory.recall.call_args
        assert call_args.kwargs.get("threshold") == 0.5

    @pytest.mark.asyncio
    async def test_recall_with_tiers_filter(self, patched_memory):
        """Test recall filters by tiers."""
        await call_tool("memory_recall", {
            "query": "test",
            "tiers": ["user", "project"]
        })

        call_args = patched_memory.recall.call_args
        tiers = call_args.kwargs.get("tiers")
        assert len(tiers) == 2
        assert MemoryTier.USER in tiers
        assert MemoryTier.PROJECT in tiers


class TestMemoryForgetTool:
    """Tests for memory_forget tool."""

    @pytest.mark.asyncio
    async def test_forget_basic(self, patched_memory):
        """Test basic forget call."""
        patched_memory.forget.return_value = 3

        result = await call_tool("memory_forget", {
            "query": "old information"
        })

        assert "Forgot 3 memories" in result[0].text
        patched_memory.forget.assert_called_once()

    @pytest.mark.asyncio
    async def test_forget_with_threshold(self, patched_memory):
        """Test forget respects threshold parameter."""
        await call_tool("memory_forget", {
            "query": "test",
            "threshold": 0.9
        })

        call_args = patched_memory.forget.call_args
        assert call_args.kwargs.get("threshold") == 0.9

    @pytest.mark.asyncio
    async def test_forget_with_tiers(self, patched_memory):
        """Test forget filters by tiers."""
        await call_tool("memory_forget", {
            "query": "test",
            "tiers": ["working"]
        })

        call_args = patched_memory.forget.call_args
        tiers = call_args.kwargs.get("tiers")
        assert len(tiers) == 1
        assert MemoryTier.WORKING in tiers


class TestMemoryStatsTool:
    """Tests for memory_stats tool."""

    @pytest.mark.asyncio
    async def test_stats_basic(self, patched_memory):
        """Test basic stats call."""
        result = await call_tool("memory_stats", {})

        assert "Memory Statistics" in result[0].text
        assert "Total: 8 memories" in result[0].text

    @pytest.mark.asyncio
    async def test_stats_includes_all_tiers(self, patched_memory):
        """Test stats includes all tier counts."""
        result = await call_tool("memory_stats", {})

        text = result[0].text
        assert "working: 0" in text
        assert "session: 1" in text
        assert "project: 5" in text
        assert "user: 2" in text


class TestUnknownTool:
    """Tests for unknown tool handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, patched_memory):
        """Test that unknown tool name returns error message."""
        result = await call_tool("nonexistent_tool", {})

        assert "Unknown tool" in result[0].text
        assert "nonexistent_tool" in result[0].text


# --- Error Handling Tests ---

class TestErrorHandling:
    """Tests for error handling in tool calls."""

    @pytest.mark.asyncio
    async def test_remember_handles_exception(self, patched_memory):
        """Test that exceptions in remember are handled gracefully."""
        patched_memory.remember.side_effect = Exception("Database error")

        result = await call_tool("memory_remember", {
            "content": "test"
        })

        assert "Error" in result[0].text or "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_recall_handles_exception(self, patched_memory):
        """Test that exceptions in recall are handled gracefully."""
        patched_memory.recall.side_effect = Exception("Search failed")

        result = await call_tool("memory_recall", {
            "query": "test"
        })

        assert "Error" in result[0].text or "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_invalid_tier_handled(self, patched_memory):
        """Test that invalid tier value is handled."""
        result = await call_tool("memory_remember", {
            "content": "test",
            "tier": "invalid_tier"
        })

        # Should return error message, not crash
        assert result is not None
        assert len(result) > 0
