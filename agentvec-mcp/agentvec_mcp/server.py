"""
MCP Server for AgentVec Memory.

Exposes memory tools (remember, recall, forget) to AI agents via MCP protocol.
"""

import os
import sys
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from agentvec_memory import ProjectMemory, MemoryTier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # MCP uses stdio, so log to stderr
)
logger = logging.getLogger("agentvec-mcp")

# Initialize server
server = Server("agentvec-memory")

# Memory instance (initialized lazily)
_memory: ProjectMemory | None = None


class MemoryInitError(Exception):
    """Raised when memory initialization fails."""
    pass


def get_memory() -> ProjectMemory:
    """Get or create the memory instance."""
    global _memory
    if _memory is None:
        # Use environment variable or default path
        memory_path = os.environ.get(
            "AGENTVEC_MEMORY_PATH",
            os.path.join(os.getcwd(), ".agentvec-memory")
        )
        logger.info(f"Initializing memory at: {memory_path}")
        try:
            _memory = ProjectMemory(memory_path)
            logger.info("Memory initialized successfully")
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            raise MemoryInitError(
                f"Failed to initialize memory: {e}. "
                "Make sure agentvec and sentence-transformers are installed."
            )
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            raise MemoryInitError(f"Failed to initialize memory: {e}")
    return _memory


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available memory tools."""
    return [
        Tool(
            name="memory_remember",
            description=(
                "Store a fact or piece of information in persistent memory. "
                "Use this to remember important context, user preferences, "
                "project details, or anything that should persist across sessions. "
                "Choose the appropriate tier based on how long the memory should last."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact or information to remember"
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["working", "session", "project", "user"],
                        "description": (
                            "Memory tier: "
                            "'working' (5 min, current task), "
                            "'session' (1 hour, current session), "
                            "'project' (30 days, project knowledge), "
                            "'user' (1 year, user preferences)"
                        ),
                        "default": "project"
                    },
                    "ttl": {
                        "type": "integer",
                        "description": "Optional custom TTL in seconds (overrides tier default)"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="memory_recall",
            description=(
                "Search for relevant memories using natural language. "
                "Use this to retrieve previously stored information, context, "
                "or facts that might be relevant to the current task."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search memories"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "default": 0.3
                    },
                    "tiers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["working", "session", "project", "user"]
                        },
                        "description": "Limit search to specific tiers (searches all if not specified)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memory_forget",
            description=(
                "Remove memories that match a query. "
                "Use this to delete outdated, incorrect, or no longer relevant information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to match memories for deletion"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold for deletion (higher = stricter match)",
                        "default": 0.8
                    },
                    "tiers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["working", "session", "project", "user"]
                        },
                        "description": "Limit deletion to specific tiers"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="memory_stats",
            description=(
                "Get statistics about stored memories. "
                "Shows count of memories in each tier."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


def _error_response(message: str) -> list[TextContent]:
    """Create an error response."""
    return [TextContent(type="text", text=f"Error: {message}")]


def _parse_tier(tier_str: str) -> MemoryTier:
    """Parse tier string to MemoryTier enum with validation."""
    valid_tiers = ["working", "session", "project", "user"]
    if tier_str not in valid_tiers:
        raise ValueError(
            f"Invalid tier '{tier_str}'. Must be one of: {', '.join(valid_tiers)}"
        )
    return MemoryTier(tier_str)


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls with comprehensive error handling."""

    # Initialize memory (may fail)
    try:
        memory = get_memory()
    except MemoryInitError as e:
        logger.error(f"Memory initialization failed: {e}")
        return _error_response(str(e))

    # Handle each tool
    try:
        if name == "memory_remember":
            return await _handle_remember(memory, arguments)
        elif name == "memory_recall":
            return await _handle_recall(memory, arguments)
        elif name == "memory_forget":
            return await _handle_forget(memory, arguments)
        elif name == "memory_stats":
            return await _handle_stats(memory)
        else:
            logger.warning(f"Unknown tool called: {name}")
            return _error_response(f"Unknown tool: {name}")
    except Exception as e:
        logger.exception(f"Error in tool {name}: {e}")
        return _error_response(f"Tool '{name}' failed: {e}")


async def _handle_remember(memory: ProjectMemory, arguments: dict) -> list[TextContent]:
    """Handle memory_remember tool."""
    content = arguments.get("content")
    if not content:
        return _error_response("'content' is required for memory_remember")

    tier_str = arguments.get("tier", "project")
    ttl = arguments.get("ttl")

    try:
        tier = _parse_tier(tier_str)
    except ValueError as e:
        return _error_response(str(e))

    try:
        memory_id = memory.remember(content, tier=tier, ttl=ttl)
    except Exception as e:
        logger.error(f"Failed to remember: {e}")
        return _error_response(f"Failed to store memory: {e}")

    truncated = content[:100] + "..." if len(content) > 100 else content
    logger.info(f"Remembered memory {memory_id[:8]} in tier {tier.value}")

    return [TextContent(
        type="text",
        text=f"Remembered: \"{truncated}\"\nTier: {tier.value}\nID: {memory_id[:8]}..."
    )]


async def _handle_recall(memory: ProjectMemory, arguments: dict) -> list[TextContent]:
    """Handle memory_recall tool."""
    query = arguments.get("query")
    if not query:
        return _error_response("'query' is required for memory_recall")

    k = arguments.get("k", 5)
    threshold = arguments.get("threshold", 0.3)
    tier_strs = arguments.get("tiers")

    # Validate k and threshold
    if not isinstance(k, int) or k < 1:
        return _error_response("'k' must be a positive integer")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        return _error_response("'threshold' must be a number between 0 and 1")

    # Parse tiers
    tiers = None
    if tier_strs:
        try:
            tiers = [_parse_tier(t) for t in tier_strs]
        except ValueError as e:
            return _error_response(str(e))

    try:
        results = memory.recall(query, k=k, threshold=threshold, tiers=tiers)
    except Exception as e:
        logger.error(f"Failed to recall: {e}")
        return _error_response(f"Failed to search memories: {e}")

    if not results:
        return [TextContent(
            type="text",
            text=f"No memories found for: \"{query}\""
        )]

    output_lines = [f"Found {len(results)} memories for: \"{query}\"\n"]
    for i, mem in enumerate(results, 1):
        output_lines.append(
            f"{i}. [{mem.tier.value}] (score: {mem.score:.3f})\n"
            f"   {mem.content}\n"
        )

    logger.info(f"Recalled {len(results)} memories for query: {query[:50]}")
    return [TextContent(
        type="text",
        text="\n".join(output_lines)
    )]


async def _handle_forget(memory: ProjectMemory, arguments: dict) -> list[TextContent]:
    """Handle memory_forget tool."""
    query = arguments.get("query")
    if not query:
        return _error_response("'query' is required for memory_forget")

    threshold = arguments.get("threshold", 0.8)
    tier_strs = arguments.get("tiers")

    # Validate threshold
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        return _error_response("'threshold' must be a number between 0 and 1")

    # Parse tiers
    tiers = None
    if tier_strs:
        try:
            tiers = [_parse_tier(t) for t in tier_strs]
        except ValueError as e:
            return _error_response(str(e))

    try:
        removed = memory.forget(query, threshold=threshold, tiers=tiers)
    except Exception as e:
        logger.error(f"Failed to forget: {e}")
        return _error_response(f"Failed to delete memories: {e}")

    logger.info(f"Forgot {removed} memories matching: {query[:50]}")
    return [TextContent(
        type="text",
        text=f"Forgot {removed} memories matching: \"{query}\""
    )]


async def _handle_stats(memory: ProjectMemory) -> list[TextContent]:
    """Handle memory_stats tool."""
    try:
        stats = memory.get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return _error_response(f"Failed to get memory statistics: {e}")

    lines = [
        "Memory Statistics",
        f"Path: {stats['path']}",
        f"Total: {stats['total_memories']} memories",
        "",
        "By tier:"
    ]
    for tier_name, count in stats["tiers"].items():
        lines.append(f"  {tier_name}: {count}")

    return [TextContent(
        type="text",
        text="\n".join(lines)
    )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def run():
    """Entry point for the MCP server."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
