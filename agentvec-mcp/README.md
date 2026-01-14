# agentvec-mcp

**MCP server for AgentVec memory - gives AI agents persistent memory.**

This MCP server exposes memory tools that allow AI agents (like Claude) to remember, recall, and forget information across sessions.

## Installation

```bash
pip install agentvec-mcp
```

Or for development:

```bash
cd agentvec-mcp
pip install -e .
```

## Configuration

### Claude Code

Add to your Claude Code MCP settings (`~/.claude/claude_desktop_config.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "agentvec-mcp",
      "env": {
        "AGENTVEC_MEMORY_PATH": "/path/to/your/project/.agentvec-memory"
      }
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "agentvec_mcp.server"],
      "env": {
        "AGENTVEC_MEMORY_PATH": "C:/Users/you/project/.agentvec-memory"
      }
    }
  }
}
```

## Available Tools

Once configured, Claude will have access to these tools:

### memory_remember

Store a fact or piece of information in persistent memory.

```
Arguments:
  content (required): The fact or information to remember
  tier: Memory tier - "working" (5min), "session" (1hr), "project" (30d), "user" (1yr)
  ttl: Optional custom TTL in seconds
```

Example usage by Claude:
> "I'll remember that for you."
> *Calls memory_remember with content="User prefers dark mode", tier="user"*

### memory_recall

Search for relevant memories using natural language.

```
Arguments:
  query (required): Natural language query to search memories
  k: Maximum results (default: 5)
  threshold: Minimum similarity score (default: 0.3)
  tiers: Limit to specific tiers
```

Example usage by Claude:
> "Let me check what I know about your preferences."
> *Calls memory_recall with query="user preferences"*

### memory_forget

Remove memories that match a query.

```
Arguments:
  query (required): Query to match memories for deletion
  threshold: Similarity threshold (default: 0.8)
  tiers: Limit to specific tiers
```

### memory_stats

Get statistics about stored memories.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTVEC_MEMORY_PATH` | Path to memory database | `./.agentvec-memory` |

## Memory Tiers

| Tier | TTL | Use Case |
|------|-----|----------|
| `working` | 5 minutes | Current task context |
| `session` | 1 hour | Single work session |
| `project` | 30 days | Project knowledge |
| `user` | 1 year | User preferences |

## Example Conversation

After configuring the MCP server:

**User:** Remember that I prefer tabs over spaces.

**Claude:** *Calls memory_remember(content="User prefers tabs over spaces for indentation", tier="user")*

I've remembered your preference for tabs over spaces.

---

*Next session*

**User:** Format this code file for me.

**Claude:** *Calls memory_recall(query="code formatting preferences")*

I see you prefer tabs over spaces. I'll format the file using tabs for indentation.

## How It Works

1. **Storage**: Memories are embedded using sentence-transformers and stored in AgentVec
2. **Recall**: Queries are semantically matched against stored memories
3. **TTL**: Memories automatically expire based on their tier
4. **Persistence**: Data persists across sessions in the configured path

## License

MIT
