# agentvec-memory

**Persistent tiered memory for AI agents.**

Give your AI agents long-term memory that persists across sessions.

## The Problem

AI agents forget everything between sessions. Every conversation starts from scratch.

```python
# Session 1
agent.chat("My API key is stored in .env")
# Agent knows this

# Session 2 (next day)
agent.chat("Where's my API key?")
# Agent has no idea
```

## The Solution

```python
from agentvec_memory import ProjectMemory, MemoryTier

memory = ProjectMemory("./agent-memory")

# Remember facts with appropriate lifetimes
memory.remember("API key is in .env file", tier=MemoryTier.PROJECT)
memory.remember("User prefers dark mode", tier=MemoryTier.USER)
memory.remember("Currently working on auth feature", tier=MemoryTier.SESSION)

# Later... recall relevant memories
results = memory.recall("where is the API key")
# Returns: "API key is in .env file" (score: 0.847)
```

## Installation

```bash
pip install agentvec-memory
```

Or for development:

```bash
cd agentvec-memory
pip install -e .
```

## Quick Start

### Python API

```python
from agentvec_memory import ProjectMemory, MemoryTier

# Initialize memory
memory = ProjectMemory("./memory.db")

# Remember facts
memory.remember("User prefers dark mode", tier=MemoryTier.USER)
memory.remember("Project uses PostgreSQL", tier=MemoryTier.PROJECT)
memory.remember("Working on login feature", tier=MemoryTier.SESSION)
memory.remember("Current file: auth.py", tier=MemoryTier.WORKING)

# Recall relevant memories
results = memory.recall("database configuration")
for mem in results:
    print(f"{mem.content} (score: {mem.score:.3f}, tier: {mem.tier.value})")

# Forget memories
memory.forget("dark mode")  # Removes by semantic similarity

# Cleanup expired memories
memory.cleanup_expired()
```

### CLI

```bash
# Remember facts
agentvec-memory remember "API key is in .env"
agentvec-memory remember "User prefers vim keybindings" --tier user

# Recall memories
agentvec-memory recall "API configuration"
agentvec-memory recall "user preferences" --tier user

# Forget memories
agentvec-memory forget "API key" --dry-run  # Preview
agentvec-memory forget "API key"            # Delete

# Statistics
agentvec-memory stats

# Interactive mode
agentvec-memory interactive
```

## Memory Tiers

| Tier | Default TTL | Use Case |
|------|-------------|----------|
| `WORKING` | 5 minutes | Current task context, temporary notes |
| `SESSION` | 1 hour | Single work session, current feature |
| `PROJECT` | 30 days | Project knowledge, configurations |
| `USER` | 1 year | User preferences, long-term facts |

## Commands

| Command | Description |
|---------|-------------|
| `remember <content>` | Store a memory |
| `recall <query>` | Search memories |
| `forget <query>` | Remove similar memories |
| `stats` | Show memory statistics |
| `cleanup` | Remove expired memories |
| `clear` | Clear all memories |
| `interactive` | Interactive mode |

## Options

### Remember Options

| Option | Description |
|--------|-------------|
| `-m, --memory PATH` | Memory database path |
| `-t, --tier` | Memory tier (working/session/project/user) |
| `--ttl SECONDS` | Custom time-to-live |

### Recall Options

| Option | Description |
|--------|-------------|
| `-m, --memory PATH` | Memory database path |
| `-k, --results NUM` | Number of results (default: 5) |
| `-t, --threshold` | Minimum similarity (default: 0.3) |
| `--tier` | Limit to specific tier(s) |
| `--include-expired` | Include expired memories |

### Forget Options

| Option | Description |
|--------|-------------|
| `-m, --memory PATH` | Memory database path |
| `-t, --threshold` | Similarity threshold (default: 0.8) |
| `--tier` | Limit to specific tier(s) |
| `--dry-run` | Preview without deleting |

## How It Works

1. **Storage**: Memories are embedded using sentence-transformers and stored in AgentVec
2. **Recall**: Queries are embedded and matched against stored memories using cosine similarity
3. **TTL**: Each memory has an expiration time based on its tier
4. **Cleanup**: Expired memories can be automatically removed

## Integration Examples

### With LangChain

```python
from agentvec_memory import ProjectMemory, MemoryTier

memory = ProjectMemory("./agent-memory")

def agent_remember(fact: str, tier: str = "project"):
    """Tool for agent to store memories."""
    memory.remember(fact, tier=MemoryTier(tier))
    return f"Remembered: {fact}"

def agent_recall(query: str, k: int = 3):
    """Tool for agent to recall memories."""
    results = memory.recall(query, k=k)
    return "\n".join(f"- {m.content}" for m in results)
```

### With Claude/OpenAI

```python
# Before each conversation, inject relevant memories
context = memory.recall("current task", k=5)
system_prompt = f"""You have access to these memories:
{chr(10).join(f'- {m.content}' for m in context)}
"""
```

## Tips

- **Be specific**: "User prefers Vim keybindings" is better than "user settings"
- **Use appropriate tiers**: Don't store temporary notes in USER tier
- **Regular cleanup**: Run `cleanup` periodically to remove expired memories
- **Threshold tuning**: Lower recall threshold (0.2-0.3) for broader results

## Limitations

- Works best with English text
- Requires sentence-transformers (downloads ~90MB model on first use)
- Memory embeddings are not updated if content meaning changes

## License

MIT
