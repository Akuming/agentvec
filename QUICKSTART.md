# AgentVec Quick Start Guide

Get AI agent memory working in 5 minutes.

## Installation

### Option 1: Lightweight (Recommended)

Install with fastembed - only ~100MB total:

```bash
pip install agentvec-memory
```

**First run downloads a small ONNX model (~50MB). Total install: ~100MB.**

### Option 2: With GPU Support

Install with sentence-transformers for GPU acceleration (~3GB):

```bash
pip install agentvec-memory[gpu]
```

### Option 3: With OpenAI Embeddings

Use OpenAI's embedding API (requires API key):

```bash
pip install agentvec-memory[openai]
```

### Option 4: Core Only

Just the vector database, bring your own embeddings:

```bash
pip install agentvec
```

### Option 5: With MCP (for Claude integration)

For Claude Code or Claude Desktop integration:

```bash
pip install agentvec-mcp
```

## Package Overview

```
┌─────────────────────────────────────────────────────────┐
│                    agentvec-mcp                         │
│            (MCP server for Claude integration)          │
└─────────────────────────┬───────────────────────────────┘
                          │ depends on
┌─────────────────────────▼───────────────────────────────┐
│                   agentvec-memory                       │
│         (Tiered memory with embeddings built-in)        │
└─────────────────────────┬───────────────────────────────┘
                          │ depends on
┌─────────────────────────▼───────────────────────────────┐
│                      agentvec                           │
│              (Core vector database - Rust)              │
└─────────────────────────────────────────────────────────┘
```

| Package | Use Case | Manages Embeddings? |
|---------|----------|---------------------|
| `agentvec` | Low-level vector storage | No (BYOE) |
| `agentvec-memory` | AI agent memory | Yes (automatic) |
| `agentvec-mcp` | Claude integration | Yes (via agentvec-memory) |

## Quick Start: Agent Memory

```python
from agentvec_memory import ProjectMemory, MemoryTier

# Initialize (creates DB + loads embedding model)
memory = ProjectMemory("./agent-memory")

# Remember facts with appropriate lifetimes
memory.remember("User prefers dark mode", tier=MemoryTier.USER)      # 1 year
memory.remember("Project uses FastAPI", tier=MemoryTier.PROJECT)    # 30 days
memory.remember("Working on auth bug", tier=MemoryTier.SESSION)     # 1 hour
memory.remember("Current file: auth.py", tier=MemoryTier.WORKING)   # 5 minutes

# Recall relevant memories
results = memory.recall("What framework does this project use?")
for mem in results:
    print(f"[{mem.tier.value}] {mem.content} (score: {mem.score:.2f})")

# Forget outdated information
memory.forget("auth bug", threshold=0.8)

# Cleanup expired memories
memory.cleanup_expired()
```

## Quick Start: Core Vector DB

```python
import agentvec

# Open database
db = agentvec.AgentVec("./vectors.db")

# Create collection (you provide embeddings)
collection = db.collection("memories", dim=384, metric="cosine")

# Add vector (bring your own embedding)
embedding = get_embedding("Hello world")  # Your embedding function
memory_id = collection.add(
    vector=embedding,
    metadata={"text": "Hello world", "source": "user"},
    ttl=3600  # Optional: expires in 1 hour
)

# Search
results = collection.search(query_embedding, k=10)
for r in results:
    print(f"{r.id}: {r.score:.3f} - {r.metadata}")
```

## Quick Start: Claude Integration

### 1. Install

```bash
pip install agentvec-mcp
```

### 2. Configure Claude Code

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "memory": {
      "command": "agentvec-mcp",
      "env": {
        "AGENTVEC_MEMORY_PATH": "./.agentvec-memory"
      }
    }
  }
}
```

### 3. Use in Conversation

Claude now has access to memory tools:

> **You:** Remember that I prefer tabs over spaces.
>
> **Claude:** I've stored that preference in your long-term memory.
>
> *[Later session]*
>
> **You:** Format this code file.
>
> **Claude:** I'll format using tabs, based on your preference.

## Memory Tiers

| Tier | Default TTL | Use Case |
|------|-------------|----------|
| `WORKING` | 5 minutes | Current task context |
| `SESSION` | 1 hour | Single work session |
| `PROJECT` | 30 days | Project knowledge |
| `USER` | 1 year | User preferences |

## Common Patterns

### Integrate with LangChain

```python
from agentvec_memory import ProjectMemory, MemoryTier

memory = ProjectMemory("./agent-memory")

# Create tools for your agent
def remember_tool(fact: str, tier: str = "project") -> str:
    """Store a fact in memory."""
    memory.remember(fact, tier=MemoryTier(tier))
    return f"Remembered: {fact}"

def recall_tool(query: str) -> str:
    """Retrieve relevant memories."""
    results = memory.recall(query, k=5)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"- {m.content}" for m in results)
```

### Context Injection for LLM Calls

```python
def get_context_for_query(query: str) -> str:
    """Get relevant context to inject into system prompt."""
    memories = memory.recall(query, k=5, threshold=0.4)
    if not memories:
        return ""

    context_lines = ["Relevant context from memory:"]
    for mem in memories:
        context_lines.append(f"- [{mem.tier.value}] {mem.content}")

    return "\n".join(context_lines)

# Use in your LLM call
system_prompt = f"""You are a helpful assistant.

{get_context_for_query(user_message)}
"""
```

## Choosing an Embedder

| Embedder | Install Size | First Run | GPU | Best For |
|----------|-------------|-----------|-----|----------|
| `fastembed` (default) | ~100MB | ~30s | No | Most users |
| `sentence-transformers` | ~3GB | ~2min | Yes | GPU acceleration |
| `openai` | ~10MB | Instant | N/A | Production, high quality |
| Custom | 0 | 0 | You decide | Full control |

```python
# Default (fastembed - lightweight)
memory = ProjectMemory("./memory")

# Explicit choice
memory = ProjectMemory("./memory", embedder="fastembed")
memory = ProjectMemory("./memory", embedder="sentence-transformers")
memory = ProjectMemory("./memory", embedder="openai")

# Custom function
def my_embedder(texts):
    return [[0.1] * 384 for _ in texts]
memory = ProjectMemory("./memory", embedder=my_embedder, dimension=384)
```

## Troubleshooting

### "ImportError: No embedding provider found"

Install an embedding backend:

```bash
pip install fastembed              # Lightweight, recommended
# OR
pip install sentence-transformers  # Full-featured, GPU support
```

### "DimensionMismatch: expected 384, got 512"

You're using a different embedding model than what the collection was created with. Either:
1. Use the same model consistently
2. Create a new collection with the correct dimensions

### Memory not persisting

Make sure you're using the same path:

```python
# Always use the same path
memory = ProjectMemory("./agent-memory")  # Consistent path
```

### MCP server not starting

Check the path exists and is writable:

```bash
# Verify path
echo $AGENTVEC_MEMORY_PATH

# Test manually
python -m agentvec_mcp.server
```

## Next Steps

- Run `python examples/end_to_end_memory.py` for a full demo
- Read the [main README](./README.md) for API reference
- Check [agentvec-memory](./agentvec-memory/README.md) for CLI usage
- See [agentvec-mcp](./agentvec-mcp/README.md) for Claude configuration
