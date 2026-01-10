# AgentVec Examples

This directory contains example code demonstrating how to use AgentVec.

## Rust Examples

### Prerequisites

```bash
cd /path/to/agentvec
cargo build
```

### Running Examples

```bash
# Basic usage - CRUD operations
cargo run --example basic_usage

# AI agent memory patterns
cargo run --example agent_memory

# Advanced filtering
cargo run --example filtering
```

### Example Descriptions

| Example | Description |
|---------|-------------|
| `basic_usage.rs` | Core operations: create, read, update, delete, search |
| `agent_memory.rs` | Working, episodic, and semantic memory patterns for AI agents |
| `filtering.rs` | All filter operators: `$eq`, `$ne`, `$gt`, `$lt`, `$in`, `$nin` |

---

## Python Examples

### Prerequisites

```bash
# Install from PyPI
pip install agentvec

# Or build from source
cd agentvec-python
maturin develop
```

### Running Examples

```bash
# Basic usage
python examples/basic_usage.py

# AI agent memory patterns
python examples/agent_memory.py
```

### Example Descriptions

| Example | Description |
|---------|-------------|
| `basic_usage.py` | Core operations in Python |
| `agent_memory.py` | Memory patterns for AI agents in Python |

---

## Common Patterns

### Embedding Integration

AgentVec stores vectors - you need an embedding model to convert text to vectors:

```python
from openai import OpenAI
import agentvec

client = OpenAI()
db = agentvec.AgentVec("./memory")
memories = db.collection("memories", dim=1536, metric="cosine")

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Store
memories.add(
    vector=embed("User asked about the weather"),
    metadata={"type": "conversation", "content": "Weather inquiry"}
)

# Search
results = memories.search(
    vector=embed("What's the temperature?"),
    k=5
)
```

### Batch Operations

For better performance when adding many vectors:

```python
# Python
ids = memories.add_batch(
    vectors=[embed(text) for text in texts],
    metadatas=[{"content": t} for t in texts]
)
```

```rust
// Rust - add within a single flush window
for (text, embedding) in texts.iter().zip(embeddings.iter()) {
    memories.add(embedding, json!({"content": text}), None, None)?;
}
memories.sync()?;
```

### Memory Decay Strategy

Implement tiered memory with different TTLs:

```python
# Working memory - very short term
working.add(vec, meta, ttl=300)       # 5 minutes

# Short-term memory
short_term.add(vec, meta, ttl=3600)   # 1 hour

# Episodic memory
episodic.add(vec, meta, ttl=86400*7)  # 7 days

# Long-term/semantic - permanent
semantic.add(vec, meta, ttl=None)     # No expiry
```

### Graceful Degradation

When context window is limited, retrieve hierarchically:

```python
def get_context(query_vec, max_tokens=4000):
    context = []
    tokens_used = 0

    # 1. Recent working memory (highest priority)
    for r in working.search(query_vec, k=3):
        if tokens_used + len(r.metadata["content"]) < max_tokens:
            context.append(r.metadata["content"])
            tokens_used += len(r.metadata["content"])

    # 2. Relevant episodic memories
    for r in episodic.search(query_vec, k=5):
        if tokens_used + len(r.metadata["summary"]) < max_tokens:
            context.append(r.metadata["summary"])
            tokens_used += len(r.metadata["summary"])

    # 3. Semantic knowledge
    for r in semantic.search(query_vec, k=3):
        if tokens_used + len(r.metadata["content"]) < max_tokens:
            context.append(r.metadata["content"])
            tokens_used += len(r.metadata["content"])

    return context
```
