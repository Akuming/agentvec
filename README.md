# AgentVec

**The SQLite for Mutable Agent Memory**

AgentVec is a lightweight, serverless, embedded vector database written in pure Rust. It's designed for AI agents that need to remember, forget, and update knowledge in real-time.

```python
import agentvec

db = agentvec.AgentVec("./agent_memory")
memories = db.collection("episodic", dim=384, metric="cosine")

# Store a memory (with 1-hour TTL)
memories.add(embedding, {"event": "user said hello"}, ttl=3600)

# Recall similar memories
results = memories.search(query_embedding, k=10)
```

---

## Why AgentVec?

| Feature | AgentVec | LanceDB | ChromaDB | FAISS |
|---------|----------|---------|----------|-------|
| Architecture | **Row-Oriented** | Columnar | Memory-First | Index-Only |
| Primary Use | **Agent Memory** | RAG / Analytics | Static Search | Batch Search |
| Dependencies | **Zero (Pure Rust)** | Arrow/C++ | Python/C++ | C++/BLAS |
| In-Place Updates | **ACID** | Copy-on-Write | Variable | Rebuild |
| TTL / Expiry | **Native** | No | No | No |
| Incremental Index | **Yes** | Rebuild | Variable | Rebuild |

### Built For

- **AI Agents** - Conversational memory, working memory, long-term knowledge
- **Local-First Apps** - No server, no cloud, just a directory
- **Mutable Workloads** - Frequent upserts, deletes, and TTL expiry
- **Embedded Systems** - Single binary, low memory footprint
- **Rust Applications** - Native integration, no FFI overhead

### Not Built For

- **Massive Scale Analytics** - Use LanceDB, ClickHouse, or DuckDB
- **Distributed Systems** - No sharding, no replication
- **Billion-Vector Search** - Optimized for 1K-1M vectors
- **Static Datasets** - If you never update, use FAISS or usearch
- **Multi-Process Access** - Single-process, multi-threaded only

---

## Installation

### Python

```bash
pip install agentvec
```

### Rust

```toml
[dependencies]
agentvec = "0.1"
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/user/agentvec
cd agentvec

# Build Rust library
cargo build --release

# Build Python wheel
cd agentvec-python
maturin build --release
pip install target/wheels/agentvec-*.whl
```

---

## Quick Start

### Python

```python
import agentvec

# Open or create database
db = agentvec.AgentVec("./my_agent.avdb")

# Create collections for different memory types
episodic = db.collection("episodic", dim=384, metric="cosine")   # Conversations
semantic = db.collection("semantic", dim=1536, metric="cosine")  # Knowledge
working = db.collection("working", dim=384, metric="cosine")     # Scratchpad

# Add a memory
id = episodic.add(
    vector=[0.1, 0.2, ...],  # 384 dimensions
    metadata={"type": "conversation", "user": "alice"},
    ttl=86400  # Expires in 24 hours
)

# Upsert (insert or update) - idempotent, safe to call multiple times
episodic.upsert(
    id="conv_turn_42",
    vector=[0.1, 0.2, ...],
    metadata={"summary": "Discussed project timeline"}
)

# Batch insert (10-100x faster)
ids = episodic.add_batch(
    vectors=[[0.1, ...], [0.2, ...], ...],
    metadatas=[{"msg": "hello"}, {"msg": "world"}, ...]
)

# Search with metadata filter
results = episodic.search(
    vector=query_embedding,
    k=10,
    where_={"user": "alice"}
)

for r in results:
    print(f"{r.id}: {r.score:.3f} - {r.metadata}")

# Cleanup expired memories
stats = episodic.compact()
print(f"Removed {stats.expired_removed} expired records")

# Ensure durability
db.sync()
```

### Rust

```rust
use agentvec::{AgentVec, Metric, Filter};
use serde_json::json;

// Open or create database
let db = AgentVec::open("./my_agent.avdb")?;

// Create a collection
let memories = db.collection("memories", 384, Metric::Cosine)?;

// Add a vector
let id = memories.add(
    &embedding,
    json!({"type": "conversation", "user": "alice"}),
    None,        // Auto-generate ID
    Some(86400), // TTL: 24 hours
)?;

// Search
let results = memories.search(&query, 10, None)?;

// Search with filter
let filter = Filter::new().eq("user", "alice");
let results = memories.search(&query, 10, Some(filter))?;

// Upsert
memories.upsert(
    "my_custom_id",
    &embedding,
    json!({"updated": true}),
    None,
)?;
```

---

## Core Concepts

### Collections

Collections are isolated namespaces with independent dimensions and storage:

```python
db = agentvec.AgentVec("./memory")

# Each collection can have different dimensions
episodic = db.collection("episodic", dim=384)    # Small model
semantic = db.collection("semantic", dim=1536)   # Large model
images = db.collection("images", dim=512, metric="l2")  # CLIP embeddings
```

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `cosine` | Normalized dot product | OpenAI, Cohere, most text embeddings |
| `dot` | Raw dot product | Some sentence transformers |
| `l2` | Euclidean distance | Image embeddings (CLIP) |

**Note:** Cosine metric normalizes vectors on ingest for faster search.

### TTL (Time-To-Live)

Memory decay is a first-class feature:

```python
# Short-term memory (1 hour)
memories.add(vec, {"type": "working"}, ttl=3600)

# Medium-term (24 hours)
memories.add(vec, {"type": "episodic"}, ttl=86400)

# Permanent (no TTL)
memories.add(vec, {"type": "semantic"}, ttl=None)

# Cleanup expired records
memories.compact()
```

### HNSW Index

AgentVec automatically builds an HNSW index for fast approximate search:

- **Auto-enabled** when collection exceeds threshold (default: 1000 vectors)
- **Parallel construction** using all CPU cores
- **Incremental updates** - new vectors added without full rebuild
- **Persistent** - saved to disk, loaded on restart

```python
from agentvec import HnswConfig, CollectionConfig

# Custom HNSW parameters
config = CollectionConfig.with_hnsw(
    name="memories",
    dimensions=384,
    metric="cosine",
    hnsw=HnswConfig(
        m=16,                 # Connections per node
        ef_construction=200,  # Build-time beam width
        ef_search=50,         # Search-time beam width
    )
)
```

---

## Performance

Benchmarks on AMD Ryzen / Intel i7 (single-threaded search):

| Vectors | Dimensions | Index | Search Time | Recall@10 |
|---------|------------|-------|-------------|-----------|
| 10,000 | 384 | Exact | ~5ms | 100% |
| 10,000 | 384 | HNSW | ~1ms | 98% |
| 100,000 | 384 | HNSW | ~5ms | 95% |
| 100,000 | 384 | Exact | ~40ms | 100% |
| 500,000 | 384 | HNSW | ~10ms | 93% |

**Index Build Performance:**

| Vectors | Build Time | Rate |
|---------|------------|------|
| 50,000 | ~15s | 3,300 vec/s |
| 100,000 | ~35s | 2,800 vec/s |

**Incremental Insertion:**

| Operation | Time | Rate |
|-----------|------|------|
| Add 10K to 50K index | ~19s | 520 vec/s |
| Full rebuild 60K | ~20s | 3,000 vec/s |

Incremental insertion avoids rebuilding the entire index when adding new vectors.

---

## Architecture

```
my_agent.avdb/
├── meta.redb              # Global metadata (ACID key-value store)
└── collections/
    ├── episodic/
    │   ├── vectors.bin    # Memory-mapped vector storage
    │   └── hnsw.index     # HNSW graph (if enabled)
    └── semantic/
        ├── vectors.bin
        └── hnsw.index
```

### Design Principles

1. **Hybrid Storage** - Vectors in mmap'd files (zero-copy), metadata in ACID store
2. **Row-Oriented** - Optimized for point lookups and updates, not analytics
3. **Crash-Safe** - Write-ahead reservation protocol for ACID guarantees
4. **Cache-Friendly** - Sequential vector layout maximizes CPU cache utilization

### Concurrency Model

**Single-Writer, Multiple-Reader (SWMR)**

| Operation | Concurrency |
|-----------|-------------|
| `search()` | Parallel reads allowed |
| `get()` | Parallel reads allowed |
| `add()` | Exclusive write lock |
| `upsert()` | Exclusive write lock |
| `delete()` | Exclusive write lock |

For write-heavy workloads, use `add_batch()` to amortize lock overhead.

---

## API Reference

### Database

```python
db = agentvec.AgentVec(path)      # Open or create database
db.collection(name, dim, metric)  # Get or create collection
db.get_collection(name)           # Get existing collection
db.collections()                  # List collection names
db.drop_collection(name)          # Delete collection
db.sync()                         # Flush to disk
db.recovery_stats()               # Crash recovery info
```

### Collection

```python
col.add(vector, metadata, id=None, ttl=None)  # Insert
col.upsert(id, vector, metadata, ttl=None)    # Insert or update
col.add_batch(vectors, metadatas, ...)        # Bulk insert
col.search(vector, k, where_=None)            # Find nearest
col.get(id)                                   # Get by ID
col.delete(id)                                # Remove
col.compact()                                 # Cleanup expired
col.preload()                                 # Warm cache
col.sync()                                    # Flush to disk
len(col)                                      # Record count
col.dimensions                                # Vector dimensions
col.metric                                    # Distance metric
col.vectors_size_bytes                        # Storage size
```

### SearchResult

```python
result.id        # Record ID
result.score     # Similarity score (higher = more similar for cosine/dot)
result.metadata  # JSON metadata dict
```

---

## Filtering

Filter search results by metadata fields:

```python
# Equality
results = col.search(vec, k=10, where_={"user": "alice"})

# Multiple conditions (AND)
results = col.search(vec, k=10, where_={
    "user": "alice",
    "type": "conversation"
})

# Comparison operators
results = col.search(vec, k=10, where_={
    "score": {"$gt": 0.8},           # Greater than
    "count": {"$lte": 100},          # Less than or equal
})

# Set operators
results = col.search(vec, k=10, where_={
    "status": {"$in": ["active", "pending"]},   # In set
    "tag": {"$nin": ["spam", "deleted"]},       # Not in set
})

# Not equal
results = col.search(vec, k=10, where_={
    "type": {"$ne": "system"}
})
```

**Supported Operators:**
| Operator | Description | Example |
|----------|-------------|---------|
| (none) | Equality | `{"user": "alice"}` |
| `$eq` | Explicit equality | `{"user": {"$eq": "alice"}}` |
| `$ne` | Not equal | `{"type": {"$ne": "system"}}` |
| `$gt` | Greater than | `{"score": {"$gt": 0.5}}` |
| `$gte` | Greater than or equal | `{"count": {"$gte": 10}}` |
| `$lt` | Less than | `{"age": {"$lt": 30}}` |
| `$lte` | Less than or equal | `{"priority": {"$lte": 5}}` |
| `$in` | In set | `{"status": {"$in": ["a", "b"]}}` |
| `$nin` | Not in set | `{"tag": {"$nin": ["x", "y"]}}` |

**Notes:**
- Multiple conditions use AND semantics
- Post-filter with 10x over-fetch heuristic
- For highly selective filters, increase `k`

---

## Memory Management

AgentVec uses memory-mapped I/O:

```python
# Check storage size
print(f"Vectors: {col.vectors_size_bytes / 1024 / 1024:.1f} MB")

# Preload into RAM before latency-sensitive operations
col.preload()
```

**Behavior by Size:**

| vectors.bin | Available RAM | Behavior |
|-------------|---------------|----------|
| < RAM | Plenty | Fully cached, optimal |
| < RAM | Limited | OS pages out gracefully |
| > RAM | Any | Working set cached, rest on disk |

---

## Limitations

### What AgentVec Does NOT Do

1. **Distributed Search** - Single machine only
2. **Multi-Process Access** - One process at a time
3. **OR Filters** - Only AND semantics (no OR conditions)
4. **Billion-Scale** - Optimized for up to ~1M vectors
5. **Real-Time Sync** - No built-in replication
6. **SQL Queries** - Vector search only

### When to Use Something Else

| Need | Alternative |
|------|-------------|
| Distributed vector search | Qdrant, Milvus, Pinecone |
| Analytical queries on vectors | LanceDB, DuckDB |
| Static billion-scale index | FAISS, usearch |
| Full-text + vector hybrid | Elasticsearch, Vespa |

---

## Roadmap

- [x] Core vector storage with ACID metadata
- [x] Python bindings (PyO3 + maturin)
- [x] Mobile bindings (UniFFI for iOS/Android)
- [x] HNSW approximate search
- [x] Parallel index construction
- [x] Incremental index updates
- [x] TTL / memory decay
- [x] Metadata filtering
- [x] Filter operators (`$gt`, `$lt`, `$in`, `$nin`, `$ne`)
- [x] Product quantization
- [x] Export/import for backup
- [ ] JavaScript/WASM bindings
- [ ] CLI tool
- [ ] HTTP server
- [ ] Secondary indexes on metadata
- [ ] LangChain/LlamaIndex integration

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Build Python wheel
cd agentvec-python && maturin develop
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## Acknowledgments

AgentVec draws inspiration from:

- [hnswlib](https://github.com/nmslib/hnswlib) - HNSW algorithm reference
- [redb](https://github.com/cberner/redb) - Pure Rust ACID storage
- [LanceDB](https://github.com/lancedb/lancedb) - Embedded vector DB concepts
- [SQLite](https://sqlite.org) - The gold standard for embedded databases
