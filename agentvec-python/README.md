# AgentVec

**The SQLite for mutable agent memory.** A lightweight, serverless, embedded vector database written in pure Rust — built for AI agents that need to remember, forget, and update knowledge in real time.

No server. No cloud. No external dependencies. Just point it at a directory.

```bash
pip install agentvec
```

```python
import agentvec

db = agentvec.AgentVec("./agent_memory.avdb")
memories = db.collection("episodic", dim=384, metric="cosine")

# Store a memory that expires in one hour
memories.add([0.1, 0.2, ...], {"event": "user said hello"}, ttl=3600)

# Recall the most similar memories
for r in memories.search(query_embedding, k=10):
    print(r.id, r.score, r.metadata)
```

---

## Why AgentVec?

Most vector databases are built for **search over static datasets** — you index once and query forever. Agent memory is the opposite workload: it changes constantly. Agents add new observations every turn, update what they already know, and let stale context decay away.

AgentVec is designed for that **mutable, transactional, row-oriented** pattern instead of bulk analytics.

| | AgentVec | LanceDB | ChromaDB | FAISS |
|---|---|---|---|---|
| Primary use | **Agent memory** | RAG / analytics | Static search | Batch search |
| Architecture | **Row-oriented** | Columnar | Memory-first | Index-only |
| Dependencies | **Zero (pure Rust)** | Arrow / C++ | Python / C++ | C++ / BLAS |
| In-place updates | **ACID** | Copy-on-write | Variable | Rebuild |
| TTL / expiry | **Native** | — | — | — |
| Incremental index | **Yes** | Rebuild | Variable | Rebuild |

### Built for
- **AI agents** — conversational memory, working memory, long-term knowledge
- **Local-first apps** — no server, no network, just a directory on disk
- **Mutable workloads** — frequent upserts, deletes, and TTL-based expiry
- **Embedded use** — single library, low memory footprint, instant startup

### Not built for
- Billion-vector or distributed search (use Qdrant, Milvus, or Pinecone)
- Heavy analytical queries over vectors (use LanceDB or DuckDB)
- Static datasets you never update (use FAISS or usearch)
- Multi-process concurrent access (AgentVec is single-process, multi-threaded)

---

## Features

- **Memory decay (TTL)** — give any record a time-to-live; expired memories are skipped on read and reclaimed on `compact()`.
- **In-place upserts** — `upsert(id, ...)` is idempotent and ACID-safe; update a memory without rebuilding anything.
- **HNSW approximate search** — auto-enabled past a threshold, built in parallel across all cores, with *incremental* inserts (no full rebuilds) and on-disk persistence.
- **Metadata filtering** — filter by metadata with `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.
- **Multiple distance metrics** — `cosine`, `dot`, and `l2` (Euclidean).
- **Crash-safe storage** — vectors live in memory-mapped files for zero-copy reads; metadata lives in an ACID key-value store with checksums and a recovery path.
- **Product quantization** — optional compression for larger collections.
- **Export / import** — snapshot a collection to a file for backup or transfer.

---

## Quick start

```python
import agentvec

db = agentvec.AgentVec("./my_agent.avdb")

# Different memory types can use different embedding dimensions
episodic = db.collection("episodic", dim=384,  metric="cosine")   # conversations
semantic = db.collection("semantic", dim=1536, metric="cosine")   # knowledge
working  = db.collection("working",  dim=384,  metric="cosine")   # scratchpad

# Add a memory (auto-generated id, expires in 24h)
mem_id = episodic.add(vector, {"user": "alice", "type": "chat"}, ttl=86400)

# Update it later — same id, no rebuild
episodic.upsert("conv_turn_42", vector, {"summary": "discussed timeline"})

# Bulk insert is 10-100x faster than one-at-a-time
episodic.add_batch(vectors=[...], metadatas=[...])

# Search, filtered by metadata
results = episodic.search(query_vector, k=10, where_={"user": "alice"})

# Reclaim expired memories and flush to disk
episodic.compact()
db.sync()
```

---

## Companion packages

AgentVec is the storage engine. Higher-level libraries build on top of it:

- **`agentvec-memory`** — tiered agent memory (working / session / project / user) with pluggable embedders, so you store text and let it handle vectors and TTLs.
- **`agentvec-mcp`** — a Model Context Protocol server that exposes AgentVec memory to MCP-compatible AI tools.

---

## Documentation & source

- **Repository:** https://github.com/Akuming/agentvec
- **Issues:** https://github.com/Akuming/agentvec/issues
- Full API reference, architecture notes, and benchmarks are in the repository README.

## License

Dual-licensed under either of [MIT](https://github.com/Akuming/agentvec/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/Akuming/agentvec/blob/main/LICENSE-APACHE), at your option.
