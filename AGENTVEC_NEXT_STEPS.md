# AgentVec Next Steps

This document outlines the planned development roadmap for AgentVec.

---

## Priority 1: JavaScript/WASM Bindings

**Impact:** Very High | **Effort:** Medium

JavaScript dominates AI agent development (LangChain.js, Vercel AI SDK, browser-based agents). This unlocks massive reach.

### Structure
```
agentvec-js/
├── Cargo.toml          # wasm-bindgen dependencies
├── src/lib.rs          # WASM bindings
├── package.json        # npm package config
├── index.d.ts          # TypeScript definitions
└── README.md
```

### Key Features
- Full API parity with Python bindings
- TypeScript-first with complete type definitions
- Works in Node.js and browsers
- Zero-copy where possible via SharedArrayBuffer

### Target API
```typescript
import { AgentVec, Metric } from 'agentvec';

const db = await AgentVec.open('./agent_memory');
const memories = await db.collection('episodic', 384, Metric.Cosine);

await memories.add(embedding, { type: 'conversation' }, { ttl: 3600 });
const results = await memories.search(queryEmbedding, 10);
```

---

## Priority 2: CLI Tool

**Impact:** High | **Effort:** Low

Essential for debugging, operations, and developer experience. SQLite's CLI is a major reason for its adoption.

### Structure
```
agentvec-cli/
├── Cargo.toml
└── src/main.rs
```

### Commands
```bash
# Database info
agentvec info ./my.avdb
agentvec collections ./my.avdb

# Collection operations
agentvec stats ./my.avdb memories
agentvec search ./my.avdb memories --vector "0.1,0.2,..." --k 10
agentvec get ./my.avdb memories --id "record_123"

# Maintenance
agentvec compact ./my.avdb memories
agentvec export ./my.avdb memories -o backup.jsonl
agentvec import ./my.avdb memories -i backup.jsonl

# Debugging
agentvec validate ./my.avdb              # Check integrity
agentvec hnsw-stats ./my.avdb memories   # Index statistics
```

### Dependencies
- `clap` for argument parsing
- `colored` / `termcolor` for output formatting
- `indicatif` for progress bars

---

## Priority 3: HTTP Server (Optional Crate)

**Impact:** Medium | **Effort:** Medium

Language-agnostic access for polyglot teams and microservice architectures.

### Structure
```
agentvec-server/
├── Cargo.toml
└── src/
    ├── main.rs
    ├── routes.rs
    └── handlers.rs
```

### API Endpoints
```
GET    /health                           # Health check
GET    /collections                      # List collections
POST   /collections                      # Create collection
DELETE /collections/{name}               # Drop collection

POST   /collections/{name}/add           # Add vector
POST   /collections/{name}/upsert        # Upsert vector
POST   /collections/{name}/search        # Search vectors
GET    /collections/{name}/records/{id}  # Get by ID
DELETE /collections/{name}/records/{id}  # Delete record

POST   /collections/{name}/compact       # Cleanup expired
GET    /collections/{name}/stats         # Collection stats
```

### Example Request
```bash
curl -X POST http://localhost:8080/collections/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "k": 10,
    "filter": {"user": "alice"}
  }'
```

### Dependencies
- `axum` or `actix-web` for HTTP framework
- `tokio` for async runtime
- `serde_json` for JSON handling

---

## Priority 4: LangChain/LlamaIndex Integration

**Impact:** High | **Effort:** Low

Direct integration with the most popular AI agent frameworks.

### LangChain Integration
```python
# langchain-agentvec package
from langchain_agentvec import AgentVecVectorStore

vectorstore = AgentVecVectorStore(
    path="./agent_memory",
    collection_name="documents",
    embedding=OpenAIEmbeddings(),
)

# Standard LangChain interface
vectorstore.add_documents(documents)
results = vectorstore.similarity_search(query, k=10)

# Use as retriever
retriever = vectorstore.as_retriever()
```

### LlamaIndex Integration
```python
# llama-index-vector-stores-agentvec package
from llama_index.vector_stores.agentvec import AgentVecVectorStore

vector_store = AgentVecVectorStore(path="./memory")
index = VectorStoreIndex.from_vector_store(vector_store)
```

---

## Priority 5: Embedding Integration (Optional)

**Impact:** Medium | **Effort:** Medium

Reduce boilerplate by integrating embedding models directly.

### API Design
```python
# Option 1: Built-in embedding
memories.add_text(
    "User said hello",
    metadata={"type": "greeting"},
    model="text-embedding-3-small"
)

# Option 2: Callback-based
db = agentvec.AgentVec("./memory", embedder=openai_embed)
memories.add_text("User said hello", metadata={})

# Option 3: Separate utility
from agentvec.embeddings import OpenAIEmbedder
embedder = OpenAIEmbedder(model="text-embedding-3-small")
memories.add(embedder.embed("User said hello"), metadata={})
```

### Supported Providers
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Cohere (embed-english-v3.0)
- Local models via sentence-transformers
- Ollama for local inference

---

## Future Considerations

### Secondary Indexes on Metadata
Create indexes on frequently filtered fields for faster queries:
```python
memories.create_index("user")  # Index on "user" field
memories.search(vec, k=10, where_={"user": "alice"})  # Uses index
```

### Memory Patterns (High-Level Abstractions)
```python
from agentvec.patterns import ConversationMemory, EntityMemory

conv_memory = ConversationMemory(db, ttl_short=3600, ttl_long=86400)
conv_memory.add_turn(user_msg, assistant_msg)
context = conv_memory.get_relevant_context(current_msg, k=5)
```

### Hybrid Search (Vector + Keyword)
```python
results = memories.hybrid_search(
    vector=embedding,
    text="project timeline",
    k=10,
    alpha=0.7  # 70% vector, 30% keyword
)
```

### Multi-Vector Records
Store multiple embeddings per record (e.g., title + content):
```python
memories.add_multi(
    vectors={"title": title_vec, "content": content_vec},
    metadata={"doc_id": "123"}
)
```

---

## Implementation Order

| Phase | Item | Dependencies |
|-------|------|--------------|
| 1 | JavaScript/WASM Bindings | None |
| 2 | CLI Tool | None |
| 3 | HTTP Server | CLI (shared code) |
| 4 | LangChain Integration | Python bindings |
| 5 | LlamaIndex Integration | Python bindings |
| 6 | Embedding Integration | LangChain/LlamaIndex |

---

## Quick Wins (Completed)

- [x] Update README roadmap with completed features
- [x] Add examples/ directory
- [x] Cleanup dead code
- [x] Mobile bindings (UniFFI)

## Quick Wins (Pending)

- [ ] Publish to crates.io
- [ ] Publish Python package to PyPI
- [ ] Add GitHub Actions for CI/CD
- [ ] Add benchmarks to CI
