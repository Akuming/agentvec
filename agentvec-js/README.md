# AgentVec JavaScript/WASM Bindings

JavaScript and WebAssembly bindings for [AgentVec](https://github.com/Akuming/agentvec), a lightweight vector database for AI agent memory.

## Installation

```bash
npm install agentvec
```

## Quick Start

```javascript
import { AgentVec, Metric } from 'agentvec';

// Open or create a database
const db = new AgentVec('./agent_memory');

// Create a collection for storing memories
const memories = db.collection('episodic', 384, Metric.Cosine);

// Add a vector with metadata
const id = memories.add(
  new Float32Array([0.1, 0.2, /* ... 384 dimensions */]),
  { type: 'conversation', user: 'alice' },
  null,  // auto-generate ID
  3600   // TTL: 1 hour
);

// Search for similar vectors
const results = memories.search(
  new Float32Array([0.15, 0.25, /* ... */]),
  10,    // top 10 results
  null   // no filter
);

for (const result of results) {
  console.log(`${result.id}: ${result.score} - ${JSON.stringify(result.metadata)}`);
}

// Search with filter
const filtered = memories.search(
  queryVector,
  10,
  { user: 'alice' }
);

// Cleanup
db.sync();
```

## API Reference

### AgentVec

```typescript
class AgentVec {
  constructor(path: string);

  collection(name: string, dimensions: number, metric: Metric): Collection;
  getCollection(name: string): Collection;
  collections(): string[];
  dropCollection(name: string): boolean;
  sync(): void;
  recoveryStats(): RecoveryStats;
}
```

### Collection

```typescript
class Collection {
  add(vector: Float32Array, metadata: object, id?: string, ttl?: number): string;
  upsert(id: string, vector: Float32Array, metadata: object, ttl?: number): void;
  search(vector: Float32Array, k: number, filter?: object): SearchResult[];
  get(id: string): SearchResult | null;
  delete(id: string): boolean;
  compact(): CompactStats;
  preload(): void;
  sync(): void;

  readonly length: number;
  readonly dimensions: number;
  readonly name: string;
  readonly metric: Metric;
  readonly vectorsSizeBytes: number;

  isEmpty(): boolean;
}
```

### Metric

```typescript
enum Metric {
  Cosine = 0,  // Normalized dot product (best for text embeddings)
  Dot = 1,     // Raw dot product
  L2 = 2       // Euclidean distance
}
```

### SearchResult

```typescript
interface SearchResult {
  readonly id: string;
  readonly score: number;
  readonly metadata: object;
  readonly metadataJson: string;
}
```

### Filter Operators

Filters support MongoDB-like query operators:

```javascript
// Equality
memories.search(vec, 10, { user: 'alice' });

// Comparison
memories.search(vec, 10, {
  score: { $gt: 0.8 },
  count: { $lte: 100 }
});

// Set membership
memories.search(vec, 10, {
  status: { $in: ['active', 'pending'] },
  tag: { $nin: ['spam', 'deleted'] }
});

// Not equal
memories.search(vec, 10, { type: { $ne: 'system' } });
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`

## Building from Source

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [Node.js](https://nodejs.org/) (16+)

### Build

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for Node.js
wasm-pack build --target nodejs --out-dir pkg

# Build for browser
wasm-pack build --target web --out-dir pkg-web
```

### Test

```bash
# Run Rust tests
cargo test

# Run wasm-bindgen tests
wasm-pack test --node
```

## Platform Support

| Platform | Support |
|----------|---------|
| Node.js 16+ | Full support |
| Deno | Should work (untested) |
| Bun | Should work (untested) |
| Browser | Limited (no filesystem) |

**Note:** Browser support is limited because AgentVec requires filesystem access. For browser use cases, consider using IndexedDB or a server-side deployment.

## Memory Management

AgentVec uses memory-mapped files under the hood. In Node.js, this works seamlessly with the filesystem. Memory usage scales with the working set, not the total dataset size.

```javascript
// Preload vectors into memory for faster access
memories.preload();

// Check storage size
console.log(`Storage: ${memories.vectorsSizeBytes / 1024 / 1024} MB`);

// Compact to reclaim space from deleted/expired records
const stats = memories.compact();
console.log(`Freed ${stats.bytesFreed} bytes`);
```

## License

MIT OR Apache-2.0
