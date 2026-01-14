# agentvec-search

**Semantic code search - search your codebase with natural language.**

Stop grepping for keywords. Ask questions about your code in plain English.

## The Problem

```bash
# Traditional search - keyword matching
grep -r "auth" ./src
# Returns: 500 results including comments, variable names, and irrelevant matches

# What you actually want
"where is user authentication handled?"
# Returns: src/auth/login.rs:42 - the actual authentication logic
```

## The Solution

```bash
# Index your codebase once
agentvec-search index ./src

# Search with natural language
agentvec-search search "where is authentication handled"
# 1. src/auth/login.rs:42 (score: 0.623)
# 2. src/middleware/auth.rs:15 (score: 0.571)
```

## Installation

```bash
pip install agentvec-search
```

Or for development:

```bash
cd agentvec-search
pip install -e .
```

## Quick Start

### 1. Index your codebase

```bash
# Index current directory
agentvec-search index .

# Index specific directory
agentvec-search index ./src

# Index specific languages
agentvec-search index ./src -l python -l rust

# Index specific extensions
agentvec-search index ./src -e .py -e .rs
```

### 2. Search

```bash
# Basic search
agentvec-search search "where is authentication handled"

# More results
agentvec-search search "database connection" -k 10

# Higher threshold (stricter matching)
agentvec-search search "error handling" -t 0.4

# File paths only (no content preview)
agentvec-search search "caching logic" --no-content
```

### 3. Interactive Mode

```bash
agentvec-search interactive

search> where is authentication handled
1. src/auth/login.rs:42 (score: 0.623)
2. src/middleware/auth.rs:15 (score: 0.571)

search> how does caching work
1. src/cache/manager.rs:28 (score: 0.589)

search> :quit
```

## Commands

| Command | Description |
|---------|-------------|
| `index <dir>` | Index a codebase |
| `search <query>` | Search for code |
| `interactive` | Interactive search mode |
| `stats` | Show index statistics |
| `languages` | List supported languages |

## Options

### Index Options

| Option | Description |
|--------|-------------|
| `-i, --index PATH` | Index location (default: .agentvec-index) |
| `-l, --languages` | Languages to index (e.g., `-l python -l rust`) |
| `-e, --extensions` | Extensions to index (e.g., `-e .py -e .rs`) |
| `-c, --chunk-size` | Lines per chunk (default: 50) |
| `-o, --overlap` | Overlap between chunks (default: 10) |
| `--clear` | Clear existing index first |

### Search Options

| Option | Description |
|--------|-------------|
| `-i, --index PATH` | Index location |
| `-k, --results NUM` | Number of results (default: 10) |
| `-t, --threshold` | Minimum similarity score (default: 0.0) |
| `-c, --context` | Lines of context to show (default: 5) |
| `--no-content` | Only show file paths |

## Supported Languages

```bash
agentvec-search languages
```

- Rust (.rs)
- Python (.py)
- JavaScript (.js, .jsx, .mjs)
- TypeScript (.ts, .tsx)
- Go (.go)
- Java (.java)
- C (.c, .h)
- C++ (.cpp, .hpp, .cc)
- And many more...

## How It Works

1. **Indexing**: Code files are split into overlapping chunks and embedded using sentence-transformers
2. **Storage**: Embeddings are stored in AgentVec (local vector database)
3. **Search**: Your query is embedded and compared against code chunks using cosine similarity
4. **Results**: Most similar chunks are returned with file paths and line numbers

## Tips

- **Start broad**: Begin with general queries, then refine
- **Use domain terms**: "authentication" works better than "auth"
- **Lower threshold if needed**: Use `-t 0.3` for more results
- **Re-index after changes**: Run `index --clear` to update

## Limitations

- Works best with English queries and code with English identifiers
- Similarity scores for code are typically lower (0.4-0.7) than natural language
- Very short or cryptic code may not match well
- Index needs to be rebuilt after code changes

## License

MIT
