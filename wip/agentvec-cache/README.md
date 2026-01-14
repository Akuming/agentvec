# agentvec-cache

**Semantic caching for LLM responses. Cut your API costs by 30-70%.**

Stop paying for the same answers twice. `agentvec-cache` caches LLM responses and returns cached results for semantically similar queries.

## The Problem

```python
# These are the same question, but you're paying for 3 API calls:
ask("What is the capital of France?")
ask("what's the capital of france")
ask("Tell me France's capital")
```

## The Solution

```python
from agentvec_cache import SemanticCache

cache = SemanticCache("./my_cache.db")

@cache.cached()
def ask(question: str) -> str:
    return openai.chat.completions.create(...)

ask("What is the capital of France?")  # API call (~800ms, $0.002)
ask("Tell me France's capital")        # Cache hit (~5ms, $0.00)
```

## Installation

```bash
pip install agentvec-cache
```

Or for development:

```bash
cd agentvec-cache
pip install -e .
```

## Quick Start

```python
from agentvec_cache import SemanticCache

# Initialize (creates local database)
cache = SemanticCache(
    path="./cache.db",
    threshold=0.92,  # Similarity threshold (0.0-1.0)
    ttl=3600,        # Cache TTL in seconds
)

# Option 1: Decorator (recommended)
@cache.cached()
def my_llm_function(prompt: str) -> str:
    # Your expensive LLM call here
    return response

# Option 2: Manual get/set
cached = cache.get("What is Python?")
if cached is None:
    result = call_llm("What is Python?")
    cache.set("What is Python?", result)
```

## Tuning the Threshold

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.98+ | Almost exact match only | High-stakes, accuracy critical |
| 0.92-0.95 | Conservative (default) | General purpose |
| 0.85-0.90 | Moderate | FAQ bots, support |
| < 0.85 | Aggressive | Cost optimization priority |

## Cache Statistics

```python
print(cache.stats)
# CacheStats(hits=150, misses=23, hit_rate=86.7%, latency_saved=45000ms)
```

## Cache Management

```python
# Invalidate specific query
cache.invalidate("What is Python?")

# Invalidate similar queries
cache.invalidate_similar("pricing information", threshold=0.8)

# Clear everything
cache.clear()

# Remove expired entries
cache.compact()
```

## License

MIT
