#!/usr/bin/env python3
"""
Demo: Semantic Cache in action.

Shows cache hits for semantically similar queries.

Usage:
    python examples/demo.py
"""

import sys
import time
import tempfile
import shutil

# Add parent directory to path for local development
sys.path.insert(0, str(__file__).rsplit("examples", 1)[0])

from agentvec_cache import SemanticCache


def simulate_llm_call(prompt: str) -> str:
    """Simulate an expensive LLM call (1 second delay)."""
    time.sleep(1.0)
    return f"This is the answer to: {prompt}"


def main():
    temp_dir = tempfile.mkdtemp(prefix="cache_demo_")
    print(f"Cache location: {temp_dir}\n")

    try:
        # Initialize cache
        print("Loading embedding model (first time may take a moment)...")
        cache = SemanticCache(
            path=temp_dir,
            threshold=0.88,  # 88% similarity required for hit
            ttl=3600,        # 1 hour default TTL
        )
        print(f"Cache initialized: {cache}\n")

        # Wrap the LLM function
        @cache.cached()
        def ask(question: str) -> str:
            return simulate_llm_call(question)

        print("=" * 60)
        print("SEMANTIC CACHE DEMO")
        print("=" * 60)

        # Test 1: First call - cache miss (slow)
        print("\n[Query 1] 'What is the capital of France?'")
        start = time.perf_counter()
        result = ask("What is the capital of France?")
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Result: {result}")
        print(f"  Time: {elapsed:.0f}ms [MISS - called LLM]")

        # Test 2: Exact same query - cache hit (fast)
        print("\n[Query 2] 'What is the capital of France?' (exact repeat)")
        start = time.perf_counter()
        result = ask("What is the capital of France?")
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Result: {result}")
        print(f"  Time: {elapsed:.0f}ms [HIT - exact match]")

        # Test 3: Semantically similar - cache hit (fast)
        print("\n[Query 3] 'Tell me the capital of France' (semantic match)")
        start = time.perf_counter()
        result = ask("Tell me the capital of France")
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Result: {result}")
        print(f"  Time: {elapsed:.0f}ms [HIT - semantic match]")

        # Test 4: Another variation
        print("\n[Query 4] 'France capital city?' (semantic match)")
        start = time.perf_counter()
        result = ask("France capital city?")
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Result: {result}")
        print(f"  Time: {elapsed:.0f}ms [HIT - semantic match]")

        # Test 5: Different question - cache miss (slow)
        print("\n[Query 5] 'What is Python?' (different topic)")
        start = time.perf_counter()
        result = ask("What is Python?")
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Result: {result}")
        print(f"  Time: {elapsed:.0f}ms [MISS - new topic]")

        # Test 6: Similar to Python question
        print("\n[Query 6] 'Tell me about Python programming' (semantic match)")
        start = time.perf_counter()
        result = ask("Tell me about Python programming")
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Result: {result}")
        print(f"  Time: {elapsed:.0f}ms [HIT - semantic match]")

        # Stats
        print("\n" + "=" * 60)
        print("CACHE STATISTICS")
        print("=" * 60)
        print(f"  {cache.stats}")
        print(f"  Entries in cache: {len(cache)}")
        print(f"  Estimated time saved: {cache.stats.total_latency_saved_ms:.0f}ms")

        # Demonstrate invalidation
        print("\n" + "=" * 60)
        print("CACHE INVALIDATION DEMO")
        print("=" * 60)

        removed = cache.invalidate_similar("capital of France", threshold=0.8)
        print(f"  Invalidated {removed} entries similar to 'capital of France'")
        print(f"  Entries remaining: {len(cache)}")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\n[Cleaned up temporary cache]")


if __name__ == "__main__":
    main()
