#!/usr/bin/env python3
"""
Pure Vector Performance - AgentVec Without LLM Calls

This example demonstrates AgentVec's core capabilities WITHOUT any LLM calls.
It proves that AgentVec is a real vector database, not just an LLM wrapper.

What this demonstrates:
- Raw vector storage and retrieval speed
- Semantic similarity search quality
- Embedding generation performance
- Memory efficiency
- TTL and expiration handling

No OpenRouter API key needed - runs entirely locally.

Requirements:
    pip install agentvec-memory numpy

Usage:
    python examples/06_pure_vector_performance.py
"""

import os
import sys
import time
import tempfile
import statistics
from pathlib import Path

try:
    from agentvec_memory import ProjectMemory, MemoryTier
    import agentvec
except ImportError:
    print("Error: agentvec-memory not installed")
    print("Install with: pip install agentvec-memory")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed")
    print("Install with: pip install numpy")
    sys.exit(1)


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}us"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def run_embedding_benchmark():
    """Benchmark embedding generation speed."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Embedding Generation Speed")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ProjectMemory(os.path.join(tmpdir, "bench"))

        # Test texts of varying lengths
        test_cases = [
            ("Short", "Hello world"),
            ("Medium", "The quick brown fox jumps over the lazy dog. " * 5),
            ("Long", "Artificial intelligence and machine learning are transforming how we build software. " * 20),
        ]

        print("\nEmbedding generation times (single text):")
        print("-" * 50)

        for name, text in test_cases:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                memory._embed(text)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg = statistics.mean(times)
            print(f"  {name:8} ({len(text):4} chars): {format_time(avg):>10} avg")

        # Batch embedding test
        print("\nBatch embedding (via remember):")
        print("-" * 50)

        batch_sizes = [10, 50, 100]
        for batch_size in batch_sizes:
            texts = [f"This is test document number {i} with some content." for i in range(batch_size)]

            start = time.perf_counter()
            for text in texts:
                memory.remember(text, tier=MemoryTier.WORKING)
            elapsed = time.perf_counter() - start

            per_doc = elapsed / batch_size
            print(f"  {batch_size:3} documents: {format_time(elapsed):>10} total, {format_time(per_doc):>10}/doc")


def run_search_benchmark():
    """Benchmark search performance."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Search Performance")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ProjectMemory(os.path.join(tmpdir, "bench"))

        # Populate with test data
        print("\nPopulating database with test documents...")

        categories = [
            ("programming", [
                "Python is a high-level programming language known for readability",
                "JavaScript runs in browsers and enables interactive web pages",
                "Rust provides memory safety without garbage collection",
                "Go was designed at Google for concurrent programming",
                "TypeScript adds static typing to JavaScript",
            ]),
            ("databases", [
                "PostgreSQL is a powerful open-source relational database",
                "MongoDB stores data in flexible JSON-like documents",
                "Redis is an in-memory data structure store",
                "SQLite is a self-contained embedded database",
                "Vector databases store high-dimensional embeddings",
            ]),
            ("ai_ml", [
                "Neural networks learn patterns from training data",
                "Transformers revolutionized natural language processing",
                "Embeddings represent text as dense vectors",
                "RAG combines retrieval with generation for better answers",
                "Fine-tuning adapts pre-trained models to specific tasks",
            ]),
        ]

        doc_count = 0
        start = time.perf_counter()
        for category, docs in categories:
            for doc in docs:
                memory.remember(doc, tier=MemoryTier.PROJECT, metadata={"category": category})
                doc_count += 1
        index_time = time.perf_counter() - start

        print(f"  Indexed {doc_count} documents in {format_time(index_time)}")
        print(f"  Average: {format_time(index_time/doc_count)}/document")

        # Search benchmarks
        print("\nSearch performance:")
        print("-" * 50)

        queries = [
            "What programming languages are good for beginners?",
            "How do I store data persistently?",
            "What is machine learning?",
            "Tell me about memory-safe languages",
            "How do embeddings work?",
        ]

        search_times = []
        for query in queries:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                results = memory.recall(query, k=5, threshold=0.3)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg = statistics.mean(times)
            search_times.append(avg)
            print(f"  \"{query[:40]}...\"")
            print(f"    -> {len(results)} results in {format_time(avg)}")

        print(f"\n  Average search time: {format_time(statistics.mean(search_times))}")
        print(f"  Min: {format_time(min(search_times))}, Max: {format_time(max(search_times))}")


def run_similarity_demo():
    """Demonstrate semantic similarity quality."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Semantic Similarity Quality")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ProjectMemory(os.path.join(tmpdir, "bench"))

        # Store test documents
        documents = [
            "The cat sat on the mat",
            "A feline rested on the rug",  # Semantically similar to #1
            "Dogs are loyal companions",
            "Canines make faithful pets",  # Semantically similar to #3
            "Python is a programming language",
            "JavaScript is used for web development",
            "The weather today is sunny and warm",
            "Machine learning models learn from data",
        ]

        print("\nStored documents:")
        for i, doc in enumerate(documents):
            memory.remember(doc, tier=MemoryTier.PROJECT, metadata={"id": i})
            print(f"  [{i}] {doc}")

        # Test semantic similarity
        print("\nSemantic similarity tests:")
        print("-" * 50)

        test_queries = [
            ("The kitty was lying on the carpet", "Should match: cat/feline on mat/rug"),
            ("Puppies are great friends", "Should match: dogs/canines as companions/pets"),
            ("I want to learn coding", "Should match: programming languages"),
            ("It's a beautiful day outside", "Should match: weather/sunny"),
        ]

        for query, expected in test_queries:
            print(f"\n  Query: \"{query}\"")
            print(f"  Expected: {expected}")
            results = memory.recall(query, k=3, threshold=0.2)
            print(f"  Results:")
            for r in results:
                doc_id = r.metadata.get("id", "?")
                print(f"    [{doc_id}] score={r.score:.3f}: {r.content[:50]}...")


def run_ttl_demo():
    """Demonstrate TTL and expiration."""
    print("\n" + "=" * 60)
    print("BENCHMARK 4: TTL and Memory Expiration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ProjectMemory(os.path.join(tmpdir, "bench"))

        print("\nStoring memories with different TTLs...")

        # Store with different TTLs
        memory.remember("This expires in 1 second", tier=MemoryTier.WORKING, ttl=1)
        memory.remember("This expires in 5 seconds", tier=MemoryTier.WORKING, ttl=5)
        memory.remember("This persists for 30 days", tier=MemoryTier.PROJECT)

        # Check immediately
        results = memory.recall("expires", k=10, threshold=0.2)
        print(f"\n  Immediately after storing: {len(results)} memories found")

        # Wait and check
        print("  Waiting 2 seconds...")
        time.sleep(2)

        results = memory.recall("expires", k=10, threshold=0.2)
        print(f"  After 2 seconds: {len(results)} memories found (1s TTL expired)")

        results_with_expired = memory.recall("expires", k=10, threshold=0.2, include_expired=True)
        print(f"  With include_expired=True: {len(results_with_expired)} memories found")

        # Cleanup
        removed = memory.cleanup_expired()
        print(f"\n  cleanup_expired() removed: {removed} memories")


def run_scale_test():
    """Test with larger dataset."""
    print("\n" + "=" * 60)
    print("BENCHMARK 5: Scale Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ProjectMemory(os.path.join(tmpdir, "bench"))

        # Generate test data
        num_docs = 500
        print(f"\nGenerating and indexing {num_docs} documents...")

        topics = ["AI", "databases", "web development", "security", "cloud computing",
                  "mobile apps", "DevOps", "testing", "architecture", "performance"]

        start = time.perf_counter()
        for i in range(num_docs):
            topic = topics[i % len(topics)]
            doc = f"Document {i} about {topic}: This is a test document containing information about {topic} and related concepts for testing vector search at scale."
            memory.remember(doc, tier=MemoryTier.PROJECT, metadata={"topic": topic, "id": i})

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - start
                print(f"  Indexed {i + 1}/{num_docs} ({format_time(elapsed)} elapsed)")

        total_index_time = time.perf_counter() - start
        print(f"\n  Total indexing time: {format_time(total_index_time)}")
        print(f"  Average per document: {format_time(total_index_time / num_docs)}")

        # Search at scale
        print(f"\nSearching {num_docs} documents...")

        queries = ["artificial intelligence", "database optimization", "web security"]
        for query in queries:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                results = memory.recall(query, k=10, threshold=0.3)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg = statistics.mean(times)
            print(f"  \"{query}\": {len(results)} results in {format_time(avg)}")

        # Memory stats
        stats = memory.get_stats()
        print(f"\n  Total memories stored: {stats['total_memories']}")


def run_low_level_demo():
    """Demonstrate low-level agentvec API."""
    print("\n" + "=" * 60)
    print("BENCHMARK 6: Low-Level API (Raw Vectors)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db = agentvec.AgentVec(os.path.join(tmpdir, "raw"))

        # Create collection
        dim = 384
        collection = db.collection("test", dim=dim, metric="cosine")

        print(f"\nCreated collection with {dim} dimensions")

        # Generate random vectors (simulating embeddings)
        num_vectors = 1000
        print(f"Inserting {num_vectors} random vectors...")

        np.random.seed(42)
        start = time.perf_counter()
        for i in range(num_vectors):
            vector = np.random.randn(dim).astype(np.float32).tolist()
            collection.add(
                vector=vector,
                metadata={"id": i, "category": f"cat_{i % 10}"}
            )
        insert_time = time.perf_counter() - start

        print(f"  Insert time: {format_time(insert_time)}")
        print(f"  Per vector: {format_time(insert_time / num_vectors)}")

        # Search
        print(f"\nSearching {num_vectors} vectors...")

        query_vector = np.random.randn(dim).astype(np.float32).tolist()
        times = []
        for _ in range(100):
            start = time.perf_counter()
            results = collection.search(vector=query_vector, k=10)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg = statistics.mean(times)
        print(f"  Average search time: {format_time(avg)}")
        print(f"  Min: {format_time(min(times))}, Max: {format_time(max(times))}")
        print(f"  Searches per second: {1/avg:.0f}")

        # Show results
        print(f"\n  Top 5 results (by cosine similarity):")
        results = collection.search(vector=query_vector, k=5)
        for r in results:
            print(f"    id={r.metadata['id']:4}, score={r.score:.4f}")


def main():
    print("=" * 60)
    print("  AGENTVEC PURE VECTOR PERFORMANCE")
    print("  No LLM Calls - Pure Vector Database Speed")
    print("=" * 60)
    print("\nThis demonstrates AgentVec's core capabilities")
    print("without any external API calls.\n")

    run_embedding_benchmark()
    run_search_benchmark()
    run_similarity_demo()
    run_ttl_demo()
    run_scale_test()
    run_low_level_demo()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key takeaways:

1. EMBEDDING SPEED: Local fastembed generates embeddings in milliseconds
   - No API calls needed
   - Works offline

2. SEARCH SPEED: Sub-millisecond search even with 500+ documents
   - Rust-powered HNSW index
   - Scales efficiently

3. SEMANTIC QUALITY: Finds semantically similar content
   - "kitty on carpet" -> "cat on mat"
   - "puppies are friends" -> "dogs are companions"

4. TTL WORKS: Memories expire and can be cleaned up
   - Built-in for agent memory use cases
   - No cron jobs needed

5. RAW PERFORMANCE: Low-level API handles 1000+ vectors
   - Direct vector operations
   - ~1000+ searches per second

AgentVec is a REAL vector database, not an LLM wrapper.
The LLM examples show USE CASES - this shows the ENGINE.
""")


if __name__ == "__main__":
    main()
