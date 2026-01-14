#!/usr/bin/env python3
"""
End-to-End Example: AI Agent with Persistent Memory

This example demonstrates the complete flow of using AgentVec for AI agent memory:
1. Setting up the memory system
2. Storing facts with appropriate tiers
3. Recalling relevant context
4. Managing memory lifecycle (TTL, cleanup)

Requirements:
    pip install agentvec agentvec-memory

First run will download the embedding model (~90MB).
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Check dependencies
def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import agentvec
    except ImportError:
        missing.append("agentvec")

    try:
        from agentvec_memory import ProjectMemory, MemoryTier
    except ImportError:
        missing.append("agentvec-memory")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")

    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from agentvec_memory import ProjectMemory, MemoryTier


def main():
    """Run the end-to-end memory demonstration."""

    print("=" * 60)
    print("AgentVec Memory - End-to-End Example")
    print("=" * 60)
    print()

    # Use a temporary directory for this demo
    # In production, use a persistent path like "./agent-memory" or "~/.agentvec"
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = os.path.join(tmpdir, "demo-memory")

        print(f"[1] Initializing memory at: {memory_path}")
        print("    (First run downloads ~90MB embedding model)")
        print()

        # Initialize the memory system
        # This creates collections for each tier: working, session, project, user
        memory = ProjectMemory(memory_path)

        print("[2] Storing memories in different tiers...")
        print()

        # Store facts with appropriate tiers and TTLs
        memories_to_store = [
            # USER tier: Long-term preferences (1 year default TTL)
            ("User prefers dark mode for all interfaces", MemoryTier.USER, None),
            ("User's name is Alice and she works as a software engineer", MemoryTier.USER, None),
            ("User prefers concise explanations over verbose ones", MemoryTier.USER, None),

            # PROJECT tier: Project-specific knowledge (30 days default TTL)
            ("This project uses Python 3.11 with FastAPI framework", MemoryTier.PROJECT, None),
            ("The database is PostgreSQL running on localhost:5432", MemoryTier.PROJECT, None),
            ("API keys are stored in the .env file, never commit to git", MemoryTier.PROJECT, None),
            ("The main entry point is src/main.py", MemoryTier.PROJECT, None),

            # SESSION tier: Current work session (1 hour default TTL)
            ("Currently debugging the authentication middleware", MemoryTier.SESSION, None),
            ("Last error was a JWT token expiration issue", MemoryTier.SESSION, None),

            # WORKING tier: Immediate task context (5 minutes default TTL)
            ("Looking at file auth/jwt_handler.py lines 45-60", MemoryTier.WORKING, None),

            # Custom TTL example: Override default with 30 seconds
            ("Temporary note: check the logs for error code E401", MemoryTier.WORKING, 30),
        ]

        for content, tier, custom_ttl in memories_to_store:
            memory_id = memory.remember(content, tier=tier, ttl=custom_ttl)
            ttl_display = f"{custom_ttl}s" if custom_ttl else f"{tier.default_ttl}s (default)"
            print(f"    [{tier.value:8}] {content[:50]}...")
            print(f"              ID: {memory_id[:8]}... TTL: {ttl_display}")

        print()

        # Show memory statistics
        stats = memory.get_stats()
        print("[3] Memory Statistics:")
        print(f"    Total memories: {stats['total_memories']}")
        for tier_name, count in stats['tiers'].items():
            print(f"    - {tier_name}: {count}")
        print()

        # Demonstrate semantic recall
        print("[4] Semantic Recall Demonstrations:")
        print()

        queries = [
            # Should find user preferences
            ("What are the user's preferences?", None),

            # Should find database info
            ("How do I connect to the database?", None),

            # Should find current work context
            ("What am I currently working on?", None),

            # Filter by tier
            ("Tell me about the user", [MemoryTier.USER]),

            # Search with higher threshold (stricter matching)
            ("JWT authentication issues", None),
        ]

        for query, tier_filter in queries:
            print(f"    Query: \"{query}\"")
            if tier_filter:
                print(f"    Filter: {[t.value for t in tier_filter]}")

            results = memory.recall(
                query,
                k=3,
                threshold=0.3,
                tiers=tier_filter
            )

            if results:
                for i, mem in enumerate(results, 1):
                    print(f"      {i}. [{mem.tier.value}] (score: {mem.score:.3f})")
                    print(f"         {mem.content[:60]}...")
            else:
                print("      No matching memories found")
            print()

        # Demonstrate forgetting
        print("[5] Forgetting memories...")
        print()

        # Forget specific memories by semantic similarity
        removed = memory.forget("temporary note", threshold=0.7)
        print(f"    Forgot {removed} memories matching 'temporary note'")

        # Show updated stats
        stats = memory.get_stats()
        print(f"    New total: {stats['total_memories']} memories")
        print()

        # Demonstrate memory expiration
        print("[6] Memory Expiration Demo:")
        print()

        # Add a memory with very short TTL
        short_lived_id = memory.remember(
            "This memory expires in 2 seconds",
            tier=MemoryTier.WORKING,
            ttl=2
        )
        print(f"    Added short-lived memory (TTL: 2s)")

        # Recall immediately - should find it
        results = memory.recall("expires", k=1, threshold=0.3)
        print(f"    Immediate recall: {len(results)} result(s)")

        # Wait for expiration
        print("    Waiting 3 seconds...")
        time.sleep(3)

        # Recall again - should be filtered out by default
        results = memory.recall("expires", k=1, threshold=0.3)
        print(f"    After expiration: {len(results)} result(s) (filtered by default)")

        # Can still find with include_expired=True
        results = memory.recall("expires", k=1, threshold=0.3, include_expired=True)
        print(f"    With include_expired=True: {len(results)} result(s)")
        print()

        # Cleanup expired memories
        print("[7] Cleanup expired memories:")
        removed = memory.cleanup_expired()
        print(f"    Removed {removed} expired memories")

        stats = memory.get_stats()
        print(f"    Final total: {stats['total_memories']} memories")
        print()

        print("=" * 60)
        print("Demo complete!")
        print()
        print("To use in your own project:")
        print("  1. pip install agentvec agentvec-memory")
        print("  2. from agentvec_memory import ProjectMemory, MemoryTier")
        print("  3. memory = ProjectMemory('./my-agent-memory')")
        print("  4. memory.remember('fact', tier=MemoryTier.PROJECT)")
        print("  5. results = memory.recall('query')")
        print("=" * 60)


if __name__ == "__main__":
    main()
