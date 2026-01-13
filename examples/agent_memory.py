#!/usr/bin/env python3
"""
AI Agent Memory Patterns with AgentVec.

This example demonstrates how to use AgentVec for different types
of AI agent memory: working memory, episodic memory, and semantic memory.

Run with: python examples/agent_memory.py
"""

import agentvec
import tempfile
import shutil
from datetime import datetime


def main():
    temp_dir = tempfile.mkdtemp(prefix="agent_memory_")
    print("=== AI Agent Memory Example ===\n")

    try:
        db = agentvec.AgentVec(temp_dir)

        # =====================================================================
        # Working Memory (Short-term, high turnover)
        # =====================================================================
        # Use for: Current conversation context, temporary state
        # TTL: Minutes to hours
        print("--- Working Memory ---")

        working = db.collection("working", dim=384, metric="cosine")

        # Store current conversation turns with short TTL
        turn1 = [0.1] * 384  # Embedding for "What's the weather?"
        working.add(
            vector=turn1,
            metadata={
                "role": "user",
                "content": "What's the weather like today?",
                "turn": 1
            },
            ttl=1800,  # 30 minute TTL
        )

        turn2 = [0.12] * 384  # Embedding for weather response
        working.add(
            vector=turn2,
            metadata={
                "role": "assistant",
                "content": "It's sunny and 72°F today.",
                "turn": 2
            },
            ttl=1800,
        )

        print(f"Working memory: {len(working)} items")

        # =====================================================================
        # Episodic Memory (Medium-term, experiences)
        # =====================================================================
        # Use for: Past conversations, user interactions, events
        # TTL: Hours to days
        print("\n--- Episodic Memory ---")

        episodic = db.collection("episodic", dim=384, metric="cosine")

        # Store conversation summaries
        conv_summary = [0.2] * 384
        episodic.add(
            vector=conv_summary,
            metadata={
                "type": "conversation",
                "user_id": "user_123",
                "summary": "User asked about weather and project deadlines",
                "sentiment": "neutral",
                "timestamp": datetime.now().isoformat()
            },
            ttl=86400 * 7,  # 7 day TTL
        )

        # Store user preferences learned from interactions
        pref_embedding = [0.25] * 384
        episodic.add(
            vector=pref_embedding,
            metadata={
                "type": "preference",
                "user_id": "user_123",
                "preference": "prefers concise responses",
                "confidence": 0.8
            },
            ttl=86400 * 30,  # 30 day TTL
        )

        print(f"Episodic memory: {len(episodic)} items")

        # =====================================================================
        # Semantic Memory (Long-term, knowledge)
        # =====================================================================
        # Use for: Facts, documentation, permanent knowledge
        # TTL: None (permanent) or very long
        print("\n--- Semantic Memory ---")

        # Larger dimensions for more capable embedding model
        semantic = db.collection("semantic", dim=1536, metric="cosine")

        # Store factual knowledge (no TTL - permanent)
        fact1 = [0.3] * 1536
        semantic.add(
            vector=fact1,
            metadata={
                "type": "fact",
                "domain": "company",
                "content": "The company was founded in 2020",
                "source": "about_page"
            },
            id="fact_founding",
            # No TTL - permanent
        )

        fact2 = [0.35] * 1536
        semantic.add(
            vector=fact2,
            metadata={
                "type": "procedure",
                "domain": "support",
                "content": "To reset password: go to settings > security > reset",
                "source": "help_docs"
            },
            id="proc_password_reset",
        )

        print(f"Semantic memory: {len(semantic)} items")

        # =====================================================================
        # Memory Retrieval Patterns
        # =====================================================================
        print("\n--- Memory Retrieval ---")

        # 1. Get relevant context for current query
        query_embedding = [0.11] * 384

        # Search working memory first (most recent context)
        working_results = working.search(vector=query_embedding, k=3)
        print("Recent context from working memory:")
        for r in working_results:
            print(f"  - {r.id}: {r.metadata['content']}")

        # 2. Get user-specific memories
        user_memories = episodic.search(
            vector=query_embedding,
            k=5,
            where_={"user_id": "user_123"}
        )
        print("\nUser-specific memories:")
        for r in user_memories:
            if "summary" in r.metadata:
                print(f"  - {r.metadata['summary']}")
            if "preference" in r.metadata:
                print(f"  - Preference: {r.metadata['preference']}")

        # 3. Get domain-specific knowledge
        domain_query = [0.32] * 1536
        knowledge = semantic.search(
            vector=domain_query,
            k=3,
            where_={"domain": "support"}
        )
        print("\nSupport knowledge:")
        for r in knowledge:
            print(f"  - {r.metadata['content']}")

        # =====================================================================
        # Memory Maintenance
        # =====================================================================
        print("\n--- Memory Maintenance ---")

        # Update a memory (upsert)
        updated_embedding = [0.26] * 384
        episodic.upsert(
            id="user_123_preferences",
            vector=updated_embedding,
            metadata={
                "type": "preference",
                "user_id": "user_123",
                "preference": "prefers concise responses with examples",
                "confidence": 0.9,
                "updated_at": datetime.now().isoformat()
            },
            ttl=86400 * 30,
        )
        print("Updated user preferences")

        # Compact expired memories
        working_stats = working.compact()
        episodic_stats = episodic.compact()
        print(f"Compacted: {working_stats.expired_removed} working, "
              f"{episodic_stats.expired_removed} episodic expired")

        # Sync all changes
        db.sync()
        print("All memories synced to disk")

        # =====================================================================
        # Memory Statistics
        # =====================================================================
        print("\n--- Memory Statistics ---")
        print(f"Working memory:  {len(working)} records, "
              f"{working.vectors_size_bytes} bytes")
        print(f"Episodic memory: {len(episodic)} records, "
              f"{episodic.vectors_size_bytes} bytes")
        print(f"Semantic memory: {len(semantic)} records, "
              f"{semantic.vectors_size_bytes} bytes")

        print("\nExample completed!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
