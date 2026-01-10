#!/usr/bin/env python3
"""
Basic AgentVec usage example in Python.

Run with: python examples/basic_usage.py
"""

import agentvec
import tempfile
import shutil
import os


def main():
    # Create a temporary directory for this example
    temp_dir = tempfile.mkdtemp(prefix="agentvec_example_")
    print(f"Opening database at {temp_dir}")

    try:
        # Open or create a database
        db = agentvec.AgentVec(temp_dir)

        # Create a collection for storing memories
        # - 384 dimensions (typical for small embedding models like all-MiniLM-L6-v2)
        # - Cosine similarity (best for text embeddings)
        memories = db.collection("memories", dim=384, metric="cosine")

        print(f"Created collection: {memories.name}")
        print(f"Dimensions: {memories.dimensions}")
        print(f"Metric: {memories.metric}")

        # Generate some sample embeddings (in real usage, use an embedding model)
        embedding1 = [0.1] * 384
        embedding2 = [0.2] * 384
        embedding3 = [-0.1] * 384

        # Add vectors with metadata
        id1 = memories.add(
            vector=embedding1,
            metadata={
                "type": "conversation",
                "user": "alice",
                "message": "Hello, how are you?"
            },
        )
        print(f"Added record: {id1}")

        # Add with custom ID and TTL
        id2 = memories.add(
            vector=embedding2,
            metadata={
                "type": "conversation",
                "user": "bob",
                "message": "I'm working on a project"
            },
            id="conv_002",
            ttl=3600,  # 1 hour TTL
        )
        print(f"Added record: {id2}")

        # Upsert (insert or update)
        memories.upsert(
            id="conv_003",
            vector=embedding3,
            metadata={
                "type": "note",
                "user": "alice",
                "message": "Remember to follow up"
            },
        )
        print("Upserted record: conv_003")

        # Check collection size
        print(f"Collection size: {len(memories)} records")

        # Search for similar vectors
        query = [0.15] * 384
        results = memories.search(vector=query, k=10)

        print("\nSearch results (top 10):")
        for result in results:
            print(f"  {result.id} (score: {result.score:.4f}): {result.metadata}")

        # Search with filter
        filtered_results = memories.search(
            vector=query,
            k=10,
            where_={"user": "alice"}
        )

        print("\nFiltered results (user = alice):")
        for result in filtered_results:
            print(f"  {result.id} (score: {result.score:.4f}): {result.metadata}")

        # Get a specific record by ID
        record = memories.get("conv_003")
        if record:
            print(f"\nRetrieved record conv_003:")
            print(f"  ID: {record.id}")
            print(f"  Metadata: {record.metadata}")

        # Delete a record
        deleted = memories.delete(id1)
        print(f"\nDeleted {id1}: {deleted}")

        # Compact to remove expired/deleted records
        stats = memories.compact()
        print(f"Compact stats: {stats.expired_removed} expired, {stats.tombstones_removed} tombstones removed")

        # Sync to disk (ensures durability)
        db.sync()
        print("Database synced to disk")

        print("\nExample completed successfully!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
