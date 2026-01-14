#!/usr/bin/env python3
"""
Smart Context Injection - RAG Without the Complexity

This example demonstrates how to use AgentVec for RAG-style context injection.
Instead of complex retrieval pipelines, you get semantic context injection with
just a few lines of code.

The pattern:
1. Store documents/facts in memory
2. On each query, retrieve relevant context
3. Inject context into system prompt
4. LLM answers with full knowledge

This is simpler than traditional RAG because:
- No separate vector DB setup
- No chunking strategies to tune
- Built-in TTL for stale content
- Tiered memory for different content types

Requirements:
    pip install agentvec-memory requests python-dotenv

Usage:
    python examples/05_smart_context_injection.py
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found")
    sys.exit(1)

try:
    from agentvec_memory import ProjectMemory, MemoryTier
except ImportError:
    print("Error: agentvec-memory not installed")
    sys.exit(1)


def call_llm(messages: list, system_prompt: str) -> str:
    """Call LLM via OpenRouter API."""
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "anthropic/claude-3.5-haiku",
            "messages": full_messages,
            "max_tokens": 1000,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


class SmartAssistant:
    """
    An assistant that uses semantic memory for context injection.

    This is a simple RAG pattern:
    1. Index documents/facts into memory
    2. For each query, retrieve relevant context
    3. Inject context into system prompt
    4. Get contextually-aware responses
    """

    def __init__(self, memory_path: str, base_prompt: str = None):
        self.memory = ProjectMemory(memory_path)
        self.base_prompt = base_prompt or "You are a helpful assistant."
        self.messages = []

    def index_document(self, content: str, metadata: dict = None, tier: MemoryTier = MemoryTier.PROJECT):
        """
        Index a document or fact into memory.

        For longer documents, consider chunking them first.
        Each chunk becomes a separate memory entry.
        """
        self.memory.remember(content, tier=tier, metadata=metadata)

    def index_chunks(self, chunks: list[str], source: str = None, tier: MemoryTier = MemoryTier.PROJECT):
        """Index multiple chunks from a document."""
        for i, chunk in enumerate(chunks):
            self.memory.remember(
                chunk,
                tier=tier,
                metadata={"source": source, "chunk_index": i} if source else None
            )

    def _build_context(self, query: str, k: int = 5, threshold: float = 0.3) -> str:
        """Retrieve relevant context for a query."""
        memories = self.memory.recall(query, k=k, threshold=threshold)

        if not memories:
            return ""

        context_parts = ["RELEVANT CONTEXT:"]
        for mem in memories:
            source = mem.metadata.get("source", "")
            source_label = f" (from: {source})" if source else ""
            context_parts.append(f"- {mem.content}{source_label}")

        context_parts.append("\nUse this context to inform your response. If the context doesn't help, you can ignore it.")
        return "\n".join(context_parts)

    def ask(self, question: str, k: int = 5) -> tuple[str, list]:
        """
        Ask a question with automatic context injection.

        Returns: (response, retrieved_memories)
        """
        # Retrieve relevant context
        context = self._build_context(question, k=k)
        retrieved = self.memory.recall(question, k=k, threshold=0.3)

        # Build system prompt with context
        if context:
            system = f"{self.base_prompt}\n\n{context}"
        else:
            system = self.base_prompt

        # Add to conversation history
        self.messages.append({"role": "user", "content": question})

        # Get response
        response = call_llm(self.messages[-10:], system)

        self.messages.append({"role": "assistant", "content": response})

        return response, retrieved

    def clear_conversation(self):
        """Clear conversation history (memory persists)."""
        self.messages = []

    def get_stats(self):
        """Get memory statistics."""
        return self.memory.get_stats()


def run_demo():
    """Demonstrate smart context injection."""
    import tempfile

    print("=" * 60)
    print("  SMART CONTEXT INJECTION")
    print("  RAG Without the Complexity")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize assistant
        assistant = SmartAssistant(
            memory_path=os.path.join(tmpdir, "knowledge"),
            base_prompt="You are a helpful technical assistant. Be concise and accurate."
        )

        # ============================================================
        # PHASE 1: Index Knowledge
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 1: Indexing Knowledge Base")
        print("=" * 60)

        # Simulate indexing documentation
        documentation = [
            # API Documentation
            {
                "content": "The /users endpoint accepts GET requests to list all users. "
                          "It supports pagination with ?page=N and ?limit=N parameters. "
                          "Returns JSON array of user objects.",
                "source": "api_docs",
                "tier": MemoryTier.PROJECT,
            },
            {
                "content": "The /users endpoint accepts POST requests to create a new user. "
                          "Required fields: email, password, name. "
                          "Returns the created user object with id.",
                "source": "api_docs",
                "tier": MemoryTier.PROJECT,
            },
            {
                "content": "Authentication uses JWT tokens. Include header: Authorization: Bearer <token>. "
                          "Tokens expire after 24 hours. Use /auth/refresh to get a new token.",
                "source": "api_docs",
                "tier": MemoryTier.PROJECT,
            },
            {
                "content": "Rate limiting: 100 requests per minute per IP. "
                          "Exceeding returns 429 Too Many Requests. "
                          "Use X-RateLimit-Remaining header to track quota.",
                "source": "api_docs",
                "tier": MemoryTier.PROJECT,
            },
            # Company Policies
            {
                "content": "Coding standards: Use TypeScript for all new frontend code. "
                          "Backend services should be in Python or Go. "
                          "All code requires unit tests with >80% coverage.",
                "source": "company_policy",
                "tier": MemoryTier.PROJECT,
            },
            {
                "content": "PR process: All PRs require at least one approval. "
                          "CI must pass before merging. "
                          "Squash commits when merging to main.",
                "source": "company_policy",
                "tier": MemoryTier.PROJECT,
            },
            # Project-Specific
            {
                "content": "Current sprint focus: Improving API response times. "
                          "Target: p99 latency under 200ms. "
                          "Priority endpoints: /users, /orders, /products.",
                "source": "sprint_goals",
                "tier": MemoryTier.SESSION,  # Short-term, will expire
            },
            {
                "content": "Known bug: /orders endpoint sometimes returns duplicate entries. "
                          "Workaround: dedupe by order_id on client side. "
                          "Fix scheduled for next sprint.",
                "source": "known_issues",
                "tier": MemoryTier.SESSION,
            },
        ]

        print("\nIndexing documentation...")
        for doc in documentation:
            assistant.index_document(
                doc["content"],
                metadata={"source": doc["source"]},
                tier=doc["tier"]
            )
            print(f"  [Indexed] {doc['source']}: {doc['content'][:50]}...")

        stats = assistant.get_stats()
        print(f"\nKnowledge base: {stats['total_memories']} documents indexed")

        # ============================================================
        # PHASE 2: Ask Questions
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 2: Asking Questions (Context Auto-Injected)")
        print("=" * 60)

        questions = [
            "How do I create a new user via the API?",
            "What are our coding standards for frontend?",
            "Is there any known issue with the orders endpoint?",
            "How does authentication work?",
            "What's the rate limit for the API?",
        ]

        for question in questions:
            print(f"\n[Question]: {question}")
            print("-" * 50)

            response, retrieved = assistant.ask(question)

            print(f"[Retrieved {len(retrieved)} relevant docs]")
            if retrieved:
                for mem in retrieved[:2]:
                    source = mem.metadata.get("source", "unknown")
                    print(f"  - [{source}] {mem.content[:40]}... (score: {mem.score:.2f})")

            print(f"\n[Answer]: {response}")

        # ============================================================
        # PHASE 3: Show Context Injection
        # ============================================================
        print("\n" + "=" * 60)
        print("PHASE 3: Demonstrating Context Injection")
        print("=" * 60)

        print("\nLet's see exactly what context gets injected for a query...")
        test_query = "How do I handle rate limiting?"

        print(f"\nQuery: \"{test_query}\"")
        print("\nRetrieved context:")

        context = assistant._build_context(test_query, k=3)
        print(context)

        print("\nThis context is automatically prepended to the system prompt,")
        print("giving the LLM access to relevant documentation for each query.")

        # ============================================================
        # Summary
        # ============================================================
        print("\n" + "=" * 60)
        print("HOW IT WORKS")
        print("=" * 60)
        print("""
Traditional RAG pipeline:
  1. Set up vector database (Pinecone, Weaviate, etc.)
  2. Chunk documents with overlap
  3. Generate embeddings via API
  4. Store in vector DB
  5. On query: embed query, search, retrieve, inject

With AgentVec:
  1. assistant.index_document(content)  # That's it!
  2. response = assistant.ask(question)  # Auto-retrieval + injection

Key benefits:
  - No external services needed
  - Embeddings handled automatically
  - TTL for stale content (SESSION tier expires)
  - Tiered storage for different content types
  - Simple Python API

This is RAG without the complexity.
""")


def run_interactive():
    """Interactive mode for experimenting."""
    print("=" * 60)
    print("  SMART CONTEXT INJECTION - INTERACTIVE")
    print("=" * 60)

    memory_path = Path(__file__).parent / ".smart_assistant_memory"
    assistant = SmartAssistant(
        memory_path=str(memory_path),
        base_prompt="You are a helpful assistant with access to indexed knowledge."
    )

    stats = assistant.get_stats()
    print(f"\nLoaded {stats['total_memories']} indexed documents.")

    print("\nCommands:")
    print("  /index <text>   - Index a document")
    print("  /search <query> - Search without asking LLM")
    print("  /stats          - Show index statistics")
    print("  /clear          - Clear all indexed documents")
    print("  /quit           - Exit")
    print("\nOr just ask a question to get a context-aware answer.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit":
                    print("Goodbye!")
                    break

                elif cmd == "/index":
                    if arg:
                        assistant.index_document(arg)
                        print("Document indexed.")
                    else:
                        print("Usage: /index <document text>")

                elif cmd == "/search":
                    if arg:
                        memories = assistant.memory.recall(arg, k=5, threshold=0.2)
                        print(f"\nFound {len(memories)} relevant documents:")
                        for mem in memories:
                            print(f"  [{mem.score:.2f}] {mem.content[:60]}...")
                    else:
                        print("Usage: /search <query>")

                elif cmd == "/stats":
                    stats = assistant.get_stats()
                    print(f"\nIndexed documents: {stats['total_memories']}")

                elif cmd == "/clear":
                    confirm = input("Clear all documents? (yes/no): ")
                    if confirm.lower() == "yes":
                        assistant.memory.clear()
                        print("All documents cleared.")

            else:
                # Ask question with context injection
                print("\n[Retrieving context...]")
                response, retrieved = assistant.ask(user_input)

                if retrieved:
                    print(f"[Using {len(retrieved)} relevant documents]")

                print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        run_interactive()
    else:
        run_demo()
