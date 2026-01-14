#!/usr/bin/env python3
"""
Research Accumulator - Knowledge Building Over Time

This example demonstrates an AI agent that accumulates knowledge from
conversations and can synthesize answers from everything it has learned.

Use cases:
- Research assistant that builds up domain knowledge
- Learning companion that tracks what you've studied
- Project documentation bot that remembers decisions

Requirements:
    pip install agentvec-memory requests python-dotenv

Usage:
    python examples/04_research_accumulator.py          # Auto demo
    python examples/04_research_accumulator.py --interactive  # Interactive mode
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in environment")
    sys.exit(1)

try:
    from agentvec_memory import ProjectMemory, MemoryTier
except ImportError:
    print("Error: agentvec-memory not installed")
    sys.exit(1)


MEMORY_PATH = Path(__file__).parent / ".research_accumulator_memory"


def call_llm(messages: list, system_prompt: str, max_tokens: int = 1000) -> str:
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
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def extract_facts(text: str, topic: str) -> list[str]:
    """Use LLM to extract key facts from text."""
    prompt = f"""Extract the key facts from this text about "{topic}".
Return a JSON array of strings, each being a single fact.
Only include concrete, factual information worth remembering.
Keep each fact concise (1-2 sentences max).

Text:
{text}

Return ONLY a valid JSON array, no other text:"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "anthropic/claude-3.5-haiku",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Clean up JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        return json.loads(content)
    except Exception as e:
        print(f"  [Warning: Could not extract facts: {e}]")
        return []


def synthesize_knowledge(memory: ProjectMemory, query: str) -> str:
    """Retrieve relevant knowledge and synthesize an answer."""
    # Get relevant memories
    memories = memory.recall(query, k=10, threshold=0.25)

    if not memories:
        return None

    # Build context from memories
    knowledge_points = []
    for mem in memories:
        knowledge_points.append(f"- {mem.content}")

    knowledge_context = "\n".join(knowledge_points)

    synthesis_prompt = f"""Based on the following knowledge I've accumulated, answer the question.
Only use information from the provided knowledge. If the knowledge doesn't cover the question, say so.

ACCUMULATED KNOWLEDGE:
{knowledge_context}

QUESTION: {query}

Provide a clear, synthesized answer:"""

    response = call_llm(
        [{"role": "user", "content": synthesis_prompt}],
        "You are a research assistant that synthesizes knowledge from accumulated facts.",
        max_tokens=800
    )

    return response, memories


def run_auto_demo():
    """Run automated demonstration of knowledge accumulation."""
    import shutil

    print("=" * 60)
    print("  RESEARCH ACCUMULATOR - AUTO DEMO")
    print("  Building Knowledge Over Time")
    print("=" * 60)

    # Clean start for demo
    if MEMORY_PATH.exists():
        shutil.rmtree(MEMORY_PATH)

    memory = ProjectMemory(str(MEMORY_PATH))

    # Simulate research sessions on different topics
    research_sessions = [
        {
            "topic": "Rust programming language",
            "content": """
            Rust is a systems programming language focused on safety, speed, and concurrency.
            It was created by Mozilla and first released in 2010.
            Rust uses a borrow checker to ensure memory safety without garbage collection.
            The language has concepts like ownership, borrowing, and lifetimes.
            Rust is often used for systems programming, WebAssembly, and CLI tools.
            Cargo is Rust's package manager and build system.
            """
        },
        {
            "topic": "Vector databases",
            "content": """
            Vector databases store high-dimensional vectors for similarity search.
            They use algorithms like HNSW (Hierarchical Navigable Small World) for fast nearest neighbor search.
            Common use cases include semantic search, recommendation systems, and RAG pipelines.
            Popular vector databases include Pinecone, Weaviate, Milvus, and Qdrant.
            Vectors are typically generated by embedding models like OpenAI's text-embedding-ada-002.
            The cosine similarity metric is commonly used to compare vectors.
            """
        },
        {
            "topic": "AI agent memory",
            "content": """
            AI agents benefit from persistent memory to maintain context across sessions.
            Memory can be tiered: working memory (short-term), episodic (experiences), semantic (facts).
            TTL (time-to-live) allows memories to decay naturally over time.
            Semantic search enables retrieval of relevant memories based on meaning, not keywords.
            Memory enables personalization - agents can learn user preferences over time.
            AgentVec is a vector database designed specifically for AI agent memory use cases.
            """
        },
    ]

    print("\n" + "=" * 60)
    print("PHASE 1: Accumulating Knowledge")
    print("=" * 60)

    total_facts = 0
    for session in research_sessions:
        print(f"\n--- Researching: {session['topic']} ---")

        # Extract facts from the content
        facts = extract_facts(session['content'], session['topic'])

        # Store each fact
        for fact in facts:
            memory.remember(
                fact,
                tier=MemoryTier.PROJECT,
                metadata={"topic": session['topic'], "date": datetime.now().isoformat()}
            )
            print(f"  [Stored] {fact[:70]}...")
            total_facts += 1

    stats = memory.get_stats()
    print(f"\n[Total knowledge accumulated: {stats['total_memories']} facts]")

    print("\n" + "=" * 60)
    print("PHASE 2: Querying Accumulated Knowledge")
    print("=" * 60)

    queries = [
        "What is Rust and what is it used for?",
        "How do vector databases work?",
        "What are the benefits of AI agent memory?",
        "What do Rust and vector databases have in common?",  # Cross-topic synthesis
    ]

    for query in queries:
        print(f"\n[Query]: {query}")
        print("-" * 50)

        result = synthesize_knowledge(memory, query)
        if result:
            answer, sources = result
            print(f"[Answer]: {answer}")
            print(f"\n[Sources used: {len(sources)} memories]")
        else:
            print("[No relevant knowledge found]")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"""
Key observations:
1. Knowledge was extracted and stored from multiple research sessions
2. Each fact is stored separately with semantic embeddings
3. Queries retrieve relevant facts across ALL topics
4. Cross-topic queries synthesize information from multiple sources

This pattern enables:
- Research assistants that build domain expertise
- Documentation bots that remember project decisions
- Learning companions that track your study progress
- Any agent that needs to accumulate knowledge over time

Total facts accumulated: {total_facts}
Memory persists at: {MEMORY_PATH}
""")


def run_interactive():
    """Run interactive research accumulator."""
    print("=" * 60)
    print("  RESEARCH ACCUMULATOR - INTERACTIVE")
    print("=" * 60)

    memory = ProjectMemory(str(MEMORY_PATH))
    stats = memory.get_stats()

    if stats['total_memories'] > 0:
        print(f"\nLoaded {stats['total_memories']} existing facts.")
    else:
        print("\nStarting fresh. Feed me knowledge!")

    print("\nCommands:")
    print("  /learn <topic>  - Start learning about a topic")
    print("  /query <question> - Ask a question from accumulated knowledge")
    print("  /facts          - Show all stored facts")
    print("  /stats          - Show memory statistics")
    print("  /clear          - Clear all knowledge")
    print("  /quit           - Exit")
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
                    print("Knowledge saved. Goodbye!")
                    break

                elif cmd == "/learn":
                    if not arg:
                        print("Usage: /learn <topic>")
                        continue

                    print(f"Tell me about '{arg}'. I'll extract and remember the key facts.")
                    print("(Enter your knowledge, then type 'done' on a new line)")

                    lines = []
                    while True:
                        line = input()
                        if line.lower() == 'done':
                            break
                        lines.append(line)

                    content = "\n".join(lines)
                    if content.strip():
                        facts = extract_facts(content, arg)
                        for fact in facts:
                            memory.remember(
                                fact,
                                tier=MemoryTier.PROJECT,
                                metadata={"topic": arg}
                            )
                            print(f"  [Learned] {fact}")
                        print(f"\nStored {len(facts)} facts about '{arg}'")

                elif cmd == "/query":
                    if not arg:
                        print("Usage: /query <question>")
                        continue

                    result = synthesize_knowledge(memory, arg)
                    if result:
                        answer, sources = result
                        print(f"\n{answer}")
                        print(f"\n[Based on {len(sources)} relevant facts]")
                    else:
                        print("I don't have any relevant knowledge to answer that.")

                elif cmd == "/facts":
                    all_facts = memory.recall("", k=100, threshold=0.0)
                    print(f"\nStored facts ({len(all_facts)}):")
                    for i, fact in enumerate(all_facts, 1):
                        topic = fact.metadata.get('topic', 'general')
                        print(f"  {i}. [{topic}] {fact.content[:60]}...")

                elif cmd == "/stats":
                    stats = memory.get_stats()
                    print(f"\nTotal facts: {stats['total_memories']}")
                    for tier, count in stats['tiers'].items():
                        if count > 0:
                            print(f"  {tier}: {count}")

                elif cmd == "/clear":
                    confirm = input("Clear all knowledge? (yes/no): ")
                    if confirm.lower() == "yes":
                        memory.clear()
                        print("Knowledge cleared.")

                else:
                    print(f"Unknown command: {cmd}")

            else:
                # Treat as a query
                result = synthesize_knowledge(memory, user_input)
                if result:
                    answer, sources = result
                    print(f"\n{answer}")
                else:
                    print("I don't have relevant knowledge. Use /learn <topic> to teach me.")

        except KeyboardInterrupt:
            print("\nKnowledge saved. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        run_interactive()
    else:
        run_auto_demo()
