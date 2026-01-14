#!/usr/bin/env python3
"""
Persistent Assistant - Memory That Survives Restarts

This example demonstrates the SQLite-like persistence of AgentVec.
Run the script, chat with the assistant, close it, run it again -
the assistant remembers everything from previous sessions.

Requirements:
    pip install agentvec-memory requests python-dotenv

Usage:
    python examples/02_persistent_assistant.py

    First run: Tell it about yourself and your project
    Close the script (Ctrl+C)
    Run it again: It remembers everything!
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in environment")
    print("Create a .env file with: OPENROUTER_API_KEY=your-key-here")
    sys.exit(1)

try:
    from agentvec_memory import ProjectMemory, MemoryTier
except ImportError:
    print("Error: agentvec-memory not installed")
    print("Install with: pip install agentvec-memory")
    sys.exit(1)


# Persistent memory location - this is the key!
# Unlike a temp directory, this persists across runs
MEMORY_PATH = Path(__file__).parent / ".persistent_assistant_memory"


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


def extract_facts_to_remember(user_message: str, assistant_response: str) -> list[tuple[str, MemoryTier]]:
    """Use LLM to extract facts worth remembering from the conversation."""
    extraction_prompt = """Analyze this conversation turn and extract any facts worth remembering.
Return a JSON array of objects with "fact" and "tier" fields.

Tiers:
- "user": Long-term user preferences, personal info (name, role, preferences)
- "project": Project-specific info (tech stack, architecture, goals)
- "session": Current task or temporary context

Only extract concrete, useful facts. Return empty array [] if nothing worth remembering.

Example output:
[
  {"fact": "User's name is Sarah", "tier": "user"},
  {"fact": "Project uses React with TypeScript", "tier": "project"}
]

User said: {user}
Assistant said: {assistant}

Return ONLY valid JSON array, no other text:"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "anthropic/claude-3.5-haiku",
                "messages": [{"role": "user", "content": extraction_prompt.format(
                    user=user_message,
                    assistant=assistant_response
                )}],
                "max_tokens": 500,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Parse JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        facts = json.loads(content)
        tier_map = {
            "user": MemoryTier.USER,
            "project": MemoryTier.PROJECT,
            "session": MemoryTier.SESSION,
            "working": MemoryTier.WORKING,
        }
        return [(f["fact"], tier_map.get(f["tier"], MemoryTier.PROJECT)) for f in facts]
    except Exception:
        return []


def get_memory_context(memory: ProjectMemory, query: str) -> str:
    """Retrieve relevant memories for context."""
    memories = memory.recall(query, k=8, threshold=0.35)
    if not memories:
        return ""

    context_lines = ["[Retrieved from memory:]"]
    for mem in memories:
        tier_label = mem.tier.value.upper()
        context_lines.append(f"- [{tier_label}] {mem.content}")
    return "\n".join(context_lines)


def show_memory_stats(memory: ProjectMemory):
    """Display current memory statistics."""
    stats = memory.get_stats()
    print("\n--- Memory Stats ---")
    print(f"Total memories: {stats['total_memories']}")
    for tier, count in stats['tiers'].items():
        if count > 0:
            print(f"  {tier}: {count}")
    print("-------------------\n")


def main():
    print("=" * 60)
    print("  PERSISTENT ASSISTANT")
    print("  Memory that survives restarts")
    print("=" * 60)

    # Initialize memory (creates or opens existing)
    memory = ProjectMemory(str(MEMORY_PATH))
    stats = memory.get_stats()

    if stats['total_memories'] > 0:
        print(f"\nWelcome back! I remember {stats['total_memories']} things about you.")
        print("(Memory loaded from previous sessions)")

        # Show what we remember
        print("\nLet me recall what I know...")
        memories = memory.recall("user preferences and project info", k=5, threshold=0.2)
        if memories:
            for mem in memories[:3]:
                print(f"  - {mem.content}")
    else:
        print("\nThis is our first session! I'll remember what you tell me.")
        print(f"Memory will be saved to: {MEMORY_PATH}")

    print("\nCommands:")
    print("  /stats  - Show memory statistics")
    print("  /recall <query> - Search memories")
    print("  /forget <query> - Forget matching memories")
    print("  /clear  - Clear all memories")
    print("  /quit   - Exit (memory persists!)")
    print("\n" + "-" * 60)

    messages = []
    base_system = """You are a helpful assistant with persistent memory.
You remember information about the user across sessions.
Be concise but personable. Reference what you know about the user when relevant."""

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]

                if cmd == "/quit":
                    print("\nGoodbye! Your memories are saved. See you next time!")
                    break

                elif cmd == "/stats":
                    show_memory_stats(memory)
                    continue

                elif cmd == "/recall":
                    query = user_input[7:].strip() or "everything"
                    print(f"\nSearching for: {query}")
                    results = memory.recall(query, k=10, threshold=0.2)
                    if results:
                        for mem in results:
                            print(f"  [{mem.tier.value}] {mem.content} (score: {mem.score:.2f})")
                    else:
                        print("  No matching memories found.")
                    continue

                elif cmd == "/forget":
                    query = user_input[7:].strip()
                    if query:
                        removed = memory.forget(query, threshold=0.7)
                        print(f"Forgot {removed} memories matching '{query}'")
                    else:
                        print("Usage: /forget <what to forget>")
                    continue

                elif cmd == "/clear":
                    confirm = input("Are you sure? Type 'yes' to clear all memories: ")
                    if confirm.lower() == "yes":
                        memory.clear()
                        print("All memories cleared.")
                    continue

                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Get relevant context from memory
            context = get_memory_context(memory, user_input)
            if context:
                system_prompt = f"{base_system}\n\n{context}"
            else:
                system_prompt = base_system

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Get response
            print("\nAssistant: ", end="", flush=True)
            response = call_llm(messages[-10:], system_prompt)  # Keep last 10 turns
            print(response)

            messages.append({"role": "assistant", "content": response})

            # Extract and store facts
            facts = extract_facts_to_remember(user_input, response)
            if facts:
                for fact, tier in facts:
                    memory.remember(fact, tier=tier)
                    print(f"  [Remembered: {fact}]")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Your memories are saved. See you next time!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def auto_demo():
    """Automated demo showing persistence across sessions."""
    import shutil

    print("=" * 60)
    print("  PERSISTENT ASSISTANT - AUTO DEMO")
    print("  Demonstrating memory persistence across sessions")
    print("=" * 60)

    # Clean start
    if MEMORY_PATH.exists():
        shutil.rmtree(MEMORY_PATH)

    # SESSION 1: Introduce ourselves
    print("\n" + "=" * 60)
    print("  SESSION 1: First meeting")
    print("=" * 60)

    memory = ProjectMemory(str(MEMORY_PATH))
    messages = []
    base_system = """You are a helpful assistant with persistent memory.
You remember information about the user across sessions.
Be concise but personable."""

    # Session 1 conversations with explicit facts to store
    session1_data = [
        (
            "Hi! My name is Casey and I'm a data scientist.",
            "User's name is Casey. Casey is a data scientist.",
            MemoryTier.USER
        ),
        (
            "I'm working on a machine learning project using PyTorch.",
            "User is working on a machine learning project using PyTorch framework.",
            MemoryTier.PROJECT
        ),
        (
            "I prefer code examples over lengthy explanations.",
            "User prefers code examples over lengthy explanations. Keep responses concise.",
            MemoryTier.USER
        ),
    ]

    for user_input, fact_to_store, tier in session1_data:
        print(f"\nYou: {user_input}")

        context = get_memory_context(memory, user_input)
        system = f"{base_system}\n\n{context}" if context else base_system

        messages.append({"role": "user", "content": user_input})
        response = call_llm(messages[-10:], system)
        print(f"Assistant: {response}")
        messages.append({"role": "assistant", "content": response})

        # Store the fact explicitly (for demo reliability)
        memory.remember(fact_to_store, tier=tier)
        print(f"  [Stored in {tier.value}: {fact_to_store}]")

    stats = memory.get_stats()
    print(f"\n[Session 1 ended - {stats['total_memories']} memories stored]")
    print("[Closing the application...]\n")

    # Clear in-memory state (simulating app restart)
    del memory
    del messages

    # SESSION 2: Return and test memory
    print("=" * 60)
    print("  SESSION 2: Returning later (memory should persist!)")
    print("=" * 60)

    memory = ProjectMemory(str(MEMORY_PATH))
    stats = memory.get_stats()
    print(f"\n[Loaded {stats['total_memories']} memories from disk]")

    # Show what we remember
    print("\nRecalling what we know about the user...")
    all_memories = memory.recall("user information preferences project", k=10, threshold=0.2)
    if all_memories:
        for mem in all_memories:
            print(f"  [{mem.tier.value}] {mem.content}")

    messages = []
    session2_inputs = [
        "Hey, I'm back! What do you remember about me?",
        "Can you help me write a training loop?",
    ]

    print()
    for user_input in session2_inputs:
        print(f"\nYou: {user_input}")

        context = get_memory_context(memory, user_input)
        system = f"{base_system}\n\n{context}" if context else base_system

        messages.append({"role": "user", "content": user_input})
        response = call_llm(messages[-10:], system)
        print(f"Assistant: {response}")
        messages.append({"role": "assistant", "content": response})

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("""
Key observations:
1. Session 1: We told the assistant about ourselves
2. The app "closed" (memory object destroyed)
3. Session 2: We reopened - memories loaded from disk!
4. The assistant remembered our name, role, and preferences

This is the SQLite-like persistence of AgentVec.
Your agent's memory survives restarts.
""")


if __name__ == "__main__":
    import sys
    if "--auto" in sys.argv:
        auto_demo()
    else:
        main()
