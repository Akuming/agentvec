#!/usr/bin/env python3
"""
Memory vs No Memory - The Killer Demo

This example demonstrates WHY memory matters for AI agents.
Two parallel conversations with the same LLM - one with memory, one without.
Watch how memory transforms a generic chatbot into a personalized assistant.

Requirements:
    pip install agentvec-memory requests python-dotenv

Usage:
    python examples/01_memory_vs_no_memory.py
"""

import os
import sys
import json
import tempfile
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in environment")
    print("Create a .env file with: OPENROUTER_API_KEY=your-key-here")
    sys.exit(1)

# Check for agentvec-memory
try:
    from agentvec_memory import ProjectMemory, MemoryTier
except ImportError:
    print("Error: agentvec-memory not installed")
    print("Install with: pip install agentvec-memory")
    sys.exit(1)


def call_llm(messages: list, system_prompt: str = None) -> str:
    """Call LLM via OpenRouter API."""
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "anthropic/claude-3.5-haiku",  # Fast and cheap for demos
            "messages": full_messages,
            "max_tokens": 500,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_memory_context(memory: ProjectMemory, query: str) -> str:
    """Retrieve relevant memories and format as context."""
    memories = memory.recall(query, k=5, threshold=0.3)
    if not memories:
        return ""

    context_lines = ["[Relevant memories about this user:]"]
    for mem in memories:
        context_lines.append(f"- {mem.content}")
    return "\n".join(context_lines)


def run_conversation_without_memory(conversation: list[tuple[str, str]]) -> list[str]:
    """Run conversation without any memory - just raw LLM."""
    print("\n" + "=" * 60)
    print("ASSISTANT WITHOUT MEMORY")
    print("=" * 60)

    messages = []
    responses = []

    system = "You are a helpful assistant. Be concise."

    for i, (user_msg, _) in enumerate(conversation):
        print(f"\n[Turn {i+1}]")
        print(f"User: {user_msg}")

        messages.append({"role": "user", "content": user_msg})
        response = call_llm(messages, system_prompt=system)
        messages.append({"role": "assistant", "content": response})
        responses.append(response)

        print(f"Assistant: {response}")

    return responses


def run_conversation_with_memory(
    conversation: list[tuple[str, str]],
    memory: ProjectMemory
) -> list[str]:
    """Run conversation with memory - stores facts, retrieves context."""
    print("\n" + "=" * 60)
    print("ASSISTANT WITH MEMORY")
    print("=" * 60)

    messages = []
    responses = []

    base_system = "You are a helpful assistant. Be concise."

    for i, (user_msg, fact_to_store) in enumerate(conversation):
        print(f"\n[Turn {i+1}]")
        print(f"User: {user_msg}")

        # Store any facts from this turn
        if fact_to_store:
            memory.remember(fact_to_store, tier=MemoryTier.USER)
            print(f"  [Stored: {fact_to_store}]")

        # Retrieve relevant context
        context = get_memory_context(memory, user_msg)
        if context:
            system = f"{base_system}\n\n{context}"
            print(f"  [Retrieved context from memory]")
        else:
            system = base_system

        messages.append({"role": "user", "content": user_msg})
        response = call_llm(messages, system_prompt=system)
        messages.append({"role": "assistant", "content": response})
        responses.append(response)

        print(f"Assistant: {response}")

    return responses


def main(auto_mode: bool = False):
    def wait(msg: str):
        if auto_mode:
            print(msg)
        else:
            input(msg)

    print("=" * 60)
    print("  MEMORY VS NO MEMORY - THE KILLER DEMO")
    print("  Simulating a NEW SESSION after previous context")
    print("=" * 60)

    print("""
SCENARIO:
- Yesterday, you told an assistant about yourself
- Today, you come back and ask for help
- WITHOUT memory: The assistant knows nothing about you
- WITH memory: The assistant remembers everything
""")

    # PHASE 1: Previous session - establish context (both versions get this)
    previous_session = [
        ("Hi! My name is Alex and I'm a backend developer who works with Python.",
         "User's name is Alex. Alex is a backend developer who works with Python."),
        ("I really hate long verbose explanations. Keep things brief.",
         "User strongly prefers brief, concise explanations. Dislikes verbose responses."),
        ("I'm working on a Django REST API project.",
         "User is working on a Django REST API project."),
    ]

    # PHASE 2: New session - test if context is remembered
    new_session_questions = [
        "Hey! Can you help me add authentication to my project?",
        "What's my name again?",
    ]

    # Create temporary memory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ProjectMemory(os.path.join(tmpdir, "demo-memory"))

        # First, populate memory with previous session data
        print("=" * 60)
        print("PHASE 1: Previous Session (yesterday)")
        print("=" * 60)
        print("\nStoring context from previous conversation...")

        for user_msg, fact in previous_session:
            print(f"  User said: \"{user_msg[:50]}...\"")
            memory.remember(fact, tier=MemoryTier.USER)
            print(f"  [Stored: {fact}]")

        print(f"\nMemory now contains {memory.get_stats()['total_memories']} facts.")
        wait("\nPress Enter to simulate a NEW SESSION (next day)...")

        # NOW: New session - compare with and without memory
        print("\n" + "=" * 60)
        print("PHASE 2: New Session (today) - WITHOUT MEMORY")
        print("=" * 60)
        print("\n[No memory loaded - assistant starts fresh]\n")

        system_no_memory = "You are a helpful assistant. Be concise."
        messages_no_memory = []
        no_memory_responses = []

        for question in new_session_questions:
            print(f"User: {question}")
            messages_no_memory.append({"role": "user", "content": question})
            response = call_llm(messages_no_memory, system_no_memory)
            messages_no_memory.append({"role": "assistant", "content": response})
            no_memory_responses.append(response)
            print(f"Assistant: {response}\n")

        wait("Press Enter to see the same questions WITH MEMORY...")

        print("\n" + "=" * 60)
        print("PHASE 2: New Session (today) - WITH MEMORY")
        print("=" * 60)

        # Retrieve and show memory context
        context = get_memory_context(memory, "what do I know about the user")
        print(f"\n[Memory loaded from previous session:]")
        if context:
            for line in context.split('\n')[1:]:  # Skip header
                print(f"  {line}")
        else:
            print("  (no memories found)")
        print()

        system_with_memory = f"You are a helpful assistant. Be concise.\n\n{context}"
        messages_with_memory = []
        with_memory_responses = []

        for question in new_session_questions:
            print(f"User: {question}")
            messages_with_memory.append({"role": "user", "content": question})
            response = call_llm(messages_with_memory, system_with_memory)
            messages_with_memory.append({"role": "assistant", "content": response})
            with_memory_responses.append(response)
            print(f"Assistant: {response}\n")

        # Summary comparison
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        print("\nQuestion: \"Hey! Can you help me add authentication to my project?\"")
        print("-" * 60)
        print(f"WITHOUT memory:\n  {no_memory_responses[0][:200]}...")
        print(f"\nWITH memory:\n  {with_memory_responses[0][:200]}...")

        print("\n" + "-" * 60)
        print("Question: \"What's my name again?\"")
        print("-" * 60)
        print(f"WITHOUT memory:\n  {no_memory_responses[1]}")
        print(f"\nWITH memory:\n  {with_memory_responses[1]}")

        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        print("""
This is the real value of AgentVec:

WITHOUT memory (new session):
- Assistant has NO context from previous conversations
- Asks generic clarifying questions
- Doesn't know user's name, project, or preferences

WITH memory (context persisted):
- Knows user is Alex, a Python/Django developer
- Gives Django-specific authentication advice
- Remembers to be concise (user preference)

Memory enables CONTINUITY across sessions.
This is what makes AI assistants truly personal.
""")


if __name__ == "__main__":
    import sys
    auto = "--auto" in sys.argv
    main(auto_mode=auto)
