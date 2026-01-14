#!/usr/bin/env python3
"""
Learning Preferences - The Agent That Adapts To You

This example demonstrates how an AI agent learns and applies user preferences.
Tell it your preferences once, and it automatically applies them to all future
requests - even across sessions.

Requirements:
    pip install agentvec-memory requests python-dotenv

Usage:
    python examples/03_learning_preferences.py

Watch how:
1. You state preferences ("I prefer TypeScript", "Use 2-space indentation")
2. The agent stores them in long-term memory
3. Future code/explanations automatically follow your preferences
4. Corrections update the preferences
"""

import os
import sys
import json
import tempfile
import requests
from pathlib import Path
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
            "max_tokens": 1500,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def detect_preference(user_message: str) -> str | None:
    """Detect if user is stating a preference."""
    detection_prompt = f"""Analyze if this message contains a user preference or correction.

User message: "{user_message}"

If this is a preference statement or correction, extract it as a clear preference.
Return ONLY the preference as a simple statement, or "NONE" if no preference found.

Examples:
- "I prefer tabs over spaces" -> "User prefers tabs over spaces for indentation"
- "Use TypeScript please" -> "User prefers TypeScript over JavaScript"
- "Actually, make it async" -> "User prefers async/await pattern"
- "Can you help me?" -> "NONE"
- "Write a function" -> "NONE"

Response (preference or NONE):"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "anthropic/claude-3.5-haiku",
                "messages": [{"role": "user", "content": detection_prompt}],
                "max_tokens": 100,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        if result.upper() == "NONE" or len(result) < 5:
            return None
        return result
    except Exception:
        return None


def get_preferences_context(memory: ProjectMemory) -> str:
    """Get all stored preferences."""
    # Search for preference-related memories
    results = memory.recall(
        "user preferences coding style formatting",
        k=10,
        threshold=0.3,
        tiers=[MemoryTier.USER]
    )
    if not results:
        return ""

    prefs = ["[User Preferences - ALWAYS follow these:]"]
    for mem in results:
        prefs.append(f"- {mem.content}")
    return "\n".join(prefs)


def run_demo(auto_mode: bool = False):
    """Run the interactive preference learning demo."""
    def wait(msg: str):
        if auto_mode:
            print(msg)
        else:
            input(msg)

    print("=" * 60)
    print("  LEARNING PREFERENCES DEMO")
    print("  The Agent That Adapts To You")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = ProjectMemory(os.path.join(tmpdir, "preferences"))

        # Scripted demonstration
        demo_script = [
            {
                "phase": "PHASE 1: Establishing Preferences",
                "turns": [
                    ("I prefer TypeScript over JavaScript for all code examples.", True),
                    ("Always use 2-space indentation, never tabs.", True),
                    ("I like functional programming style - avoid classes when possible.", True),
                ],
            },
            {
                "phase": "PHASE 2: Testing Preference Application",
                "turns": [
                    ("Write a function that filters even numbers from an array.", False),
                    ("Now write a debounce utility function.", False),
                ],
            },
            {
                "phase": "PHASE 3: Updating Preferences",
                "turns": [
                    ("Actually, I changed my mind - use 4-space indentation instead.", True),
                    ("Write a simple memoization function.", False),
                ],
            },
        ]

        base_system = """You are a helpful coding assistant.
You adapt to user preferences and consistently apply them to all code you write.
When writing code, follow any stated preferences exactly."""

        messages = []

        for phase_data in demo_script:
            print(f"\n{'=' * 60}")
            print(f"  {phase_data['phase']}")
            print("=" * 60)

            for user_msg, is_preference_statement in phase_data["turns"]:
                print(f"\n[User]: {user_msg}")

                # Detect and store preference
                if is_preference_statement:
                    preference = detect_preference(user_msg)
                    if preference:
                        # Remove old conflicting preference if updating
                        if "indent" in preference.lower():
                            memory.forget("indentation", threshold=0.6, tiers=[MemoryTier.USER])
                        memory.remember(preference, tier=MemoryTier.USER)
                        print(f"  [Stored Preference: {preference}]")

                # Build system prompt with preferences
                prefs_context = get_preferences_context(memory)
                if prefs_context:
                    system = f"{base_system}\n\n{prefs_context}"
                else:
                    system = base_system

                # Get response
                messages.append({"role": "user", "content": user_msg})
                response = call_llm(messages[-6:], system)
                messages.append({"role": "assistant", "content": response})

                print(f"\n[Assistant]: {response}")

                wait("\n  Press Enter to continue...")

        # Show final state
        print("\n" + "=" * 60)
        print("  FINAL PREFERENCE STATE")
        print("=" * 60)

        all_prefs = memory.recall("preferences", k=20, threshold=0.2, tiers=[MemoryTier.USER])
        print("\nStored preferences:")
        for pref in all_prefs:
            print(f"  - {pref.content}")

        print("\n" + "=" * 60)
        print("  KEY TAKEAWAYS")
        print("=" * 60)
        print("""
1. Preferences are automatically detected from natural language
2. They're stored in USER tier (1-year TTL)
3. All future responses respect these preferences
4. Preferences can be updated - old ones are replaced
5. Across sessions, preferences persist (in real usage)

This enables truly personalized AI assistants that learn
your style and consistently apply it.
""")


def run_interactive():
    """Run interactive mode for hands-on experimentation."""
    print("=" * 60)
    print("  INTERACTIVE PREFERENCE LEARNING")
    print("=" * 60)

    memory_path = Path(__file__).parent / ".learning_preferences_memory"
    memory = ProjectMemory(str(memory_path))

    stats = memory.get_stats()
    if stats['total_memories'] > 0:
        print(f"\nLoaded {stats['total_memories']} existing preferences.")
        prefs = memory.recall("preferences", k=10, threshold=0.2, tiers=[MemoryTier.USER])
        if prefs:
            print("Current preferences:")
            for p in prefs:
                print(f"  - {p.content}")
    else:
        print("\nNo preferences stored yet. Tell me your preferences!")

    print("\nCommands: /prefs (show all), /clear (reset), /quit (exit)")
    print("-" * 60)

    messages = []
    base_system = """You are a helpful coding assistant.
You adapt to user preferences and consistently apply them to all code you write."""

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input == "/quit":
                print("Preferences saved. Goodbye!")
                break
            if user_input == "/prefs":
                prefs = memory.recall("preferences", k=20, threshold=0.2, tiers=[MemoryTier.USER])
                print("\nStored preferences:")
                for p in prefs:
                    print(f"  - {p.content}")
                continue
            if user_input == "/clear":
                memory.clear(tiers=[MemoryTier.USER])
                print("Preferences cleared.")
                continue

            # Detect preference
            preference = detect_preference(user_input)
            if preference:
                memory.remember(preference, tier=MemoryTier.USER)
                print(f"  [Learned: {preference}]")

            # Build context
            prefs_context = get_preferences_context(memory)
            system = f"{base_system}\n\n{prefs_context}" if prefs_context else base_system

            messages.append({"role": "user", "content": user_input})
            response = call_llm(messages[-8:], system)
            messages.append({"role": "assistant", "content": response})

            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nPreferences saved. Goodbye!")
            break


if __name__ == "__main__":
    import sys
    if "--interactive" in sys.argv:
        run_interactive()
    else:
        auto = "--auto" in sys.argv
        run_demo(auto_mode=auto)
