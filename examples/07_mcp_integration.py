#!/usr/bin/env python3
"""
MCP Integration Demo - Claude Code/Desktop Memory

This example demonstrates how to integrate AgentVec with Claude via MCP
(Model Context Protocol). This is the KEY DIFFERENTIATOR - seamless
memory for Claude without any code changes.

What MCP enables:
- Claude automatically has access to memory_remember, memory_recall, memory_forget
- Memory persists across conversations
- No custom code needed - just configuration

This script:
1. Shows how to configure MCP for Claude Code/Desktop
2. Tests the MCP server directly (without Claude)
3. Simulates what Claude sees and can do

Requirements:
    pip install agentvec-mcp

Usage:
    python examples/07_mcp_integration.py           # Test MCP server
    python examples/07_mcp_integration.py --setup   # Show setup instructions
"""

import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path

# Check if agentvec-mcp is available
try:
    from agentvec_memory import ProjectMemory, MemoryTier
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def show_setup_instructions():
    """Show how to set up MCP for Claude."""
    print("=" * 60)
    print("  MCP SETUP INSTRUCTIONS")
    print("  Giving Claude Persistent Memory")
    print("=" * 60)

    print("""
WHAT IS MCP?
------------
MCP (Model Context Protocol) lets you give Claude access to external tools.
AgentVec's MCP server gives Claude memory that persists across conversations.

STEP 1: Install agentvec-mcp
----------------------------
pip install agentvec-mcp

STEP 2: Configure Claude Code
-----------------------------
Create a file called `.mcp.json` in your project root:

{
  "mcpServers": {
    "memory": {
      "command": "agentvec-mcp",
      "env": {
        "AGENTVEC_MEMORY_PATH": "./.agentvec-memory"
      }
    }
  }
}

STEP 3: Configure Claude Desktop (Optional)
-------------------------------------------
Edit your Claude Desktop config (location varies by OS):

macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
Windows: %APPDATA%\\Claude\\claude_desktop_config.json

Add:
{
  "mcpServers": {
    "memory": {
      "command": "agentvec-mcp",
      "env": {
        "AGENTVEC_MEMORY_PATH": "/path/to/your/memory"
      }
    }
  }
}

STEP 4: Restart Claude
----------------------
After configuring, restart Claude Code or Claude Desktop.
Claude will now have access to memory tools!

WHAT CLAUDE CAN DO
------------------
Once configured, Claude can use these tools:

1. memory_remember - Store information
   "Remember that I prefer TypeScript"
   -> Claude calls memory_remember with tier="user"

2. memory_recall - Retrieve relevant memories
   "What do you know about my preferences?"
   -> Claude calls memory_recall with the query

3. memory_forget - Remove memories
   "Forget what I said about Python"
   -> Claude calls memory_forget

4. memory_stats - Check memory status
   "How many things do you remember?"
   -> Claude calls memory_stats

EXAMPLE CONVERSATION
--------------------
You: "Remember that this project uses FastAPI and PostgreSQL"
Claude: [calls memory_remember] "I've stored that information."

[Later, new conversation]

You: "Help me add a new endpoint"
Claude: [calls memory_recall "project framework database"]
        "Since you're using FastAPI with PostgreSQL, here's how..."

The magic: Claude remembered across conversations without you telling it again!
""")


def test_mcp_tools_directly():
    """Test MCP tools without Claude - shows what Claude sees."""
    print("=" * 60)
    print("  TESTING MCP TOOLS DIRECTLY")
    print("  (Simulating what Claude can do)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = os.path.join(tmpdir, "mcp-test")
        memory = ProjectMemory(memory_path)

        print("\nSimulating Claude's memory tools...")
        print("-" * 50)

        # Simulate memory_remember
        print("\n[Tool: memory_remember]")
        print("Input: {content: 'User prefers dark mode', tier: 'user'}")

        memory.remember("User prefers dark mode", tier=MemoryTier.USER)
        print("Output: Memory stored successfully")

        # More memories
        memories_to_store = [
            ("User's name is Alex", MemoryTier.USER),
            ("Project uses React with TypeScript", MemoryTier.PROJECT),
            ("Currently debugging authentication", MemoryTier.SESSION),
            ("Looking at file auth.tsx", MemoryTier.WORKING),
        ]

        print("\n[Storing more memories...]")
        for content, tier in memories_to_store:
            memory.remember(content, tier=tier)
            print(f"  [{tier.value}] {content}")

        # Simulate memory_recall
        print("\n" + "-" * 50)
        print("[Tool: memory_recall]")
        print("Input: {query: 'What are the user preferences?'}")

        results = memory.recall("What are the user preferences?", k=5, threshold=0.3)
        print(f"Output: Found {len(results)} relevant memories")
        for r in results:
            print(f"  - [{r.tier.value}] {r.content} (score: {r.score:.2f})")

        # Another recall
        print("\n[Tool: memory_recall]")
        print("Input: {query: 'What project am I working on?'}")

        results = memory.recall("What project am I working on?", k=5, threshold=0.3)
        print(f"Output: Found {len(results)} relevant memories")
        for r in results:
            print(f"  - [{r.tier.value}] {r.content} (score: {r.score:.2f})")

        # Simulate memory_stats
        print("\n" + "-" * 50)
        print("[Tool: memory_stats]")
        print("Input: {}")

        stats = memory.get_stats()
        print(f"Output: {json.dumps(stats, indent=2)}")

        # Simulate memory_forget
        print("\n" + "-" * 50)
        print("[Tool: memory_forget]")
        print("Input: {query: 'debugging authentication'}")

        removed = memory.forget("debugging authentication", threshold=0.7)
        print(f"Output: Removed {removed} memories")

        # Final stats
        stats = memory.get_stats()
        print(f"\nFinal memory count: {stats['total_memories']}")


def simulate_conversation():
    """Simulate a multi-turn conversation with memory."""
    print("\n" + "=" * 60)
    print("  SIMULATED CONVERSATION WITH MEMORY")
    print("  (What happens when Claude has MCP memory)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = os.path.join(tmpdir, "conversation")
        memory = ProjectMemory(memory_path)

        conversation = [
            {
                "session": 1,
                "turns": [
                    {
                        "user": "Hi! I'm Sarah, a frontend developer.",
                        "claude_action": "memory_remember",
                        "claude_args": {"content": "User's name is Sarah. Sarah is a frontend developer.", "tier": "user"},
                        "claude_response": "Nice to meet you, Sarah! I'll remember that you're a frontend developer. How can I help you today?"
                    },
                    {
                        "user": "I'm starting a new React project with Tailwind CSS.",
                        "claude_action": "memory_remember",
                        "claude_args": {"content": "Sarah's current project uses React with Tailwind CSS.", "tier": "project"},
                        "claude_response": "Great choice! React with Tailwind is a powerful combination. I've noted your project setup. What would you like to work on first?"
                    },
                    {
                        "user": "I always want to use functional components, never class components.",
                        "claude_action": "memory_remember",
                        "claude_args": {"content": "Sarah prefers functional components over class components in React.", "tier": "user"},
                        "claude_response": "Understood! I'll always suggest functional components with hooks. That's the modern React way anyway!"
                    },
                ]
            },
            {
                "session": 2,
                "turns": [
                    {
                        "user": "Hey, can you help me create a button component?",
                        "claude_action": "memory_recall",
                        "claude_args": {"query": "user preferences project setup React"},
                        "claude_response": "Hi Sarah! I remember you're working on a React project with Tailwind CSS, and you prefer functional components. Here's a button component:\n\n```tsx\nconst Button = ({ children, onClick }) => (\n  <button\n    className=\"px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600\"\n    onClick={onClick}\n  >\n    {children}\n  </button>\n);\n```\n\nIt uses Tailwind for styling, just like your project setup!"
                    },
                    {
                        "user": "What's my name again?",
                        "claude_action": "memory_recall",
                        "claude_args": {"query": "user name"},
                        "claude_response": "Your name is Sarah! You're a frontend developer working on a React + Tailwind project."
                    },
                ]
            }
        ]

        for session_data in conversation:
            session_num = session_data["session"]
            print(f"\n{'='*50}")
            print(f"SESSION {session_num}")
            print(f"{'='*50}")

            if session_num == 2:
                print("[Memory persisted from Session 1]")
                results = memory.recall("Sarah React project", k=5, threshold=0.2)
                print(f"[Loaded {len(results)} memories]")

            for turn in session_data["turns"]:
                print(f"\nUser: {turn['user']}")

                # Execute Claude's action
                action = turn["claude_action"]
                args = turn["claude_args"]

                if action == "memory_remember":
                    memory.remember(args["content"], tier=MemoryTier(args["tier"]))
                    print(f"  [Claude calls {action}: stored in {args['tier']}]")
                elif action == "memory_recall":
                    results = memory.recall(args["query"], k=5, threshold=0.3)
                    print(f"  [Claude calls {action}: found {len(results)} memories]")

                print(f"\nClaude: {turn['claude_response']}")

        print("\n" + "=" * 50)
        print("KEY INSIGHT")
        print("=" * 50)
        print("""
Notice what happened:
1. Session 1: Claude stored facts about Sarah
2. Session 2: NEW SESSION, but Claude still knew:
   - Sarah's name
   - Her project (React + Tailwind)
   - Her preference (functional components)

This is the power of MCP integration:
- No code changes needed
- Memory persists automatically
- Claude becomes truly personalized
""")


def show_mcp_config_file():
    """Generate an example .mcp.json file."""
    print("\n" + "=" * 60)
    print("  EXAMPLE .mcp.json FILE")
    print("=" * 60)

    config = {
        "mcpServers": {
            "memory": {
                "command": "agentvec-mcp",
                "env": {
                    "AGENTVEC_MEMORY_PATH": "./.agentvec-memory"
                }
            }
        }
    }

    print("\nCreate this file as `.mcp.json` in your project root:\n")
    print(json.dumps(config, indent=2))

    # Also create the file
    mcp_path = Path(__file__).parent.parent / ".mcp.json"
    if not mcp_path.exists():
        with open(mcp_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n[Created: {mcp_path}]")
    else:
        print(f"\n[Already exists: {mcp_path}]")


def main():
    print("=" * 60)
    print("  AGENTVEC MCP INTEGRATION")
    print("  Persistent Memory for Claude")
    print("=" * 60)

    if not MCP_AVAILABLE:
        print("\nError: agentvec-memory not installed")
        print("Install with: pip install agentvec-memory")
        print("\nShowing setup instructions anyway...\n")
        show_setup_instructions()
        return

    if "--setup" in sys.argv:
        show_setup_instructions()
        return

    print("\nThis demo shows how AgentVec integrates with Claude via MCP.")
    print("No LLM calls needed - we simulate what Claude would do.\n")

    # Run demos
    test_mcp_tools_directly()
    simulate_conversation()
    show_mcp_config_file()
    show_setup_instructions()


if __name__ == "__main__":
    main()
