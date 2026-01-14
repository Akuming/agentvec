"""
Memory CLI - Manage project memory from the command line.

Usage:
    agentvec-memory remember "User prefers dark mode" --tier user
    agentvec-memory recall "user preferences"
    agentvec-memory forget "dark mode"
    agentvec-memory stats
"""

import sys
import click
from pathlib import Path
from typing import Optional

from .memory import ProjectMemory, MemoryTier


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RED = "\033[31m"


def color(text: str, color_code: str) -> str:
    """Apply color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color_code}{text}{Colors.RESET}"
    return text


def get_default_memory_path() -> str:
    """Get default memory path in current directory."""
    return str(Path.cwd() / ".agentvec-memory")


def format_tier(tier: MemoryTier) -> str:
    """Format tier name with color."""
    tier_colors = {
        MemoryTier.WORKING: Colors.RED,
        MemoryTier.SESSION: Colors.YELLOW,
        MemoryTier.PROJECT: Colors.GREEN,
        MemoryTier.USER: Colors.CYAN,
    }
    return color(tier.value, tier_colors.get(tier, Colors.WHITE))


def format_score(score: float) -> str:
    """Format similarity score with color."""
    if score >= 0.7:
        return color(f"{score:.3f}", Colors.GREEN)
    elif score >= 0.5:
        return color(f"{score:.3f}", Colors.YELLOW)
    else:
        return color(f"{score:.3f}", Colors.DIM)


def format_ttl(seconds: float) -> str:
    """Format TTL as human-readable string."""
    if seconds <= 0:
        return color("expired", Colors.RED)
    elif seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Project Memory - Persistent tiered memory for AI agents.

    \b
    Examples:
        agentvec-memory remember "API key is in .env file"
        agentvec-memory remember "User prefers dark mode" --tier user
        agentvec-memory recall "configuration"
        agentvec-memory forget "API key"
    """
    pass


@cli.command()
@click.argument("content")
@click.option(
    "--memory", "-m",
    default=None,
    help="Memory database path (default: .agentvec-memory)"
)
@click.option(
    "--tier", "-t",
    type=click.Choice(["working", "session", "project", "user"]),
    default="project",
    help="Memory tier (default: project)"
)
@click.option(
    "--ttl",
    type=int,
    default=None,
    help="Time-to-live in seconds (uses tier default if not set)"
)
def remember(content: str, memory: Optional[str], tier: str, ttl: Optional[int]):
    """
    Store a memory.

    \b
    Examples:
        agentvec-memory remember "User prefers dark mode"
        agentvec-memory remember "Working on auth" --tier session
        agentvec-memory remember "Temp note" --tier working --ttl 60
    """
    memory_path = memory or get_default_memory_path()
    memory_tier = MemoryTier(tier)

    try:
        mem = ProjectMemory(memory_path)
        memory_id = mem.remember(content, tier=memory_tier, ttl=ttl)

        click.echo(f"{color('Remembered:', Colors.GREEN)} {content[:60]}{'...' if len(content) > 60 else ''}")
        click.echo(f"  Tier: {format_tier(memory_tier)}")
        click.echo(f"  TTL:  {format_ttl(ttl or memory_tier.default_ttl)}")
        click.echo(f"  ID:   {color(memory_id[:8] + '...', Colors.DIM)}")

    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--memory", "-m",
    default=None,
    help="Memory database path"
)
@click.option(
    "--results", "-k",
    default=5,
    help="Number of results (default: 5)"
)
@click.option(
    "--threshold", "-t",
    default=0.3,
    help="Minimum similarity score (default: 0.3)"
)
@click.option(
    "--tier",
    type=click.Choice(["working", "session", "project", "user"]),
    multiple=True,
    help="Limit to specific tiers"
)
@click.option(
    "--include-expired",
    is_flag=True,
    help="Include expired memories"
)
def recall(
    query: str,
    memory: Optional[str],
    results: int,
    threshold: float,
    tier: tuple,
    include_expired: bool,
):
    """
    Retrieve relevant memories.

    \b
    Examples:
        agentvec-memory recall "user preferences"
        agentvec-memory recall "configuration" -k 10
        agentvec-memory recall "settings" --tier user --tier project
    """
    memory_path = memory or get_default_memory_path()

    if not Path(memory_path).exists():
        click.echo(color(f"Error: Memory not found at {memory_path}", Colors.RED), err=True)
        click.echo("Run 'agentvec-memory remember <content>' first.", err=True)
        sys.exit(1)

    try:
        mem = ProjectMemory(memory_path)
        tiers = [MemoryTier(t) for t in tier] if tier else None

        memories = mem.recall(
            query,
            k=results,
            threshold=threshold,
            tiers=tiers,
            include_expired=include_expired,
        )

        click.echo(f"\n{color('Query:', Colors.BOLD)} {color(query, Colors.CYAN)}")
        click.echo(f"{'=' * 50}\n")

        if not memories:
            click.echo(color("No memories found.", Colors.YELLOW))
            click.echo("Try a different query or lower the threshold with -t 0.2")
            return

        for i, m in enumerate(memories, 1):
            score_str = format_score(m.score)
            tier_str = format_tier(m.tier)
            ttl_str = format_ttl(m.ttl_remaining)

            click.echo(f"{color(str(i) + '.', Colors.BOLD)} [{tier_str}] (score: {score_str}, ttl: {ttl_str})")

            # Truncate long content
            content = m.content
            if len(content) > 100:
                content = content[:100] + "..."
            click.echo(f"   {content}")
            click.echo()

        click.echo(f"{color(f'Found {len(memories)} memories', Colors.DIM)}")

    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--memory", "-m",
    default=None,
    help="Memory database path"
)
@click.option(
    "--threshold", "-t",
    default=0.8,
    help="Similarity threshold for deletion (default: 0.8)"
)
@click.option(
    "--tier",
    type=click.Choice(["working", "session", "project", "user"]),
    multiple=True,
    help="Limit to specific tiers"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without deleting"
)
def forget(
    query: str,
    memory: Optional[str],
    threshold: float,
    tier: tuple,
    dry_run: bool,
):
    """
    Remove memories similar to query.

    \b
    Examples:
        agentvec-memory forget "API key"
        agentvec-memory forget "old config" --threshold 0.9
        agentvec-memory forget "temp" --tier working --dry-run
    """
    memory_path = memory or get_default_memory_path()

    if not Path(memory_path).exists():
        click.echo(color(f"Error: Memory not found at {memory_path}", Colors.RED), err=True)
        sys.exit(1)

    try:
        mem = ProjectMemory(memory_path)
        tiers = [MemoryTier(t) for t in tier] if tier else None

        if dry_run:
            # Show what would be deleted
            matches = mem.recall(
                query,
                k=100,
                threshold=threshold,
                tiers=tiers,
                include_expired=True,
            )

            if not matches:
                click.echo(color("No matching memories found.", Colors.YELLOW))
                return

            click.echo(f"\n{color('Would delete:', Colors.YELLOW)}")
            for m in matches:
                content = m.content[:60] + "..." if len(m.content) > 60 else m.content
                click.echo(f"  - [{format_tier(m.tier)}] {content}")

            click.echo(f"\n{color(f'{len(matches)} memories would be deleted', Colors.YELLOW)}")
            click.echo("Run without --dry-run to delete.")

        else:
            removed = mem.forget(query, threshold=threshold, tiers=tiers)

            if removed > 0:
                click.echo(color(f"Forgot {removed} memories matching '{query}'", Colors.GREEN))
            else:
                click.echo(color("No matching memories found.", Colors.YELLOW))

    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--memory", "-m",
    default=None,
    help="Memory database path"
)
def stats(memory: Optional[str]):
    """Show memory statistics."""
    memory_path = memory or get_default_memory_path()

    if not Path(memory_path).exists():
        click.echo(color(f"Error: Memory not found at {memory_path}", Colors.RED), err=True)
        sys.exit(1)

    try:
        mem = ProjectMemory(memory_path)
        s = mem.get_stats()

        click.echo(f"\n{color('Memory Statistics', Colors.BOLD)}")
        click.echo(f"{'=' * 30}")
        click.echo(f"Path: {color(s['path'], Colors.CYAN)}")
        click.echo(f"\nMemories by tier:")

        for tier_name, count in s["tiers"].items():
            tier = MemoryTier(tier_name)
            bar = "█" * min(count, 20)
            click.echo(f"  {format_tier(tier):12} {count:5} {Colors.DIM}{bar}{Colors.RESET}")

        click.echo(f"\nTotal: {color(str(s['total_memories']), Colors.CYAN)} memories")

    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--memory", "-m",
    default=None,
    help="Memory database path"
)
def cleanup(memory: Optional[str]):
    """Remove expired memories."""
    memory_path = memory or get_default_memory_path()

    if not Path(memory_path).exists():
        click.echo(color(f"Error: Memory not found at {memory_path}", Colors.RED), err=True)
        sys.exit(1)

    try:
        mem = ProjectMemory(memory_path)
        removed = mem.cleanup_expired()

        if removed > 0:
            click.echo(color(f"Cleaned up {removed} expired memories.", Colors.GREEN))
        else:
            click.echo("No expired memories found.")

    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--memory", "-m",
    default=None,
    help="Memory database path"
)
@click.option(
    "--tier",
    type=click.Choice(["working", "session", "project", "user"]),
    multiple=True,
    help="Tiers to clear (all if not specified)"
)
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation"
)
def clear(memory: Optional[str], tier: tuple, yes: bool):
    """Clear all memories."""
    memory_path = memory or get_default_memory_path()

    if not Path(memory_path).exists():
        click.echo(color(f"Error: Memory not found at {memory_path}", Colors.RED), err=True)
        sys.exit(1)

    tiers = [MemoryTier(t) for t in tier] if tier else None
    tier_desc = ", ".join(t.value for t in tiers) if tiers else "all tiers"

    if not yes:
        click.confirm(
            f"This will delete all memories from {tier_desc}. Continue?",
            abort=True
        )

    try:
        mem = ProjectMemory(memory_path)
        mem.clear(tiers=tiers)
        click.echo(color(f"Cleared memories from {tier_desc}.", Colors.GREEN))

    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--memory", "-m",
    default=None,
    help="Memory database path"
)
def interactive(memory: Optional[str]):
    """
    Interactive memory mode - REPL for managing memories.

    \b
    Commands in interactive mode:
        <query>          Search memories
        :r <content>     Remember (add memory)
        :f <query>       Forget (delete memories)
        :tier <name>     Set default tier
        :stats           Show statistics
        :help            Show help
        :quit            Exit
    """
    memory_path = memory or get_default_memory_path()

    try:
        click.echo(f"\n{color('Project Memory - Interactive Mode', Colors.BOLD)}")
        click.echo(f"{'=' * 40}")
        click.echo(f"Memory: {color(memory_path, Colors.CYAN)}")
        click.echo(f"\nType a query to search, or :help for commands.")
        click.echo(f"Press Ctrl+C or type :quit to exit.\n")

        mem = ProjectMemory(memory_path)
        current_tier = MemoryTier.PROJECT

        while True:
            try:
                prompt = f"{color('memory>', Colors.GREEN)} "
                query = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                click.echo("\nGoodbye!")
                break

            if not query:
                continue

            # Handle commands
            if query.startswith(':'):
                parts = query[1:].split(maxsplit=1)
                cmd = parts[0].lower() if parts else ""
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ('quit', 'exit', 'q'):
                    click.echo("Goodbye!")
                    break

                elif cmd == 'help':
                    click.echo(f"""
{color('Commands:', Colors.BOLD)}
  :r <content>     Remember a fact (current tier: {format_tier(current_tier)})
  :f <query>       Forget memories matching query
  :tier <name>     Set tier (working/session/project/user)
  :stats           Show memory statistics
  :cleanup         Remove expired memories
  :quit            Exit

{color('Search:', Colors.BOLD)}
  Just type your query to search memories.

{color('Examples:', Colors.BOLD)}
  :r User prefers dark mode
  :tier user
  :f API key
  user preferences
""")

                elif cmd == 'r' and arg:
                    memory_id = mem.remember(arg, tier=current_tier)
                    click.echo(f"{color('Remembered:', Colors.GREEN)} {arg[:50]}...")
                    click.echo(f"  Tier: {format_tier(current_tier)}")

                elif cmd == 'f' and arg:
                    removed = mem.forget(arg, threshold=0.8)
                    if removed:
                        click.echo(color(f"Forgot {removed} memories", Colors.GREEN))
                    else:
                        click.echo(color("No matching memories", Colors.YELLOW))

                elif cmd == 'tier' and arg:
                    try:
                        current_tier = MemoryTier(arg.lower())
                        click.echo(f"Tier set to {format_tier(current_tier)}")
                    except ValueError:
                        click.echo(color(f"Invalid tier: {arg}", Colors.RED))
                        click.echo("Valid tiers: working, session, project, user")

                elif cmd == 'stats':
                    s = mem.get_stats()
                    click.echo(f"Total: {s['total_memories']} memories")
                    for tier_name, count in s["tiers"].items():
                        click.echo(f"  {tier_name}: {count}")

                elif cmd == 'cleanup':
                    removed = mem.cleanup_expired()
                    click.echo(f"Cleaned up {removed} expired memories")

                else:
                    click.echo(color(f"Unknown command: {cmd}", Colors.RED))
                    click.echo("Type :help for available commands")

                continue

            # Perform search
            memories = mem.recall(query, k=5, threshold=0.3)

            if not memories:
                click.echo(color("No memories found.", Colors.YELLOW))
                continue

            click.echo()
            for i, m in enumerate(memories, 1):
                score_str = format_score(m.score)
                tier_str = format_tier(m.tier)
                content = m.content[:80] + "..." if len(m.content) > 80 else m.content
                click.echo(f"{i}. [{tier_str}] (score: {score_str})")
                click.echo(f"   {content}")
            click.echo()

    except ImportError as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
