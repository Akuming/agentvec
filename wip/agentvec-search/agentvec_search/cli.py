"""
Semantic Code Search CLI - Search your codebase with natural language.

Usage:
    agentvec-search index ./src
    agentvec-search search "where is authentication handled"
    agentvec-search interactive
"""

import sys
import click
from pathlib import Path
from typing import Optional

from .indexer import CodeIndexer, IndexStats, LANGUAGE_EXTENSIONS


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


def get_default_index_path() -> str:
    """Get default index path in current directory."""
    return str(Path.cwd() / ".agentvec-index")


def format_score(score: float) -> str:
    """Format similarity score with color."""
    if score >= 0.6:
        return color(f"{score:.3f}", Colors.GREEN)
    elif score >= 0.4:
        return color(f"{score:.3f}", Colors.YELLOW)
    else:
        return color(f"{score:.3f}", Colors.DIM)


def truncate_content(content: str, max_lines: int = 5, max_width: int = 100) -> str:
    """Truncate content for display."""
    lines = content.split('\n')[:max_lines]
    truncated = []
    for line in lines:
        if len(line) > max_width:
            truncated.append(line[:max_width - 3] + "...")
        else:
            truncated.append(line)

    result = '\n'.join(truncated)
    if len(content.split('\n')) > max_lines:
        result += f"\n{color('... (truncated)', Colors.DIM)}"

    return result


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Semantic Code Search - Search your codebase with natural language.

    \b
    Examples:
        agentvec-search index ./src
        agentvec-search search "where is authentication handled"
        agentvec-search search "how does caching work" -k 5
        agentvec-search interactive
    """
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "--index", "-i",
    default=None,
    help="Index path (default: .agentvec-index in current dir)"
)
@click.option(
    "--languages", "-l",
    multiple=True,
    help="Languages to index (e.g., -l python -l rust)"
)
@click.option(
    "--extensions", "-e",
    multiple=True,
    help="File extensions to index (e.g., -e .py -e .rs)"
)
@click.option(
    "--chunk-size", "-c",
    default=50,
    help="Lines per chunk (default: 50)"
)
@click.option(
    "--overlap", "-o",
    default=10,
    help="Overlapping lines between chunks (default: 10)"
)
@click.option(
    "--clear/--no-clear",
    default=False,
    help="Clear existing index before indexing"
)
def index(
    directory: str,
    index: Optional[str],
    languages: tuple,
    extensions: tuple,
    chunk_size: int,
    overlap: int,
    clear: bool,
):
    """
    Index a codebase for semantic search.

    \b
    Examples:
        agentvec-search index ./src
        agentvec-search index ./src -l python -l rust
        agentvec-search index ./src -e .py -e .rs
        agentvec-search index ./src --clear
    """
    index_path = index or get_default_index_path()
    ext_list = list(extensions) if extensions else None
    lang_list = list(languages) if languages else None

    click.echo(f"\n{color('Semantic Code Search - Indexer', Colors.BOLD)}")
    click.echo(f"{'=' * 40}")
    click.echo(f"Directory: {color(directory, Colors.CYAN)}")
    click.echo(f"Index:     {color(index_path, Colors.CYAN)}")

    if lang_list:
        click.echo(f"Languages: {', '.join(lang_list)}")
    if ext_list:
        click.echo(f"Extensions: {', '.join(ext_list)}")

    click.echo()

    try:
        click.echo("Loading embedding model...")
        indexer = CodeIndexer(index_path)

        if clear:
            click.echo("Clearing existing index...")
            indexer.clear()

        def progress(filepath: str, done: int, total: int):
            percent = (done / total * 100) if total > 0 else 0
            # Truncate filepath for display
            display_path = filepath
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            click.echo(f"\r[{percent:5.1f}%] Indexing: {display_path:<50}", nl=False)

        click.echo(f"Indexing {color(directory, Colors.CYAN)}...\n")

        stats = indexer.index_directory(
            directory,
            extensions=ext_list,
            languages=lang_list,
            chunk_size=chunk_size,
            overlap=overlap,
            progress_callback=progress,
        )

        click.echo("\r" + " " * 80 + "\r", nl=False)  # Clear progress line
        click.echo(f"\n{color('Indexing complete!', Colors.GREEN)}")
        click.echo(f"  Files indexed:  {color(str(stats.files_indexed), Colors.CYAN)}")
        click.echo(f"  Chunks indexed: {color(str(stats.chunks_indexed), Colors.CYAN)}")
        click.echo(f"  Files skipped:  {stats.files_skipped}")
        click.echo(f"  Errors:         {stats.errors}")
        click.echo(f"\nIndex saved to: {color(index_path, Colors.CYAN)}")

    except ImportError as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--index", "-i",
    default=None,
    help="Index path (default: .agentvec-index in current dir)"
)
@click.option(
    "--results", "-k",
    default=10,
    help="Number of results (default: 10)"
)
@click.option(
    "--threshold", "-t",
    default=0.0,
    help="Minimum similarity score (default: 0.0)"
)
@click.option(
    "--context", "-c",
    default=5,
    help="Lines of context to show (default: 5)"
)
@click.option(
    "--no-content",
    is_flag=True,
    help="Only show file paths, not content"
)
def search(
    query: str,
    index: Optional[str],
    results: int,
    threshold: float,
    context: int,
    no_content: bool,
):
    """
    Search for code using natural language.

    \b
    Examples:
        agentvec-search search "where is authentication handled"
        agentvec-search search "database connection" -k 5
        agentvec-search search "error handling" -t 0.4
    """
    index_path = index or get_default_index_path()

    if not Path(index_path).exists():
        click.echo(color(f"Error: Index not found at {index_path}", Colors.RED), err=True)
        click.echo("Run 'agentvec-search index <directory>' first.", err=True)
        sys.exit(1)

    try:
        indexer = CodeIndexer(index_path)

        click.echo(f"\n{color('Query:', Colors.BOLD)} {color(query, Colors.CYAN)}")
        click.echo(f"{'=' * 60}\n")

        search_results = indexer.search(query, k=results, threshold=threshold)

        if not search_results:
            click.echo(color("No results found.", Colors.YELLOW))
            click.echo("Try a different query or lower the threshold with -t 0.3")
            return

        for i, r in enumerate(search_results, 1):
            file_loc = f"{r['file']}:{r['line']}"
            score_str = format_score(r['score'])

            click.echo(f"{color(str(i) + '.', Colors.BOLD)} {color(file_loc, Colors.BLUE)} {Colors.DIM}(score: {score_str}{Colors.DIM}){Colors.RESET}")

            if not no_content:
                content_preview = truncate_content(r['content'], max_lines=context)
                # Indent content
                indented = '\n'.join('   ' + line for line in content_preview.split('\n'))
                click.echo(f"{Colors.DIM}{indented}{Colors.RESET}")
                click.echo()

        click.echo(f"{color(f'Found {len(search_results)} results', Colors.DIM)}")

    except ImportError as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--index", "-i",
    default=None,
    help="Index path (default: .agentvec-index in current dir)"
)
@click.option(
    "--results", "-k",
    default=5,
    help="Number of results per query (default: 5)"
)
def interactive(index: Optional[str], results: int):
    """
    Interactive search mode - enter queries in a REPL.

    \b
    Commands in interactive mode:
        <query>     Search for code
        :k <num>    Change number of results
        :clear      Clear screen
        :stats      Show index statistics
        :help       Show help
        :quit       Exit
    """
    index_path = index or get_default_index_path()

    if not Path(index_path).exists():
        click.echo(color(f"Error: Index not found at {index_path}", Colors.RED), err=True)
        click.echo("Run 'agentvec-search index <directory>' first.", err=True)
        sys.exit(1)

    try:
        click.echo(f"\n{color('Semantic Code Search - Interactive Mode', Colors.BOLD)}")
        click.echo(f"{'=' * 45}")
        click.echo(f"Index: {color(index_path, Colors.CYAN)}")
        click.echo(f"\nType a query to search, or :help for commands.")
        click.echo(f"Press Ctrl+C or type :quit to exit.\n")

        indexer = CodeIndexer(index_path)
        k = results

        while True:
            try:
                query = input(f"{color('search>', Colors.GREEN)} ").strip()
            except (EOFError, KeyboardInterrupt):
                click.echo("\nGoodbye!")
                break

            if not query:
                continue

            # Handle commands
            if query.startswith(':'):
                cmd = query[1:].lower().split()
                if not cmd:
                    continue

                if cmd[0] in ('quit', 'exit', 'q'):
                    click.echo("Goodbye!")
                    break
                elif cmd[0] == 'help':
                    click.echo(f"""
{color('Commands:', Colors.BOLD)}
  :k <num>     Set number of results (current: {k})
  :stats       Show index statistics
  :clear       Clear screen
  :quit        Exit interactive mode

{color('Search:', Colors.BOLD)}
  Just type your query in natural language.

{color('Examples:', Colors.BOLD)}
  where is authentication handled
  how does caching work
  database connection logic
""")
                elif cmd[0] == 'k' and len(cmd) > 1:
                    try:
                        k = int(cmd[1])
                        click.echo(f"Results set to {k}")
                    except ValueError:
                        click.echo(color("Invalid number", Colors.RED))
                elif cmd[0] == 'stats':
                    stats = indexer.get_stats()
                    click.echo(f"Chunks indexed: {stats['chunks']}")
                    click.echo(f"Index path: {stats['index_path']}")
                elif cmd[0] == 'clear':
                    click.clear()
                else:
                    click.echo(color(f"Unknown command: {cmd[0]}", Colors.RED))
                continue

            # Perform search
            search_results = indexer.search(query, k=k)

            if not search_results:
                click.echo(color("No results found.", Colors.YELLOW))
                continue

            click.echo()
            for i, r in enumerate(search_results, 1):
                file_loc = f"{r['file']}:{r['line']}"
                score_str = format_score(r['score'])
                click.echo(f"{i}. {color(file_loc, Colors.BLUE)} (score: {score_str})")

                # Show brief preview
                preview = r['content'].split('\n')[0][:80]
                click.echo(f"   {Colors.DIM}{preview}{Colors.RESET}")

            click.echo()

    except ImportError as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--index", "-i",
    default=None,
    help="Index path (default: .agentvec-index in current dir)"
)
def stats(index: Optional[str]):
    """Show index statistics."""
    index_path = index or get_default_index_path()

    if not Path(index_path).exists():
        click.echo(color(f"Error: Index not found at {index_path}", Colors.RED), err=True)
        sys.exit(1)

    try:
        indexer = CodeIndexer(index_path)
        index_stats = indexer.get_stats()

        click.echo(f"\n{color('Index Statistics', Colors.BOLD)}")
        click.echo(f"{'=' * 30}")
        click.echo(f"Path:   {color(index_stats['index_path'], Colors.CYAN)}")
        click.echo(f"Chunks: {color(str(index_stats['chunks']), Colors.CYAN)}")

    except Exception as e:
        click.echo(color(f"Error: {e}", Colors.RED), err=True)
        sys.exit(1)


@cli.command()
def languages():
    """List supported languages and their file extensions."""
    click.echo(f"\n{color('Supported Languages', Colors.BOLD)}")
    click.echo(f"{'=' * 40}")

    for lang, exts in sorted(LANGUAGE_EXTENSIONS.items()):
        ext_str = ', '.join(exts)
        click.echo(f"  {color(lang, Colors.CYAN):12} {ext_str}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
