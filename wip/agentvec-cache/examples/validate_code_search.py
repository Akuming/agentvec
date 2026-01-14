#!/usr/bin/env python3
"""
Code Embedding Validation Test

This script tests whether semantic code search actually works.
It indexes the AgentVec codebase and runs natural language queries
to see if relevant code is found.

SUCCESS CRITERIA:
- Top 3 results contain relevant code >70% of the time
- If <50%, code search is not viable with current embeddings

Usage:
    python examples/validate_code_search.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(__file__).rsplit("examples", 1)[0])

try:
    from sentence_transformers import SentenceTransformer
    import agentvec
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install sentence-transformers agentvec")
    sys.exit(1)


def chunk_code(content: str, chunk_size: int = 40) -> list[tuple[int, str]]:
    """
    Split code into chunks by lines.
    Returns list of (start_line, chunk_content) tuples.
    """
    lines = content.split('\n')
    chunks = []
    for i in range(0, len(lines), chunk_size // 2):  # 50% overlap
        chunk = '\n'.join(lines[i:i + chunk_size])
        if chunk.strip() and len(chunk) > 50:  # Skip tiny chunks
            chunks.append((i + 1, chunk))
    return chunks


def index_codebase(
    codebase_path: str,
    db_path: str,
    extensions: Optional[list[str]] = None,
    model: Optional[SentenceTransformer] = None,
) -> tuple:
    """
    Index a codebase for semantic search.

    Returns:
        (db, collection, model, indexed_count)
    """
    extensions = extensions or ['.rs', '.py']
    skip_dirs = {'.git', 'node_modules', 'target', '__pycache__', 'venv', '.venv', 'dist', 'build'}

    if model is None:
        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    db = agentvec.AgentVec(db_path)
    collection = db.collection("code", dim=384, metric="cosine")

    indexed = 0
    files_indexed = 0

    print(f"Indexing {codebase_path}...")

    for root, dirs, files in os.walk(codebase_path):
        # Skip non-code directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if not any(file.endswith(ext) for ext in extensions):
                continue

            filepath = Path(root) / file
            rel_path = filepath.relative_to(codebase_path)

            try:
                content = filepath.read_text(encoding='utf-8')
            except Exception:
                continue

            chunks = chunk_code(content)
            for start_line, chunk in chunks:
                embedding = model.encode(chunk).tolist()
                collection.add(
                    vector=embedding,
                    metadata={
                        "file": str(rel_path),
                        "start_line": start_line,
                        "content": chunk,
                    }
                )
                indexed += 1

            files_indexed += 1

    db.sync()
    print(f"Indexed {indexed} chunks from {files_indexed} files")

    return db, collection, model, indexed


def search_code(collection, model, query: str, k: int = 5) -> list[dict]:
    """
    Search for code semantically.

    Returns list of results with file, score, and content preview.
    """
    query_vec = model.encode(query).tolist()
    results = collection.search(vector=query_vec, k=k)

    output = []
    for r in results:
        output.append({
            "file": r.metadata["file"],
            "line": r.metadata["start_line"],
            "score": r.score,
            "preview": r.metadata["content"][:300].replace('\n', ' ')[:150],
        })

    return output


def run_validation_tests(collection, model) -> dict:
    """
    Run validation queries and assess results.

    Returns dict with pass/fail status and details.
    """
    # Define test queries and what we expect to find
    # Format: (query, expected_keywords_in_results)
    test_cases = [
        (
            "where is vector similarity search implemented",
            ["search", "distance", "cosine", "similarity", "hnsw"]
        ),
        (
            "how are vectors stored on disk",
            ["storage", "mmap", "vectors", "file", "write"]
        ),
        (
            "where is the HNSW graph built",
            ["hnsw", "graph", "insert", "build", "layer"]
        ),
        (
            "how does TTL expiration work",
            ["ttl", "expire", "time", "delete", "compact"]
        ),
        (
            "where is metadata filtered",
            ["filter", "metadata", "where", "query", "match"]
        ),
        (
            "how are crash recovery handled",
            ["recovery", "crash", "transaction", "rollback", "pending"]
        ),
        (
            "where is the Python binding defined",
            ["py", "python", "pyo3", "binding", "pyclass"]
        ),
        (
            "how is batch insertion optimized",
            ["batch", "add", "insert", "coalesce", "write"]
        ),
        (
            "where are distance metrics calculated",
            ["distance", "cosine", "dot", "l2", "metric", "simd"]
        ),
        (
            "how does the collection API work",
            ["collection", "add", "search", "get", "delete"]
        ),
    ]

    print("\n" + "=" * 70)
    print("CODE EMBEDDING VALIDATION TEST")
    print("=" * 70)
    print(f"Running {len(test_cases)} test queries...\n")

    results = []

    for query, expected_keywords in test_cases:
        search_results = search_code(collection, model, query, k=5)

        # Check if any of the expected keywords appear in top 3 results
        top_3_content = " ".join([
            r["preview"].lower() + " " + r["file"].lower()
            for r in search_results[:3]
        ])

        found_keywords = [kw for kw in expected_keywords if kw.lower() in top_3_content]
        relevant = len(found_keywords) >= 2  # At least 2 keywords found

        results.append({
            "query": query,
            "relevant": relevant,
            "found_keywords": found_keywords,
            "top_file": search_results[0]["file"] if search_results else "N/A",
            "top_score": search_results[0]["score"] if search_results else 0,
        })

        status = "PASS" if relevant else "FAIL"
        print(f"[{status}] \"{query[:50]}...\"")
        print(f"       Top result: {results[-1]['top_file']} (score: {results[-1]['top_score']:.3f})")
        print(f"       Keywords found: {found_keywords}")
        print()

    # Calculate overall results
    passed = sum(1 for r in results if r["relevant"])
    total = len(results)
    pass_rate = passed / total

    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Passed: {passed}/{total} ({pass_rate:.0%})")
    print()

    if pass_rate >= 0.7:
        print("VERDICT: PASS - Code semantic search is viable!")
        print("         Proceed with the strategy as planned.")
        verdict = "PASS"
    elif pass_rate >= 0.5:
        print("VERDICT: MARGINAL - Results are mixed.")
        print("         Consider testing different embedding models.")
        print("         May need code-specific embeddings (CodeBERT, etc.)")
        verdict = "MARGINAL"
    else:
        print("VERDICT: FAIL - Code semantic search is not working well.")
        print("         Recommend Fallback A: Focus on text/document memory instead.")
        print("         Code search needs specialized embeddings.")
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "details": results,
    }


def main():
    # Find the agentvec codebase (parent of agentvec-cache)
    script_dir = Path(__file__).resolve().parent
    cache_dir = script_dir.parent
    codebase_dir = cache_dir.parent

    # Check if we're in the right place
    agentvec_src = codebase_dir / "agentvec" / "src"
    if not agentvec_src.exists():
        print(f"Error: Cannot find AgentVec source at {agentvec_src}")
        print("Make sure you're running from the agentvec-cache directory")
        sys.exit(1)

    print(f"Codebase: {codebase_dir}")
    print(f"Source:   {agentvec_src}")

    temp_dir = tempfile.mkdtemp(prefix="code_search_test_")
    print(f"Index:    {temp_dir}\n")

    try:
        # Index the Rust source code
        db, collection, model, indexed = index_codebase(
            str(agentvec_src),
            temp_dir,
            extensions=['.rs'],
        )

        if indexed == 0:
            print("Error: No code indexed!")
            sys.exit(1)

        # Run validation tests
        results = run_validation_tests(collection, model)

        # Interactive mode: let user try their own queries
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("Try your own queries (type 'quit' to exit):\n")

        while True:
            try:
                query = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query or query.lower() in ('quit', 'exit', 'q'):
                break

            search_results = search_code(collection, model, query, k=5)
            print()
            for i, r in enumerate(search_results, 1):
                print(f"{i}. {r['file']}:{r['line']} (score: {r['score']:.3f})")
                print(f"   {r['preview'][:100]}...")
                print()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\n[Cleaned up temporary index]")


if __name__ == "__main__":
    main()
