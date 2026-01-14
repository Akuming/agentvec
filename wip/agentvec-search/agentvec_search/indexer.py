"""
Code indexer for semantic search.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import agentvec
except ImportError:
    agentvec = None


@dataclass
class IndexStats:
    """Statistics from indexing operation."""
    files_indexed: int = 0
    chunks_indexed: int = 0
    files_skipped: int = 0
    errors: int = 0


# Default file extensions to index by language
LANGUAGE_EXTENSIONS = {
    "rust": [".rs"],
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx"],
    "csharp": [".cs"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "scala": [".scala"],
    "shell": [".sh", ".bash", ".zsh"],
    "sql": [".sql"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "yaml": [".yaml", ".yml"],
    "json": [".json"],
    "markdown": [".md", ".markdown"],
    "toml": [".toml"],
}

# Directories to always skip
SKIP_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    ".cache",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    "coverage",
    ".coverage",
    ".idea",
    ".vscode",
    "*.egg-info",
}


def get_extensions_for_languages(languages: Optional[list[str]] = None) -> list[str]:
    """Get file extensions for specified languages."""
    if not languages:
        # Default: common programming languages
        languages = ["rust", "python", "javascript", "typescript", "go", "java", "c", "cpp"]

    extensions = []
    for lang in languages:
        lang_lower = lang.lower()
        if lang_lower in LANGUAGE_EXTENSIONS:
            extensions.extend(LANGUAGE_EXTENSIONS[lang_lower])
        elif lang_lower.startswith("."):
            # Direct extension provided
            extensions.append(lang_lower)

    return list(set(extensions))


def chunk_code(content: str, chunk_size: int = 50, overlap: int = 10) -> list[tuple[int, str]]:
    """
    Split code into overlapping chunks.

    Args:
        content: File content to chunk.
        chunk_size: Number of lines per chunk.
        overlap: Number of overlapping lines between chunks.

    Returns:
        List of (start_line, chunk_content) tuples.
    """
    lines = content.split('\n')
    chunks = []

    step = max(1, chunk_size - overlap)

    for i in range(0, len(lines), step):
        chunk_lines = lines[i:i + chunk_size]
        chunk = '\n'.join(chunk_lines)

        # Skip empty or very small chunks
        if chunk.strip() and len(chunk.strip()) > 20:
            chunks.append((i + 1, chunk))  # 1-indexed line numbers

    return chunks


def compute_file_hash(content: str) -> str:
    """Compute hash of file content for change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class CodeIndexer:
    """
    Indexes codebases for semantic search.

    Example:
        indexer = CodeIndexer("./index.db")
        indexer.index_directory("./src")
        results = indexer.search("authentication logic")
    """

    def __init__(
        self,
        index_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        """
        Initialize the code indexer.

        Args:
            index_path: Path to store the index database.
            embedding_model: SentenceTransformer model name.
            dimension: Embedding dimension (must match model).
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        if agentvec is None:
            raise ImportError(
                "agentvec is required. "
                "Install with: pip install agentvec"
            )

        self.index_path = index_path
        self.dimension = dimension
        self._model = SentenceTransformer(embedding_model)
        self._db = agentvec.AgentVec(index_path)
        self._collection = self._db.collection("code", dim=dimension, metric="cosine")
        self._meta = self._db.collection("meta", dim=dimension, metric="cosine")

    def _embed(self, text: str) -> list:
        """Generate embedding for text."""
        return self._model.encode(text).tolist()

    def index_directory(
        self,
        directory: str,
        extensions: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
        chunk_size: int = 50,
        overlap: int = 10,
        progress_callback: Optional[callable] = None,
    ) -> IndexStats:
        """
        Index all code files in a directory.

        Args:
            directory: Root directory to index.
            extensions: File extensions to include (e.g., [".py", ".rs"]).
            languages: Languages to include (e.g., ["python", "rust"]).
            chunk_size: Lines per chunk.
            overlap: Overlapping lines between chunks.
            progress_callback: Called with (current_file, files_done, total_files).

        Returns:
            IndexStats with indexing statistics.
        """
        if extensions is None:
            extensions = get_extensions_for_languages(languages)

        stats = IndexStats()
        directory = Path(directory).resolve()

        # Collect all files first
        files_to_index = []
        for root, dirs, files in os.walk(directory):
            # Filter out skip directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    files_to_index.append(Path(root) / file)

        total_files = len(files_to_index)

        for idx, filepath in enumerate(files_to_index):
            if progress_callback:
                progress_callback(str(filepath), idx, total_files)

            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                stats.files_skipped += 1
                continue

            # Get relative path for storage
            try:
                rel_path = filepath.relative_to(directory)
            except ValueError:
                rel_path = filepath

            # Chunk and index the file
            chunks = chunk_code(content, chunk_size, overlap)

            if not chunks:
                stats.files_skipped += 1
                continue

            file_hash = compute_file_hash(content)

            for start_line, chunk in chunks:
                # Create unique ID for this chunk
                chunk_id = f"{rel_path}:{start_line}:{file_hash[:8]}"

                try:
                    embedding = self._embed(chunk)
                    self._collection.upsert(
                        id=chunk_id,
                        vector=embedding,
                        metadata={
                            "file": str(rel_path),
                            "start_line": start_line,
                            "content": chunk,
                            "file_hash": file_hash,
                        }
                    )
                    stats.chunks_indexed += 1
                except Exception as e:
                    stats.errors += 1

            stats.files_indexed += 1

        # Save index metadata
        self._db.sync()

        return stats

    def search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Search for code semantically.

        Args:
            query: Natural language query.
            k: Maximum number of results.
            threshold: Minimum similarity score (0.0-1.0).

        Returns:
            List of results with file, line, score, and content.
        """
        query_vec = self._embed(query)
        results = self._collection.search(vector=query_vec, k=k)

        output = []
        seen_files = set()  # Deduplicate by file

        for r in results:
            if r.score < threshold:
                continue

            file_key = f"{r.metadata['file']}:{r.metadata['start_line']}"
            if file_key in seen_files:
                continue
            seen_files.add(file_key)

            output.append({
                "file": r.metadata["file"],
                "line": r.metadata["start_line"],
                "score": r.score,
                "content": r.metadata["content"],
            })

        return output

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "chunks": len(self._collection),
            "index_path": self.index_path,
        }

    def clear(self) -> None:
        """Clear the entire index."""
        self._db.drop_collection("code")
        self._collection = self._db.collection("code", dim=self.dimension, metric="cosine")
