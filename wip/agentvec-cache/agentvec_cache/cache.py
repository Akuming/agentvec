"""
Semantic Cache - LLM response caching with semantic similarity matching.
Built on AgentVec.
"""

import time
import hashlib
from functools import wraps
from typing import Callable, Optional, Any
from dataclasses import dataclass, field

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import agentvec
except ImportError:
    agentvec = None


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    total_latency_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_queries(self) -> int:
        return self.hits + self.misses

    def __repr__(self) -> str:
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate:.1%}, latency_saved={self.total_latency_saved_ms:.0f}ms)")


class SemanticCache:
    """
    Semantic cache for LLM responses.

    Caches responses based on semantic similarity of queries,
    not exact string matching. Uses AgentVec for vector storage
    and sentence-transformers for embeddings.

    Example:
        cache = SemanticCache("./cache.db")

        @cache.cached()
        def ask_llm(question: str) -> str:
            return openai.chat.completions.create(...)

        ask_llm("What is Python?")      # LLM call (~800ms)
        ask_llm("Tell me about Python") # Cache hit (~5ms)
    """

    def __init__(
        self,
        path: str = "./semantic_cache.db",
        threshold: float = 0.92,
        ttl: int = 3600,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        """
        Initialize the semantic cache.

        Args:
            path: Path to the cache database directory.
            threshold: Similarity threshold for cache hits (0.0-1.0).
                      Higher = stricter matching, fewer false hits.
                      Recommended: 0.90-0.95
            ttl: Default time-to-live in seconds (default: 1 hour).
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

        self.threshold = threshold
        self.default_ttl = ttl
        self.dimension = dimension
        self.stats = CacheStats()
        self._path = path

        # Initialize embedding model
        self._embedder = SentenceTransformer(embedding_model)

        # Initialize AgentVec
        self._db = agentvec.AgentVec(path)
        self._collection = self._db.collection(
            "cache",
            dim=dimension,
            metric="cosine"
        )

    def _embed(self, text: str) -> list:
        """Generate embedding for text."""
        return self._embedder.encode(text).tolist()

    def _make_cache_id(self, query: str) -> str:
        """Generate a stable ID for a query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def get(self, query: str, threshold: Optional[float] = None) -> Optional[Any]:
        """
        Check cache for a semantically similar query.

        Args:
            query: The query string to look up.
            threshold: Override default similarity threshold.

        Returns:
            Cached response if found, None otherwise.
        """
        threshold = threshold or self.threshold
        query_vec = self._embed(query)

        results = self._collection.search(vector=query_vec, k=1)

        if results and results[0].score >= threshold:
            self.stats.hits += 1
            metadata = results[0].metadata
            # Track latency saved if we have the original duration
            if "duration_ms" in metadata:
                self.stats.total_latency_saved_ms += metadata["duration_ms"]
            return metadata.get("response")

        return None

    def set(
        self,
        query: str,
        response: Any,
        ttl: Optional[int] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a query-response pair in the cache.

        Args:
            query: The query string.
            response: The response to cache (must be JSON-serializable).
            ttl: Time-to-live in seconds (uses default if None).
            duration_ms: How long the original call took (for stats).
            metadata: Additional metadata to store.

        Returns:
            The cache entry ID.
        """
        query_vec = self._embed(query)
        ttl = ttl or self.default_ttl

        cache_metadata = {
            "query": query,
            "response": response,
            "cached_at": time.time(),
        }

        if duration_ms is not None:
            cache_metadata["duration_ms"] = duration_ms

        if metadata:
            cache_metadata.update(metadata)

        cache_id = self._make_cache_id(query)

        self._collection.upsert(
            id=cache_id,
            vector=query_vec,
            metadata=cache_metadata,
            ttl=ttl,
        )

        return cache_id

    def cached(
        self,
        threshold: Optional[float] = None,
        ttl: Optional[int] = None,
        key_fn: Optional[Callable[..., str]] = None,
    ) -> Callable:
        """
        Decorator for caching function responses.

        Args:
            threshold: Similarity threshold override.
            ttl: TTL override.
            key_fn: Custom function to generate cache key from args.
                   Default uses first positional arg or str(args).

        Example:
            @cache.cached(threshold=0.9, ttl=7200)
            def expensive_llm_call(prompt: str) -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Generate cache key
                if key_fn:
                    query = key_fn(*args, **kwargs)
                elif args and isinstance(args[0], str):
                    # Common case: first arg is the prompt/question
                    query = args[0]
                else:
                    # Fallback: serialize all arguments
                    query = f"{func.__name__}:{args}:{kwargs}"

                # Check cache
                cached = self.get(query, threshold=threshold)
                if cached is not None:
                    return cached

                # Cache miss - call function
                self.stats.misses += 1
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000

                # Store in cache
                self.set(
                    query=query,
                    response=result,
                    ttl=ttl,
                    duration_ms=duration_ms,
                )

                return result
            return wrapper
        return decorator

    def invalidate(self, query: str) -> bool:
        """
        Remove a specific query from cache.

        Args:
            query: The exact query string to remove.

        Returns:
            True if the entry existed and was removed.
        """
        cache_id = self._make_cache_id(query)
        return self._collection.delete(cache_id)

    def invalidate_similar(self, query: str, threshold: float = 0.85) -> int:
        """
        Remove all cache entries similar to query.

        Useful when underlying data changes and related cached
        responses should be cleared.

        Args:
            query: The query to match against.
            threshold: Similarity threshold for matching.

        Returns:
            Number of entries removed.
        """
        query_vec = self._embed(query)
        results = self._collection.search(vector=query_vec, k=100)

        removed = 0
        for r in results:
            if r.score >= threshold:
                if self._collection.delete(r.id):
                    removed += 1

        return removed

    def clear(self) -> None:
        """Clear all cache entries."""
        self._db.drop_collection("cache")
        self._collection = self._db.collection(
            "cache",
            dim=self.dimension,
            metric="cosine"
        )
        self.stats = CacheStats()

    def compact(self) -> dict:
        """
        Remove expired entries and defragment storage.

        Returns:
            Dict with cleanup statistics.
        """
        stats = self._collection.compact()
        return {
            "expired_removed": stats.expired_removed,
            "tombstones_removed": stats.tombstones_removed,
            "bytes_freed": stats.bytes_freed,
        }

    def sync(self) -> None:
        """Flush pending writes to disk."""
        self._db.sync()

    def __len__(self) -> int:
        """Number of entries in cache."""
        return len(self._collection)

    def __repr__(self) -> str:
        return (f"SemanticCache(path='{self._path}', entries={len(self)}, "
                f"threshold={self.threshold}, stats={self.stats})")
