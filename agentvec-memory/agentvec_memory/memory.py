"""
Project Memory - Tiered persistent memory for AI agents.

Provides semantic memory storage with automatic TTL expiration
and tiered organization (working/session/project/user).

Supports multiple embedding backends:
- fastembed (default): Lightweight ONNX-based, ~100MB
- sentence-transformers: Full PyTorch, GPU support, ~3GB
- Custom: Bring your own embedding function
"""

import time
import uuid
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Callable

from .embedders import Embedder, create_embedder

logger = logging.getLogger(__name__)

try:
    import agentvec
except ImportError:
    agentvec = None


class MemoryTier(Enum):
    """Memory tiers with different retention characteristics."""

    WORKING = "working"      # Very short-term, current task context
    SESSION = "session"      # Single work session
    PROJECT = "project"      # Persistent project knowledge
    USER = "user"            # Long-term user preferences

    @property
    def default_ttl(self) -> int:
        """Default TTL in seconds for each tier."""
        ttls = {
            MemoryTier.WORKING: 300,        # 5 minutes
            MemoryTier.SESSION: 3600,       # 1 hour
            MemoryTier.PROJECT: 86400 * 30, # 30 days
            MemoryTier.USER: 86400 * 365,   # 1 year
        }
        return ttls[self]


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    tier: MemoryTier
    score: float
    created_at: float
    expires_at: float
    metadata: Dict[str, Any]

    @property
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Seconds until expiration."""
        return max(0, self.expires_at - time.time())


class ProjectMemory:
    """
    Tiered persistent memory for AI agents.

    Stores facts, context, and knowledge with semantic retrieval
    and automatic TTL-based expiration.

    Supports multiple embedding backends:
    - fastembed (default): Lightweight, ~100MB install
    - sentence-transformers: Full-featured, ~3GB install, GPU support
    - Custom: Bring your own embedding function

    Example:
        # Use default embedder (fastembed)
        memory = ProjectMemory("./memory.db")

        # Explicitly choose embedder
        memory = ProjectMemory("./memory.db", embedder="fastembed")
        memory = ProjectMemory("./memory.db", embedder="sentence-transformers")
        memory = ProjectMemory("./memory.db", embedder="openai")

        # Custom embedding function
        memory = ProjectMemory("./memory.db", embedder=my_embed_func, dimension=384)

        # Remember facts
        memory.remember("User prefers dark mode", tier=MemoryTier.USER)
        memory.remember("Working on auth feature", tier=MemoryTier.SESSION)

        # Recall relevant memories
        results = memory.recall("user interface preferences")
        for mem in results:
            print(f"{mem.content} (score: {mem.score:.3f})")

        # Forget specific memories
        memory.forget("dark mode")
    """

    def __init__(
        self,
        path: str,
        embedder: Union[Embedder, Callable[[List[str]], List[List[float]]], str, None] = None,
        dimension: Optional[int] = None,
        # Legacy parameters for backward compatibility
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize project memory.

        Args:
            path: Path to store the memory database.
            embedder: Embedding provider. Can be:
                - None: Auto-detect best available (fastembed > sentence-transformers)
                - str: Named embedder ('fastembed', 'sentence-transformers', 'openai')
                - Callable: Custom function (requires dimension parameter)
                - Embedder: Custom Embedder instance
            dimension: Embedding dimension. Required only for custom callables.
                Auto-detected for built-in embedders.
            embedding_model: [DEPRECATED] Use embedder parameter instead.
                Kept for backward compatibility.
        """
        if agentvec is None:
            raise ImportError(
                "agentvec is required. "
                "Install with: pip install agentvec"
            )

        self.path = path

        # Handle legacy embedding_model parameter
        if embedding_model is not None and embedder is None:
            logger.warning(
                "embedding_model parameter is deprecated. "
                "Use embedder='sentence-transformers' instead."
            )
            # Try to use sentence-transformers with the specified model
            try:
                from .embedders import SentenceTransformerEmbedder
                self._embedder = SentenceTransformerEmbedder(model_name=embedding_model)
            except ImportError:
                # Fall back to fastembed with equivalent model
                from .embedders import FastEmbedEmbedder
                fastembed_model = f"sentence-transformers/{embedding_model}"
                self._embedder = FastEmbedEmbedder(model_name=fastembed_model)
        else:
            # Create embedder using the factory
            self._embedder = create_embedder(embedder, dimension=dimension)

        self.dimension = self._embedder.dimension
        logger.info(f"ProjectMemory initialized with {type(self._embedder).__name__} "
                    f"({self.dimension} dimensions)")

        self._db = agentvec.AgentVec(path)

        # Create collection for each tier
        self._collections = {}
        for tier in MemoryTier:
            self._collections[tier] = self._db.collection(
                f"memory_{tier.value}",
                dim=self.dimension,
                metric="cosine"
            )

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self._embedder.embed_single(text)

    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        return str(uuid.uuid4())

    def remember(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.PROJECT,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a memory.

        Args:
            content: The fact or knowledge to remember.
            tier: Memory tier (affects default TTL).
            ttl: Time-to-live in seconds (uses tier default if None).
            metadata: Additional metadata to store.

        Returns:
            Memory ID for later reference.
        """
        memory_id = self._generate_id()
        now = time.time()
        actual_ttl = ttl if ttl is not None else tier.default_ttl
        expires_at = now + actual_ttl

        embedding = self._embed(content)

        mem_metadata = {
            "content": content,
            "tier": tier.value,
            "created_at": now,
            "expires_at": expires_at,
            **(metadata or {})
        }

        collection = self._collections[tier]
        collection.upsert(
            id=memory_id,
            vector=embedding,
            metadata=mem_metadata
        )

        self._db.sync()
        return memory_id

    def recall(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.3,
        tiers: Optional[List[MemoryTier]] = None,
        include_expired: bool = False,
    ) -> List[Memory]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query (natural language).
            k: Maximum number of results per tier.
            threshold: Minimum similarity score (0.0-1.0).
            tiers: Tiers to search (all if None).
            include_expired: Include expired memories.

        Returns:
            List of relevant memories, sorted by score.
        """
        query_vec = self._embed(query)
        tiers_to_search = tiers or list(MemoryTier)

        all_results = []
        now = time.time()

        for tier in tiers_to_search:
            collection = self._collections[tier]
            results = collection.search(vector=query_vec, k=k)

            for r in results:
                if r.score < threshold:
                    continue

                expires_at = r.metadata.get("expires_at", float("inf"))
                if not include_expired and now > expires_at:
                    continue

                memory = Memory(
                    id=r.id,
                    content=r.metadata.get("content", ""),
                    tier=MemoryTier(r.metadata.get("tier", tier.value)),
                    score=r.score,
                    created_at=r.metadata.get("created_at", 0),
                    expires_at=expires_at,
                    metadata={k: v for k, v in r.metadata.items()
                              if k not in ("content", "tier", "created_at", "expires_at")}
                )
                all_results.append(memory)

        # Sort by score descending
        all_results.sort(key=lambda m: m.score, reverse=True)
        return all_results[:k]

    def forget(
        self,
        query: str,
        threshold: float = 0.8,
        tiers: Optional[List[MemoryTier]] = None,
    ) -> int:
        """
        Remove memories similar to query.

        Args:
            query: What to forget (semantic match).
            threshold: Similarity threshold for deletion.
            tiers: Tiers to search (all if None).

        Returns:
            Number of memories removed.
        """
        # Find matching memories
        matches = self.recall(
            query,
            k=100,  # Get many matches
            threshold=threshold,
            tiers=tiers,
            include_expired=True,  # Also forget expired
        )

        removed = 0
        for memory in matches:
            tier = memory.tier
            collection = self._collections[tier]
            try:
                collection.delete(memory.id)
                removed += 1
            except Exception:
                pass

        if removed > 0:
            self._db.sync()

        return removed

    def forget_by_id(self, memory_id: str) -> bool:
        """
        Remove a specific memory by ID.

        Args:
            memory_id: The memory ID to remove.

        Returns:
            True if removed, False if not found.
        """
        for collection in self._collections.values():
            try:
                collection.delete(memory_id)
                self._db.sync()
                return True
            except Exception:
                continue
        return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired memories.

        Returns:
            Number of memories removed.
        """
        removed = 0
        now = time.time()

        for tier, collection in self._collections.items():
            # Get all memories (expensive but necessary for cleanup)
            # We use a very generic query to get most items
            try:
                dummy_vec = [0.0] * self.dimension
                results = collection.search(vector=dummy_vec, k=10000)

                for r in results:
                    expires_at = r.metadata.get("expires_at", float("inf"))
                    if now > expires_at:
                        try:
                            collection.delete(r.id)
                            removed += 1
                        except Exception:
                            pass
            except Exception:
                pass

        if removed > 0:
            self._db.sync()

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "path": self.path,
            "tiers": {},
            "total_memories": 0,
        }

        for tier in MemoryTier:
            collection = self._collections[tier]
            count = len(collection)
            stats["tiers"][tier.value] = count
            stats["total_memories"] += count

        return stats

    def clear(self, tiers: Optional[List[MemoryTier]] = None) -> None:
        """
        Clear all memories.

        Args:
            tiers: Tiers to clear (all if None).
        """
        tiers_to_clear = tiers or list(MemoryTier)

        for tier in tiers_to_clear:
            self._db.drop_collection(f"memory_{tier.value}")
            self._collections[tier] = self._db.collection(
                f"memory_{tier.value}",
                dim=self.dimension,
                metric="cosine"
            )
