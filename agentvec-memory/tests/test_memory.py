"""
Tests for agentvec-memory.

These tests verify the core functionality of ProjectMemory including
remember, recall, forget operations and tier management.
"""

import time
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from agentvec_memory import ProjectMemory, MemoryTier, Memory, Embedder


# --- Fixtures ---

class MockEmbedder(Embedder):
    """Mock embedder that returns deterministic embeddings for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, texts):
        """Generate deterministic embeddings based on text hash."""
        result = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self._dimension).astype(np.float32).tolist()
            result.append(embedding)
        return result

    def embed_single(self, text):
        """Embed a single text."""
        return self.embed([text])[0]

    @property
    def dimension(self):
        return self._dimension


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic embeddings."""
    return MockEmbedder(dimension=384)


@pytest.fixture
def mock_agentvec():
    """Mock agentvec module."""
    mock_db = Mock()
    mock_collection = Mock()
    mock_collection.__len__ = Mock(return_value=0)
    mock_collection.search.return_value = []
    mock_db.collection.return_value = mock_collection
    mock_db.sync = Mock()
    mock_db.drop_collection = Mock()

    mock_module = Mock()
    mock_module.AgentVec.return_value = mock_db
    return mock_module, mock_db, mock_collection


@pytest.fixture
def memory_with_mocks(mock_embedder, mock_agentvec):
    """Create ProjectMemory with mocked dependencies.

    Only the PROJECT tier collection is active (returns mock results).
    Other tiers return empty results to avoid 4x multiplication.
    """
    mock_module, mock_db, mock_collection = mock_agentvec

    with patch.dict('sys.modules', {'agentvec': mock_module}):
        with patch('agentvec_memory.memory.agentvec', mock_module):
            with patch('agentvec_memory.memory.create_embedder', return_value=mock_embedder):
                with tempfile.TemporaryDirectory() as tmpdir:
                    memory = ProjectMemory(tmpdir)
                    memory._embedder = mock_embedder
                    memory._db = mock_db

                    # Create separate mock collections for each tier
                    # Only PROJECT tier will return results (others empty)
                    for tier in MemoryTier:
                        tier_collection = Mock()
                        tier_collection.__len__ = Mock(return_value=0)
                        tier_collection.search.return_value = []
                        tier_collection.upsert = Mock()
                        tier_collection.delete = Mock()
                        memory._collections[tier] = tier_collection

                    # The main mock_collection is used for PROJECT tier
                    memory._collections[MemoryTier.PROJECT] = mock_collection

                    yield memory, mock_db, mock_collection


# --- MemoryTier Tests ---

class TestMemoryTier:
    """Tests for MemoryTier enum."""

    def test_tier_values(self):
        """Test that all tiers have expected values."""
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.SESSION.value == "session"
        assert MemoryTier.PROJECT.value == "project"
        assert MemoryTier.USER.value == "user"

    def test_tier_default_ttls(self):
        """Test that tiers have appropriate default TTLs."""
        assert MemoryTier.WORKING.default_ttl == 300        # 5 minutes
        assert MemoryTier.SESSION.default_ttl == 3600       # 1 hour
        assert MemoryTier.PROJECT.default_ttl == 86400 * 30 # 30 days
        assert MemoryTier.USER.default_ttl == 86400 * 365   # 1 year

    def test_tier_ttl_ordering(self):
        """Test that TTLs increase with tier importance."""
        assert MemoryTier.WORKING.default_ttl < MemoryTier.SESSION.default_ttl
        assert MemoryTier.SESSION.default_ttl < MemoryTier.PROJECT.default_ttl
        assert MemoryTier.PROJECT.default_ttl < MemoryTier.USER.default_ttl


# --- Memory Dataclass Tests ---

class TestMemoryDataclass:
    """Tests for Memory dataclass."""

    def test_memory_creation(self):
        """Test creating a Memory instance."""
        now = time.time()
        memory = Memory(
            id="test-id",
            content="Test content",
            tier=MemoryTier.PROJECT,
            score=0.85,
            created_at=now,
            expires_at=now + 3600,
            metadata={"key": "value"}
        )

        assert memory.id == "test-id"
        assert memory.content == "Test content"
        assert memory.tier == MemoryTier.PROJECT
        assert memory.score == 0.85
        assert memory.metadata == {"key": "value"}

    def test_memory_is_expired(self):
        """Test is_expired property."""
        now = time.time()

        # Not expired
        memory = Memory(
            id="1", content="", tier=MemoryTier.PROJECT, score=0.5,
            created_at=now, expires_at=now + 3600, metadata={}
        )
        assert not memory.is_expired

        # Expired
        expired_memory = Memory(
            id="2", content="", tier=MemoryTier.PROJECT, score=0.5,
            created_at=now - 7200, expires_at=now - 3600, metadata={}
        )
        assert expired_memory.is_expired

    def test_memory_ttl_remaining(self):
        """Test ttl_remaining property."""
        now = time.time()

        memory = Memory(
            id="1", content="", tier=MemoryTier.PROJECT, score=0.5,
            created_at=now, expires_at=now + 3600, metadata={}
        )

        # Should be close to 3600 (within a few seconds)
        assert 3590 < memory.ttl_remaining <= 3600

        # Expired memory should have 0 TTL remaining
        expired = Memory(
            id="2", content="", tier=MemoryTier.PROJECT, score=0.5,
            created_at=now - 7200, expires_at=now - 3600, metadata={}
        )
        assert expired.ttl_remaining == 0


# --- ProjectMemory Tests ---

class TestProjectMemoryRemember:
    """Tests for ProjectMemory.remember()."""

    def test_remember_returns_id(self, memory_with_mocks):
        """Test that remember returns a valid ID."""
        memory, mock_db, mock_collection = memory_with_mocks

        memory_id = memory.remember("Test fact")

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) == 36  # UUID format

    def test_remember_uses_correct_tier(self, memory_with_mocks):
        """Test that remember uses the specified tier."""
        memory, mock_db, mock_collection = memory_with_mocks

        memory.remember("User preference", tier=MemoryTier.USER)

        # Verify upsert was called on the USER tier collection
        user_collection = memory._collections[MemoryTier.USER]
        user_collection.upsert.assert_called()
        call_args = user_collection.upsert.call_args
        metadata = call_args.kwargs.get('metadata', call_args[1].get('metadata', {}))
        assert metadata.get('tier') == 'user'

    def test_remember_custom_ttl(self, memory_with_mocks):
        """Test that custom TTL overrides tier default."""
        memory, mock_db, mock_collection = memory_with_mocks

        custom_ttl = 60  # 1 minute
        memory.remember("Short-lived", tier=MemoryTier.PROJECT, ttl=custom_ttl)

        call_args = mock_collection.upsert.call_args
        metadata = call_args.kwargs.get('metadata', call_args[1].get('metadata', {}))

        # Check that expires_at is approximately now + custom_ttl
        now = time.time()
        assert metadata['expires_at'] < now + custom_ttl + 5
        assert metadata['expires_at'] > now + custom_ttl - 5

    def test_remember_syncs_db(self, memory_with_mocks):
        """Test that remember calls db.sync()."""
        memory, mock_db, mock_collection = memory_with_mocks

        memory.remember("Test fact")

        mock_db.sync.assert_called()


class TestProjectMemoryRecall:
    """Tests for ProjectMemory.recall()."""

    def test_recall_returns_list(self, memory_with_mocks):
        """Test that recall returns a list."""
        memory, mock_db, mock_collection = memory_with_mocks
        mock_collection.search.return_value = []

        results = memory.recall("test query")

        assert isinstance(results, list)

    def test_recall_filters_by_threshold(self, memory_with_mocks):
        """Test that recall filters results below threshold."""
        memory, mock_db, mock_collection = memory_with_mocks

        # Mock search results with various scores
        mock_result_high = Mock()
        mock_result_high.id = "high-score"
        mock_result_high.score = 0.9
        mock_result_high.metadata = {
            "content": "High score content",
            "tier": "project",
            "created_at": time.time(),
            "expires_at": time.time() + 3600
        }

        mock_result_low = Mock()
        mock_result_low.id = "low-score"
        mock_result_low.score = 0.1
        mock_result_low.metadata = {
            "content": "Low score content",
            "tier": "project",
            "created_at": time.time(),
            "expires_at": time.time() + 3600
        }

        mock_collection.search.return_value = [mock_result_high, mock_result_low]

        # With default threshold of 0.3
        results = memory.recall("test", threshold=0.3)

        # Should only include high score result
        assert len(results) == 1
        assert results[0].id == "high-score"

    def test_recall_excludes_expired_by_default(self, memory_with_mocks):
        """Test that recall excludes expired memories by default."""
        memory, mock_db, mock_collection = memory_with_mocks

        now = time.time()

        mock_result_valid = Mock()
        mock_result_valid.id = "valid"
        mock_result_valid.score = 0.9
        mock_result_valid.metadata = {
            "content": "Valid",
            "tier": "project",
            "created_at": now,
            "expires_at": now + 3600
        }

        mock_result_expired = Mock()
        mock_result_expired.id = "expired"
        mock_result_expired.score = 0.9
        mock_result_expired.metadata = {
            "content": "Expired",
            "tier": "project",
            "created_at": now - 7200,
            "expires_at": now - 3600
        }

        mock_collection.search.return_value = [mock_result_valid, mock_result_expired]

        results = memory.recall("test")

        assert len(results) == 1
        assert results[0].id == "valid"

    def test_recall_includes_expired_when_requested(self, memory_with_mocks):
        """Test that recall can include expired memories."""
        memory, mock_db, mock_collection = memory_with_mocks

        now = time.time()

        mock_result_expired = Mock()
        mock_result_expired.id = "expired"
        mock_result_expired.score = 0.9
        mock_result_expired.metadata = {
            "content": "Expired",
            "tier": "project",
            "created_at": now - 7200,
            "expires_at": now - 3600
        }

        mock_collection.search.return_value = [mock_result_expired]

        results = memory.recall("test", include_expired=True)

        assert len(results) == 1
        assert results[0].id == "expired"


class TestProjectMemoryForget:
    """Tests for ProjectMemory.forget()."""

    def test_forget_returns_count(self, memory_with_mocks):
        """Test that forget returns number of deleted memories."""
        memory, mock_db, mock_collection = memory_with_mocks
        mock_collection.search.return_value = []

        removed = memory.forget("test query")

        assert isinstance(removed, int)
        assert removed == 0

    def test_forget_deletes_matching_memories(self, memory_with_mocks):
        """Test that forget deletes memories matching the query."""
        memory, mock_db, mock_collection = memory_with_mocks

        mock_result = Mock()
        mock_result.id = "to-delete"
        mock_result.score = 0.9
        mock_result.metadata = {
            "content": "Delete me",
            "tier": "project",
            "created_at": time.time(),
            "expires_at": time.time() + 3600
        }

        mock_collection.search.return_value = [mock_result]

        removed = memory.forget("Delete me", threshold=0.8)

        assert removed == 1
        mock_collection.delete.assert_called_with("to-delete")


class TestProjectMemoryStats:
    """Tests for ProjectMemory.get_stats()."""

    def test_get_stats_returns_dict(self, memory_with_mocks):
        """Test that get_stats returns a dictionary."""
        memory, mock_db, mock_collection = memory_with_mocks

        stats = memory.get_stats()

        assert isinstance(stats, dict)
        assert "path" in stats
        assert "tiers" in stats
        assert "total_memories" in stats

    def test_get_stats_includes_all_tiers(self, memory_with_mocks):
        """Test that stats include all memory tiers."""
        memory, mock_db, mock_collection = memory_with_mocks

        stats = memory.get_stats()

        for tier in MemoryTier:
            assert tier.value in stats["tiers"]


# --- Embedder Tests ---

class TestEmbedders:
    """Tests for the pluggable embedder system."""

    def test_mock_embedder_interface(self, mock_embedder):
        """Test that MockEmbedder implements the Embedder interface."""
        assert hasattr(mock_embedder, 'embed')
        assert hasattr(mock_embedder, 'embed_single')
        assert hasattr(mock_embedder, 'dimension')
        assert mock_embedder.dimension == 384

    def test_mock_embedder_embed_single(self, mock_embedder):
        """Test embed_single returns correct dimension."""
        embedding = mock_embedder.embed_single("test")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_mock_embedder_embed_batch(self, mock_embedder):
        """Test embed returns list of embeddings."""
        embeddings = mock_embedder.embed(["test1", "test2", "test3"])
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_mock_embedder_deterministic(self, mock_embedder):
        """Test that same text produces same embedding."""
        embedding1 = mock_embedder.embed_single("hello world")
        embedding2 = mock_embedder.embed_single("hello world")
        assert embedding1 == embedding2

    def test_mock_embedder_different_texts(self, mock_embedder):
        """Test that different texts produce different embeddings."""
        embedding1 = mock_embedder.embed_single("hello")
        embedding2 = mock_embedder.embed_single("world")
        assert embedding1 != embedding2


class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_create_embedder_with_callable(self):
        """Test creating embedder from custom function."""
        from agentvec_memory import create_embedder, CallableEmbedder

        def custom_embed(texts):
            return [[0.1] * 128 for _ in texts]

        embedder = create_embedder(custom_embed, dimension=128)
        assert isinstance(embedder, CallableEmbedder)
        assert embedder.dimension == 128

    def test_create_embedder_callable_requires_dimension(self):
        """Test that callable embedder requires dimension."""
        from agentvec_memory import create_embedder

        def custom_embed(texts):
            return [[0.1] * 128 for _ in texts]

        with pytest.raises(ValueError, match="dimension is required"):
            create_embedder(custom_embed)

    def test_create_embedder_with_embedder_instance(self, mock_embedder):
        """Test that passing Embedder instance returns it unchanged."""
        from agentvec_memory import create_embedder

        result = create_embedder(mock_embedder)
        assert result is mock_embedder

    def test_create_embedder_invalid_type(self):
        """Test that invalid type raises TypeError."""
        from agentvec_memory import create_embedder

        with pytest.raises(TypeError):
            create_embedder(12345)


# --- Integration Tests (require real dependencies) ---

@pytest.mark.integration
class TestProjectMemoryIntegration:
    """Integration tests that require real agentvec and fastembed/sentence-transformers."""

    @pytest.fixture
    def real_memory(self):
        """Create a real ProjectMemory instance for integration tests."""
        try:
            import agentvec
        except ImportError:
            pytest.skip("Integration tests require agentvec")

        # Try fastembed first, then sentence-transformers
        try:
            from fastembed import TextEmbedding
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                pytest.skip("Integration tests require fastembed or sentence-transformers")

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ProjectMemory(tmpdir)
            yield memory

    def test_remember_and_recall_roundtrip(self, real_memory):
        """Test that we can remember and recall a fact."""
        content = "The API key is stored in .env file"

        memory_id = real_memory.remember(content, tier=MemoryTier.PROJECT)
        results = real_memory.recall("where is the API key")

        assert len(results) > 0
        assert any(content in r.content for r in results)

    def test_forget_removes_memory(self, real_memory):
        """Test that forget actually removes memories."""
        content = "Temporary fact to forget"

        real_memory.remember(content, tier=MemoryTier.WORKING)

        # Verify it exists
        results = real_memory.recall(content)
        assert len(results) > 0

        # Forget it
        removed = real_memory.forget(content, threshold=0.7)
        assert removed > 0

        # Verify it's gone
        results = real_memory.recall(content)
        assert len(results) == 0 or all(r.score < 0.7 for r in results)

    def test_custom_embedder(self):
        """Test using a custom embedder function."""
        try:
            import agentvec
        except ImportError:
            pytest.skip("Requires agentvec")

        def simple_embedder(texts):
            """Simple hash-based embedder for testing."""
            result = []
            for text in texts:
                # Create a simple deterministic embedding
                np.random.seed(hash(text) % (2**32))
                result.append(np.random.randn(64).tolist())
            return result

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ProjectMemory(tmpdir, embedder=simple_embedder, dimension=64)
            assert memory.dimension == 64

            memory.remember("Test fact", tier=MemoryTier.PROJECT)
            results = memory.recall("Test")
            assert len(results) > 0
