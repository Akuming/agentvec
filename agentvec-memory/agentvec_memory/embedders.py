"""
Pluggable embedding providers for agentvec-memory.

Supports multiple backends:
- fastembed (default): Lightweight ONNX-based, ~100MB install
- sentence-transformers: Full PyTorch, GPU support, ~3GB install
- Custom: Bring your own embedding function

Example:
    # Use default (fastembed)
    memory = ProjectMemory("./memory")

    # Explicitly choose backend
    memory = ProjectMemory("./memory", embedder="fastembed")
    memory = ProjectMemory("./memory", embedder="sentence-transformers")

    # Custom embedding function
    memory = ProjectMemory("./memory", embedder=my_embed_func)

    # Custom embedder class
    memory = ProjectMemory("./memory", embedder=MyCustomEmbedder())
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional
import logging

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed([text])[0]


class FastEmbedEmbedder(Embedder):
    """
    Lightweight embedder using fastembed (ONNX Runtime).

    This is the recommended default - only ~100MB install vs ~3GB for PyTorch.

    Supported models:
    - sentence-transformers/all-MiniLM-L6-v2 (default, 384 dim)
    - BAAI/bge-small-en-v1.5 (384 dim)
    - BAAI/bge-base-en-v1.5 (768 dim)
    - And many more: https://qdrant.github.io/fastembed/examples/Supported_Models/
    """

    # Model name -> dimension mapping for common models
    MODEL_DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/paraphrase-MiniLM-L3-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize FastEmbed embedder.

        Args:
            model_name: Model to use. Defaults to all-MiniLM-L6-v2.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is required for FastEmbedEmbedder. "
                "Install with: pip install fastembed"
            )

        self._model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"Loading fastembed model: {self._model_name}")
        self._model = TextEmbedding(self._model_name)

        # Get dimension from known models or detect it
        if self._model_name in self.MODEL_DIMENSIONS:
            self._dimension = self.MODEL_DIMENSIONS[self._model_name]
        else:
            # Detect dimension by embedding a test string
            test_embedding = list(self._model.embed(["test"]))[0]
            self._dimension = len(test_embedding)

        logger.info(f"FastEmbed ready: {self._model_name} ({self._dimension} dim)")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using fastembed."""
        # fastembed returns a generator, convert to list
        embeddings = list(self._model.embed(texts))
        return [list(e) for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class SentenceTransformerEmbedder(Embedder):
    """
    Full-featured embedder using sentence-transformers (PyTorch).

    Use this when you need:
    - GPU acceleration
    - Fine-tuned models
    - Maximum compatibility

    Note: This requires ~3GB of dependencies (PyTorch + transformers).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize SentenceTransformer embedder.

        Args:
            model_name: Model to use. Defaults to all-MiniLM-L6-v2.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"Loading sentence-transformers model: {self._model_name}")

        if device:
            self._model = SentenceTransformer(self._model_name, device=device)
        else:
            self._model = SentenceTransformer(self._model_name)

        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"SentenceTransformer ready: {self._model_name} "
            f"({self._dimension} dim, device={self._model.device})"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using sentence-transformers."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return str(self._model.device)


class CallableEmbedder(Embedder):
    """
    Wrapper for custom embedding functions.

    Use this when you want to provide your own embedding logic,
    such as calling an external API (OpenAI, Cohere, etc.).

    Example:
        def my_embedder(texts):
            # Call your embedding API
            return [[0.1, 0.2, ...] for _ in texts]

        memory = ProjectMemory("./memory", embedder=my_embedder)
    """

    def __init__(
        self,
        embed_func: Callable[[List[str]], List[List[float]]],
        dimension: int,
    ):
        """
        Initialize callable embedder.

        Args:
            embed_func: Function that takes list of strings, returns list of embeddings.
            dimension: The embedding dimension (must be known upfront).
        """
        self._embed_func = embed_func
        self._dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the custom function."""
        return self._embed_func(texts)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(Embedder):
    """
    Embedder using OpenAI's embedding API.

    Requires: pip install openai
    Requires: OPENAI_API_KEY environment variable

    Example:
        memory = ProjectMemory("./memory", embedder="openai")
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize OpenAI embedder.

        Args:
            model_name: Model to use. Defaults to text-embedding-3-small.
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIEmbedder. "
                "Install with: pip install openai"
            )

        self._model_name = model_name or self.DEFAULT_MODEL
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._dimension = self.MODEL_DIMENSIONS.get(self._model_name, 1536)

        logger.info(f"OpenAI embedder ready: {self._model_name} ({self._dimension} dim)")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI API."""
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


# --- Factory Functions ---

def get_default_embedder() -> Embedder:
    """
    Get the best available embedder.

    Priority:
    1. fastembed (lightweight, recommended)
    2. sentence-transformers (full-featured, if installed)

    Raises:
        ImportError: If no embedding provider is available.
    """
    # Try fastembed first (lightweight)
    try:
        return FastEmbedEmbedder()
    except ImportError:
        logger.debug("fastembed not available, trying sentence-transformers")

    # Fall back to sentence-transformers
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        logger.debug("sentence-transformers not available")

    raise ImportError(
        "No embedding provider found. Install one of:\n"
        "  pip install fastembed              # Lightweight (~100MB), recommended\n"
        "  pip install sentence-transformers  # Full-featured (~3GB), GPU support"
    )


def get_embedder_by_name(name: str, **kwargs) -> Embedder:
    """
    Get an embedder by name.

    Args:
        name: Embedder name ('fastembed', 'sentence-transformers', 'openai')
        **kwargs: Additional arguments passed to the embedder constructor.

    Returns:
        Configured Embedder instance.

    Raises:
        ValueError: If the embedder name is unknown.
        ImportError: If the required package is not installed.
    """
    name_lower = name.lower().replace("-", "").replace("_", "")

    if name_lower in ("fastembed", "fast"):
        return FastEmbedEmbedder(**kwargs)
    elif name_lower in ("sentencetransformers", "sentence", "st", "sbert"):
        return SentenceTransformerEmbedder(**kwargs)
    elif name_lower in ("openai", "oai"):
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(
            f"Unknown embedder: '{name}'. "
            f"Supported: 'fastembed', 'sentence-transformers', 'openai'"
        )


def create_embedder(
    embedder: Union[Embedder, Callable, str, None] = None,
    dimension: Optional[int] = None,
    **kwargs,
) -> Embedder:
    """
    Create an embedder from various input types.

    Args:
        embedder: Can be:
            - None: Use default (fastembed or sentence-transformers)
            - str: Embedder name ('fastembed', 'sentence-transformers', 'openai')
            - Callable: Custom function (requires dimension parameter)
            - Embedder: Use as-is

        dimension: Required if embedder is a callable.
        **kwargs: Additional arguments for named embedders.

    Returns:
        Configured Embedder instance.

    Example:
        # Auto-detect best available
        embedder = create_embedder()

        # By name
        embedder = create_embedder("fastembed")
        embedder = create_embedder("openai", model_name="text-embedding-3-large")

        # Custom function
        embedder = create_embedder(my_func, dimension=384)
    """
    if embedder is None:
        return get_default_embedder()

    if isinstance(embedder, str):
        return get_embedder_by_name(embedder, **kwargs)

    if isinstance(embedder, Embedder):
        return embedder

    if callable(embedder):
        if dimension is None:
            raise ValueError(
                "dimension is required when using a custom embedding function. "
                "Example: create_embedder(my_func, dimension=384)"
            )
        return CallableEmbedder(embedder, dimension=dimension)

    raise TypeError(
        f"embedder must be None, str, Embedder, or callable, got {type(embedder)}"
    )
