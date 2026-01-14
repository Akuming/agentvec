"""
AgentVec Memory - Persistent tiered memory for AI agents.

Supports multiple embedding backends:
- fastembed (default): Lightweight ONNX-based, ~100MB install
- sentence-transformers: Full PyTorch, GPU support, ~3GB install
- Custom: Bring your own embedding function

Install options:
    pip install agentvec-memory              # Lightweight (fastembed)
    pip install agentvec-memory[gpu]         # With sentence-transformers
    pip install agentvec-memory[openai]      # With OpenAI embeddings
    pip install agentvec-memory[all]         # All backends
"""

from .memory import ProjectMemory, MemoryTier, Memory
from .embedders import (
    Embedder,
    FastEmbedEmbedder,
    SentenceTransformerEmbedder,
    CallableEmbedder,
    OpenAIEmbedder,
    create_embedder,
    get_default_embedder,
)

__all__ = [
    # Core
    "ProjectMemory",
    "MemoryTier",
    "Memory",
    # Embedders
    "Embedder",
    "FastEmbedEmbedder",
    "SentenceTransformerEmbedder",
    "CallableEmbedder",
    "OpenAIEmbedder",
    "create_embedder",
    "get_default_embedder",
]
__version__ = "0.2.0"
