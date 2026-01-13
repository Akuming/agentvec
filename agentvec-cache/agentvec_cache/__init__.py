"""
agentvec-cache: Semantic caching for LLM responses.

Cut your LLM costs by caching semantically similar queries.
Built on AgentVec.
"""

from .cache import SemanticCache, CacheStats

__version__ = "0.1.0"
__all__ = ["SemanticCache", "CacheStats"]
