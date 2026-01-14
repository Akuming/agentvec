"""
agentvec-search: Semantic code search for your codebase.

Search code with natural language instead of regex.
"""

from .indexer import CodeIndexer, IndexStats
from .cli import main

__version__ = "0.1.0"
__all__ = ["CodeIndexer", "IndexStats", "main"]
