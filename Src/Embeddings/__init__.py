"""
Embeddings module for semantic vector generation and storage.
"""

from .semantic_embedder import FastTextEncoder
from .vector_store import VectorStore

__all__ = [
    "FastTextEncoder",
    "VectorStore",
]
