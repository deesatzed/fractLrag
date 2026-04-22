"""
Fractal Latent RAG — Multi-scale retrieval with derivative signals.

Usage:
    from fractrag import FractalRAG, HashEmbedding, SentenceTransformerEmbedding, DocumentProfile

    # With real embeddings
    rag = FractalRAG(backend=SentenceTransformerEmbedding("BAAI/bge-m3"))
    rag.add_document("doc1", "Your text here...")
    results, query_type = rag.retrieve("Your question here")

    # With hash embeddings (testing only)
    rag = FractalRAG(backend=HashEmbedding(dim=64))
"""

from .core import EmbeddingBackend, HashEmbedding, SentenceTransformerEmbedding, normalize
from .engine import FractalRAG, IndexEntry
from .profile import DocumentProfile
from .query import classify_query_type, get_type_weights, extract_domain_hints
from .storage import save, load, DimensionMismatchError

__all__ = [
    "FractalRAG",
    "IndexEntry",
    "EmbeddingBackend",
    "HashEmbedding",
    "SentenceTransformerEmbedding",
    "DocumentProfile",
    "classify_query_type",
    "get_type_weights",
    "extract_domain_hints",
    "normalize",
    "save",
    "load",
    "DimensionMismatchError",
]
