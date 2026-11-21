"""Personal RAG Assistant with hybrid FAISS + BM25 + RRF retrieval and grounded citations."""

from rag.embeddings import (
    BaseEmbedder,
    LocalHybridEmbedder,
    OpenAICompatibleEmbedder,
    create_embedder,
)
from rag.ingest import chunk_text, ingest_directory, ingest_file, ingest_raw_document
from rag.models import Citation, DocumentChunk, QAResponse, SearchResult
from rag.qa import GroundedQAEngine
from rag.store import BM25Index, HybridVectorStore

__all__ = [
    "BM25Index",
    "BaseEmbedder",
    "Citation",
    "DocumentChunk",
    "GroundedQAEngine",
    "HybridVectorStore",
    "LocalHybridEmbedder",
    "OpenAICompatibleEmbedder",
    "QAResponse",
    "SearchResult",
    "chunk_text",
    "create_embedder",
    "ingest_directory",
    "ingest_file",
    "ingest_raw_document",
]
