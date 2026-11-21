"""Core data structures for the Personal RAG Assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    """Single chunk extracted from a source document with full provenance metadata."""

    chunk_id: str
    text: str
    source_file: str
    page: int
    start_char: int = 0
    end_char: int = 0

    @property
    def citation_tag(self) -> str:
        return f"[{self.source_file}#{self.chunk_id}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentChunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data["text"]),
            source_file=str(data["source_file"]),
            page=int(data.get("page", 1)),
            start_char=int(data.get("start_char", 0)),
            end_char=int(data.get("end_char", 0)),
        )


@dataclass(frozen=True)
class SearchResult:
    """Ranked retrieval hit with hybrid scoring breakdown."""

    chunk: DocumentChunk
    score: float
    dense_score: float
    bm25_score: float
    rrf_score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "citation": self.chunk.citation_tag,
            "score": round(self.score, 6),
            "dense_score": round(self.dense_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class Citation:
    """Structured provenance entry attached to a grounded QA response."""

    tag: str
    source_file: str
    chunk_id: str
    page: int
    relevance_score: float
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QAResponse:
    """Final grounded QA output with anti-hallucination refusal flag and citations."""

    query: str
    answer: str
    refused: bool
    confidence: float
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[SearchResult] = field(default_factory=list)
    mode: str = "extractive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "refused": self.refused,
            "confidence": round(self.confidence, 4),
            "mode": self.mode,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [r.to_dict() for r in self.retrieved],
        }
