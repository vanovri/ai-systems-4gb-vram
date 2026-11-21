"""Pydantic data models for the AI Research Agent pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Raw search hit returned by any pluggable SearchBackend."""

    title: str
    url: str
    snippet: str = ""
    raw_html: str = ""
    source_backend: str = "unknown"
    published_date: str = "n/a"
    score: float = 0.0


class CleanedDocument(BaseModel):
    """Deduplicated, HTML-cleaned source document ready for claim extraction."""

    source_id: int
    title: str
    url: str
    canonical_url: str
    domain: str
    clean_text: str
    content_hash: str
    source_backend: str
    published_date: str = "n/a"
    word_count: int = 0


class ExtractedClaim(BaseModel):
    """Structured factual claim linked to its originating source metadata."""

    claim_text: str
    source_id: int
    title: str
    url: str
    domain: str
    stance: Literal["positive", "negative", "neutral", "metric"] = "neutral"
    metrics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class ResearchBriefResult(BaseModel):
    """Complete output of a research agent run."""

    query: str
    backend_used: str
    synthesizer_used: str
    cache_hit: bool
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_result_count: int
    deduped_source_count: int
    claim_count: int
    markdown: str
    sources: list[CleanedDocument] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    output_path: str | None = None
