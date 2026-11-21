"""Tests for HTML cleaning, URL/content deduplication, claim extraction, and SQLite cache."""

from __future__ import annotations

import time
from pathlib import Path

from research_agent.cache import ResearchCache
from research_agent.extractor import (
    canonicalize_url,
    extract_claims,
    extract_documents,
    html_to_clean_text,
)
from research_agent.models import SearchResult


def test_html_to_clean_text_strips_boilerplate_tags() -> None:
    html = (
        "<html><head><style>body{color:red}</style><script>alert(1)</script></head>"
        "<body><nav>Menu Home</nav><article><p>RAG reduces hallucinations by 42%.</p></article>"
        "<footer>Copyright 2026</footer></body></html>"
    )
    cleaned = html_to_clean_text(html)
    assert "RAG reduces hallucinations by 42%." in cleaned
    assert "alert(1)" not in cleaned
    assert "Menu Home" not in cleaned
    assert "Copyright 2026" not in cleaned


def test_canonicalize_url_strips_utm_and_trailing_slash() -> None:
    u1 = canonicalize_url("HTTPS://Example.ORG/blog/post/?utm_source=x&b=2&a=1&fbclid=99#top")
    u2 = canonicalize_url("https://example.org/blog/post?a=1&b=2")
    assert u1 == u2
    assert "utm_source" not in u1
    assert "fbclid" not in u1


def test_deduplication_removes_url_and_near_duplicate_content() -> None:
    results = [
        SearchResult(
            title="Primary Study",
            url="https://site.example/article?utm_source=a",
            raw_html="<p>LoRA fine tuning lowers inference latency by 35% on high volume workloads.</p>",
            source_backend="fixture",
        ),
        SearchResult(
            title="UTM Duplicate",
            url="https://site.example/article/?utm_medium=b",
            raw_html="<p>Different text but same canonical URL should be dropped.</p>",
            source_backend="fixture",
        ),
        SearchResult(
            title="Syndicated Near-Duplicate",
            url="https://mirror.example/repost",
            raw_html="<p>LoRA fine tuning lowers inference latency by 35% on high volume workloads today.</p>",
            source_backend="fixture",
        ),
        SearchResult(
            title="Distinct Source",
            url="https://other.example/rag-guide",
            raw_html="<p>Retrieval pipelines enable instant compliance deletion without retraining weights.</p>",
            source_backend="fixture",
        ),
    ]
    docs = extract_documents(results, near_dup_threshold=0.70)
    assert len(docs) == 2
    assert [d.source_id for d in docs] == [1, 2]
    assert docs[0].domain == "site.example"
    assert docs[1].domain == "other.example"


def test_extract_claims_captures_metrics_stance_and_metadata() -> None:
    results = [
        SearchResult(
            title="Latency Study",
            url="https://bench.example/latency",
            raw_html=(
                "<p>Fine-tuning cut p95 latency by 35% across 12,000 queries.</p>"
                "<p>However, knowledge staleness remains a major drawback for rapidly changing data.</p>"
            ),
            source_backend="fixture",
        )
    ]
    docs = extract_documents(results)
    claims = extract_claims(docs)
    assert len(claims) == 2
    assert claims[0].source_id == 1
    assert "35%" in claims[0].metrics
    assert claims[0].stance == "metric"
    assert claims[1].stance == "negative"
    assert claims[1].domain == "bench.example"


def test_sqlite_cache_hit_miss_and_ttl_expiry(tmp_path: Path) -> None:
    db_file = tmp_path / "cache.sqlite3"
    cache = ResearchCache(db_path=db_file, ttl_seconds=60)
    assert cache.get("fixture", "RAG vs LoRA", 5) is None

    sample = [SearchResult(title="Doc 1", url="https://a.example/1", snippet="hello", source_backend="fixture")]
    cache.set("fixture", "  RAG   vs LoRA ", 5, sample)

    # Case-insensitive whitespace-normalized hit
    hit = cache.get("fixture", "rag vs lora", 5)
    assert hit is not None
    assert len(hit) == 1
    assert hit[0].title == "Doc 1"
    assert cache.count() == 1

    # Expired TTL entry returns None and is purged
    cache.set("fixture", "expired topic", 5, sample, created_at=time.time() - 120)
    assert cache.get("fixture", "expired topic", 5) is None
