"""HTML-to-clean-text extraction, URL/content deduplication, and factual claim extraction."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

from research_agent.models import CleanedDocument, ExtractedClaim, SearchResult

BOILERPLATE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "svg")
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "source",
}

POSITIVE_CUES = ("reduces", "improved", "improves", "lower", "faster", "succeed", "delivered", "preserving", "eliminates", "cuts")
NEGATIVE_CUES = ("however", "drawback", "degrades", "contrary", "staleness", "risk", "leakage", "adds", "inflates", "unbounded", "sparse")
METRIC_PATTERN = re.compile(
    r"(?:\$\d[\d,]*(?:\.\d+)?(?:\s*-\s*\$?\d[\d,]*(?:\.\d+)?)?|\b\d+(?:\.\d+)?\s*(?:%|(?:ms|x|B|M|k|million|seconds|s)\b))",
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "into", "over", "than", "when", "where", "while",
    "their", "they", "them", "more", "most", "less", "also", "such", "without",
    "compared", "across", "within", "about", "been", "being", "can", "will",
}


def html_to_clean_text(html_or_text: str) -> str:
    """Convert raw HTML into normalized clean text stripped of navigation/script boilerplate."""
    if not html_or_text:
        return ""
    soup = BeautifulSoup(html_or_text, "html.parser")
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    raw_text = soup.get_text(separator="\n", strip=True)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
    return "\n".join(line for line in lines if line)


def canonicalize_url(url: str) -> str:
    """Normalize URL casing, strip tracking query parameters, fragments, and trailing slash."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    filtered_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    filtered_query.sort()
    query_str = urlencode(filtered_query)
    return urlunparse((scheme, netloc, path, "", query_str, ""))


def _content_hash(text: str) -> str:
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _shingles(text: str, k: int = 4) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    sa = _shingles(text_a)
    sb = _shingles(text_b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate_documents(
    docs: list[CleanedDocument],
    *,
    near_dup_threshold: float = 0.80,
) -> list[CleanedDocument]:
    """Remove exact URL duplicates, exact content-hash duplicates, and near-duplicate texts."""
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    kept: list[CleanedDocument] = []

    for doc in docs:
        if doc.canonical_url and doc.canonical_url in seen_urls:
            continue
        if doc.content_hash in seen_hashes:
            continue
        if any(jaccard_similarity(doc.clean_text, existing.clean_text) >= near_dup_threshold for existing in kept):
            continue

        if doc.canonical_url:
            seen_urls.add(doc.canonical_url)
        seen_hashes.add(doc.content_hash)
        kept.append(doc)

    # Re-index sequential 1-based citation IDs
    reindexed: list[CleanedDocument] = []
    for idx, doc in enumerate(kept, start=1):
        reindexed.append(doc.model_copy(update={"source_id": idx}))
    return reindexed


def extract_documents(
    results: list[SearchResult],
    *,
    near_dup_threshold: float = 0.80,
) -> list[CleanedDocument]:
    """Clean HTML payloads and deduplicate raw search results into numbered sources."""
    preliminary: list[CleanedDocument] = []
    for idx, hit in enumerate(results, start=1):
        text = html_to_clean_text(hit.raw_html) if hit.raw_html else html_to_clean_text(hit.snippet)
        if not text:
            continue
        canon = canonicalize_url(hit.url)
        domain = urlparse(canon or hit.url).netloc or "local"
        preliminary.append(
            CleanedDocument(
                source_id=idx,
                title=hit.title.strip() or domain,
                url=hit.url.strip(),
                canonical_url=canon,
                domain=domain,
                clean_text=text,
                content_hash=_content_hash(text),
                source_backend=hit.source_backend,
                published_date=hit.published_date or "n/a",
                word_count=len(text.split()),
            )
        )
    return deduplicate_documents(preliminary, near_dup_threshold=near_dup_threshold)


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for block in text.splitlines():
        block = block.strip()
        if not block:
            continue
        # Skip short standalone headings lacking terminal punctuation
        if block[-1] not in ".!?" and len(block.split()) <= 12:
            continue
        for part in re.split(r"(?<=[.!?])\s+", block):
            cleaned = part.strip()
            if len(cleaned.split()) >= 6:
                sentences.append(cleaned)
    return sentences


def _classify_stance(sentence: str, metrics: list[str]) -> str:
    lower = sentence.lower()
    has_neg = any(cue in lower for cue in NEGATIVE_CUES)
    has_pos = any(cue in lower for cue in POSITIVE_CUES)
    if has_neg:
        return "negative"
    if metrics:
        return "metric"
    if has_pos:
        return "positive"
    return "neutral"


def _extract_keywords(sentence: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", sentence.lower())
    seen: list[str] = []
    for tok in tokens:
        if tok not in STOPWORDS and tok not in seen:
            seen.append(tok)
        if len(seen) >= 8:
            break
    return seen


def extract_claims(
    documents: list[CleanedDocument],
    *,
    max_claims_per_doc: int = 4,
) -> list[ExtractedClaim]:
    """Extract crisp factual claim candidates linked to their source metadata."""
    claims: list[ExtractedClaim] = []
    seen_claim_norms: set[str] = set()

    for doc in documents:
        doc_count = 0
        for sentence in _split_sentences(doc.clean_text):
            norm = re.sub(r"\W+", " ", sentence.lower()).strip()
            if norm in seen_claim_norms:
                continue
            seen_claim_norms.add(norm)

            metrics = METRIC_PATTERN.findall(sentence)
            stance = _classify_stance(sentence, metrics)
            claims.append(
                ExtractedClaim(
                    claim_text=sentence,
                    source_id=doc.source_id,
                    title=doc.title,
                    url=doc.url,
                    domain=doc.domain,
                    stance=stance,  # type: ignore[arg-type]
                    metrics=metrics,
                    keywords=_extract_keywords(sentence),
                )
            )
            doc_count += 1
            if doc_count >= max_claims_per_doc:
                break

    return claims
