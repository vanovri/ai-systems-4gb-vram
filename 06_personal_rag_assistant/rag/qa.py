"""Grounded QA engine with inline [source#chunk_id] citations and anti-hallucination refusal."""

from __future__ import annotations

import os
import re
from typing import Literal

import httpx

from rag.embeddings import tokenize
from rag.models import Citation, QAResponse, SearchResult
from rag.store import HybridVectorStore

REFUSAL_MESSAGE = (
    "I cannot answer this question based on the indexed documents. "
    "The retrieved context does not provide sufficient supporting evidence."
)

STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "can",
        "could",
        "describe",
        "did",
        "do",
        "does",
        "explain",
        "for",
        "from",
        "give",
        "has",
        "have",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "of",
        "on",
        "or",
        "should",
        "tell",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
    }
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def extract_content_terms(text: str) -> set[str]:
    """Return meaningful non-stopword tokens for grounding verification."""
    return {tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1}


def _split_sentences(text: str) -> list[str]:
    clean_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # Skip standalone markdown headers; preserve heading+body lines cleanly
            if "." not in stripped:
                continue
            stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        clean_lines.append(stripped)
    body = " ".join(clean_lines)
    parts = [p.strip(" \t\r\n-*#") for p in SENTENCE_SPLIT_RE.split(body)]
    full_sentences = [p for p in parts if len(p) >= 24 and not p[0].islower()]
    if full_sentences:
        return full_sentences
    return [p for p in parts if len(p) >= 24]


class GroundedQAEngine:
    """Answers queries strictly from retrieved context with enforced inline citations."""

    def __init__(
        self,
        store: HybridVectorStore,
        min_dense_score: float = 0.24,
        min_bm25_score: float = 0.55,
        min_term_coverage: float = 0.25,
        llm_mode: Literal["auto", "extractive", "openai"] = "auto",
        http_client: httpx.Client | None = None,
    ) -> None:
        self.store = store
        self.min_dense_score = min_dense_score
        self.min_bm25_score = min_bm25_score
        self.min_term_coverage = min_term_coverage
        self.llm_mode = llm_mode
        self._http_client = http_client

    def _assess_grounding(
        self, query: str, results: list[SearchResult]
    ) -> tuple[bool, float, list[SearchResult]]:
        """Evaluate whether retrieved chunks genuinely support the query."""
        if not results:
            return False, 0.0, []

        q_terms = extract_content_terms(query)
        best_dense = max(r.dense_score for r in results)
        best_bm25 = max(r.bm25_score for r in results)

        supporting: list[SearchResult] = []
        best_coverage = 0.0

        for res in results:
            c_terms = extract_content_terms(res.chunk.text)
            if q_terms:
                overlap = len(q_terms & c_terms)
                coverage = overlap / len(q_terms)
            else:
                coverage = 1.0 if res.dense_score >= self.min_dense_score else 0.0

            best_coverage = max(best_coverage, coverage)
            is_chunk_supported = (
                (coverage >= self.min_term_coverage and res.bm25_score >= self.min_bm25_score)
                or (coverage > 0.0 and res.dense_score >= self.min_dense_score)
                or (res.dense_score >= self.min_dense_score + 0.22)
            )
            if is_chunk_supported:
                supporting.append(res)

        # Calibrated confidence score in [0.0, 1.0]
        dense_norm = max(0.0, min(1.0, best_dense))
        bm25_norm = max(0.0, min(1.0, best_bm25 / 6.0))
        confidence = round(0.45 * dense_norm + 0.35 * best_coverage + 0.20 * bm25_norm, 4)

        has_support = bool(supporting) and (
            best_coverage >= self.min_term_coverage
            or (best_dense >= self.min_dense_score + 0.22 and best_bm25 >= self.min_bm25_score)
        )
        return has_support, confidence, supporting

    def _synthesize_extractive(
        self, query: str, supporting: list[SearchResult], max_sentences: int = 3
    ) -> tuple[str, list[Citation]]:
        """Deterministically extract and cite top evidence sentences with [source#chunk_id]."""
        q_terms = extract_content_terms(query)
        scored_sentences: list[tuple[float, str, SearchResult]] = []

        for rank_idx, res in enumerate(supporting):
            sentences = _split_sentences(res.chunk.text)
            if not sentences:
                sentences = [res.chunk.text.strip()]
            rank_bonus = 1.0 / (1.0 + 0.7 * rank_idx)
            for sent in sentences:
                s_terms = extract_content_terms(sent)
                if not s_terms:
                    continue
                overlap = len(q_terms & s_terms)
                coverage = (overlap / len(q_terms)) if q_terms else 0.0
                density = overlap / max(1, len(s_terms))
                sent_score = (2.4 * coverage + 0.6 * density + 0.7 * res.dense_score) * rank_bonus
                scored_sentences.append((sent_score, sent, res))

        scored_sentences.sort(key=lambda item: (-item[0], item[2].rank, item[1]))

        chosen_lines: list[str] = []
        seen_norm: set[str] = set()
        used_results: dict[str, SearchResult] = {}

        for _, sent, res in scored_sentences:
            norm = re.sub(r"\s+", " ", sent.lower()).strip()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            clean_sent = sent.rstrip(".")
            chosen_lines.append(f"{clean_sent}. {res.chunk.citation_tag}")
            used_results[res.chunk.chunk_id] = res
            if len(chosen_lines) >= max_sentences:
                break

        if not chosen_lines:
            top = supporting[0]
            excerpt = top.chunk.text[:240].strip().rstrip(".")
            chosen_lines.append(f"{excerpt}. {top.chunk.citation_tag}")
            used_results[top.chunk.chunk_id] = top

        citations = [
            Citation(
                tag=res.chunk.citation_tag,
                source_file=res.chunk.source_file,
                chunk_id=res.chunk.chunk_id,
                page=res.chunk.page,
                relevance_score=round(res.score, 6),
                excerpt=res.chunk.text[:180].strip(),
            )
            for res in sorted(used_results.values(), key=lambda r: r.rank)
        ]
        return " ".join(chosen_lines), citations

    def _synthesize_openai(
        self, query: str, supporting: list[SearchResult]
    ) -> tuple[str, list[Citation]] | None:
        """Optional OpenAI-compatible grounded generator with strict citation verification."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        allowed_tags = {res.chunk.citation_tag for res in supporting}

        context_block = "\n\n".join(
            f"{res.chunk.citation_tag} (page {res.chunk.page}):\n{res.chunk.text}"
            for res in supporting
        )
        system_prompt = (
            "You are a strict grounded assistant. Answer ONLY from the provided context. "
            "Cite every factual claim inline using the exact bracket tag [source_file#chunk_id]. "
            "If the context does not answer the question, respond with exact text: "
            f'"{REFUSAL_MESSAGE}"'
        )
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{base_url}/chat/completions"

        try:
            if self._http_client is not None:
                resp = self._http_client.post(url, json=payload, headers=headers)
            else:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

        cited_tags = set(re.findall(r"\[[^\[\]]+#[^\[\]]+\]", answer))
        valid_cited = cited_tags & allowed_tags
        if not valid_cited:
            return None

        citations = [
            Citation(
                tag=res.chunk.citation_tag,
                source_file=res.chunk.source_file,
                chunk_id=res.chunk.chunk_id,
                page=res.chunk.page,
                relevance_score=round(res.score, 6),
                excerpt=res.chunk.text[:180].strip(),
            )
            for res in supporting
            if res.chunk.citation_tag in valid_cited
        ]
        return answer, citations

    def ask(
        self,
        query: str,
        top_k: int = 4,
        search_mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    ) -> QAResponse:
        """Retrieve relevant chunks and produce a grounded, cited answer or explicit refusal."""
        results = self.store.search(query, top_k=top_k, mode=search_mode)
        supported, confidence, supporting = self._assess_grounding(query, results)

        if not supported:
            return QAResponse(
                query=query,
                answer=REFUSAL_MESSAGE,
                refused=True,
                confidence=confidence,
                citations=[],
                retrieved=results,
                mode="refusal",
            )

        use_openai = self.llm_mode == "openai" or (
            self.llm_mode == "auto" and bool(os.getenv("OPENAI_API_KEY", "").strip())
        )
        if use_openai:
            llm_out = self._synthesize_openai(query, supporting)
            if llm_out is not None:
                answer, citations = llm_out
                return QAResponse(
                    query=query,
                    answer=answer,
                    refused=False,
                    confidence=confidence,
                    citations=citations,
                    retrieved=results,
                    mode="openai",
                )

        answer, citations = self._synthesize_extractive(query, supporting)
        return QAResponse(
            query=query,
            answer=answer,
            refused=False,
            confidence=confidence,
            citations=citations,
            retrieved=results,
            mode="extractive",
        )
