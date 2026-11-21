"""Pluggable search backends: Tavily, DuckDuckGo, Wikipedia, and deterministic offline Fixture."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from bs4 import BeautifulSoup

from research_agent.models import SearchResult


class SearchBackend(ABC):
    """Abstract interface for web or local research search providers."""

    name: str = "base"

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return up to max_results SearchResult entries for query."""


class TavilySearchBackend(SearchBackend):
    """Live Tavily Search API backend (requires TAVILY_API_KEY env var)."""

    name = "tavily"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.tavily.com/search",
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._client = client

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not set; set env var or choose another backend.")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_raw_content": True,
            "search_depth": "basic",
        }
        if self._client is not None:
            resp = self._client.post(self.base_url, json=payload)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(self.base_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    raw_html=item.get("raw_content") or item.get("content", ""),
                    source_backend=self.name,
                    published_date=item.get("published_date") or "n/a",
                    score=float(item.get("score", 0.0)),
                )
            )
        return results


class WikipediaSearchBackend(SearchBackend):
    """Live public MediaWiki API backend (no API key required)."""

    name = "wikipedia"

    def __init__(
        self,
        *,
        endpoint: str = "https://en.wikipedia.org/w/api.php",
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._client = client

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "utf8": "1",
            "format": "json",
        }
        headers = {"User-Agent": "AI-Research-Agent/0.1 (educational-cli)"}

        if self._client is not None:
            resp = self._client.get(self.endpoint, params=params, headers=headers)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(self.endpoint, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("query", {}).get("search", [])
        results: list[SearchResult] = []
        for idx, hit in enumerate(hits[:max_results]):
            title = hit.get("title", "Untitled")
            slug = title.replace(" ", "_")
            url = f"https://en.wikipedia.org/wiki/{slug}"
            raw_snippet = hit.get("snippet", "")
            clean_snippet = BeautifulSoup(raw_snippet, "html.parser").get_text(" ", strip=True)
            timestamp = str(hit.get("timestamp", "n/a"))[:10]
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=clean_snippet,
                    raw_html=f"<article><h1>{title}</h1><p>{raw_snippet}</p></article>",
                    source_backend=self.name,
                    published_date=timestamp,
                    score=round(max(0.1, 1.0 - idx * 0.1), 2),
                )
            )
        return results


class DuckDuckGoSearchBackend(SearchBackend):
    """Live public DuckDuckGo Instant Answer API backend (no API key required)."""

    name = "duckduckgo"

    def __init__(
        self,
        *,
        endpoint: str = "https://api.duckduckgo.com/",
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._client = client

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        headers = {"User-Agent": "AI-Research-Agent/0.1"}

        if self._client is not None:
            resp = self._client.get(self.endpoint, params=params, headers=headers)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(self.endpoint, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        abstract = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        heading = data.get("Heading", query).strip() or query
        if abstract and abstract_url:
            results.append(
                SearchResult(
                    title=heading,
                    url=abstract_url,
                    snippet=abstract,
                    raw_html=f"<main><h1>{heading}</h1><p>{abstract}</p></main>",
                    source_backend=self.name,
                    published_date="n/a",
                    score=1.0,
                )
            )

        def _collect_topics(items: list[dict[str, Any]]) -> None:
            for item in items:
                if len(results) >= max_results:
                    return
                if "Topics" in item and isinstance(item["Topics"], list):
                    _collect_topics(item["Topics"])
                    continue
                first_url = item.get("FirstURL", "")
                text = item.get("Text", "")
                if first_url and text:
                    title = text.split(" - ")[0][:90]
                    results.append(
                        SearchResult(
                            title=title,
                            url=first_url,
                            snippet=text,
                            raw_html=f"<article><p>{text}</p></article>",
                            source_backend=self.name,
                            published_date="n/a",
                            score=round(max(0.2, 0.9 - 0.1 * len(results)), 2),
                        )
                    )

        _collect_topics(data.get("RelatedTopics", []))
        return results[:max_results]


_OFFLINE_FIXTURE_CORPUS: list[dict[str, Any]] = [
    {
        "tags": ["rag", "fine-tuning", "enterprise", "llm", "retrieval", "hallucination"],
        "title": "RAG vs Fine-Tuning in Enterprise LLM Deployments (2025 Benchmark)",
        "url": "https://ml-systems.example.org/rag-vs-finetuning-2025?utm_source=newsletter",
        "published_date": "2025-11-12",
        "raw_html": (
            "<html><head><style>.nav{display:none}</style><script>track();</script></head>"
            "<body><nav>Home | Pricing | Login</nav><article>"
            "<h1>RAG vs Fine-Tuning in Enterprise LLM Deployments</h1>"
            "<p>Retrieval-Augmented Generation (RAG) reduces factual hallucination rates by 42% "
            "on rapidly changing corporate knowledge bases without retraining model weights.</p>"
            "<p>RAG systems achieve lower upfront engineering cost ($8,000-$15,000 initial pipeline) "
            "and support instant document revocation for compliance audits.</p>"
            "<p>However, multi-hop retrieval adds 180ms-420ms median latency per query and degrades "
            "when chunking boundaries split structured tables.</p>"
            "</article><footer>Copyright 2025 ML Systems Org</footer></body></html>"
        ),
    },
    {
        # Deliberate UTM duplicate of source 1 to validate URL deduplication
        "tags": ["rag", "fine-tuning", "enterprise", "llm", "retrieval"],
        "title": "RAG vs Fine-Tuning in Enterprise LLM Deployments (Mirror)",
        "url": "https://ml-systems.example.org/rag-vs-finetuning-2025/?utm_medium=social&ref=feed",
        "published_date": "2025-11-12",
        "raw_html": (
            "<article><p>Retrieval-Augmented Generation (RAG) reduces factual hallucination rates by 42% "
            "on rapidly changing corporate knowledge bases without retraining model weights.</p></article>"
        ),
    },
    {
        "tags": ["rag", "fine-tuning", "enterprise", "llm", "lora", "latency"],
        "title": "When Domain Fine-Tuning Beats RAG: Latency and Formatting Study",
        "url": "https://applied-ai-journal.example.com/domain-finetuning-study",
        "published_date": "2026-01-18",
        "raw_html": (
            "<html><body><header>Applied AI Journal</header><main>"
            "<p>LoRA fine-tuning on 12,000 curated domain examples improved JSON schema adherence "
            "from 78% to 97.4% while cutting p95 inference latency by 35% compared to RAG pipelines.</p>"
            "<p>Contrary to claims that RAG always lowers total cost, high-volume workloads exceeding "
            "50 million queries per month showed fine-tuned small models (8B) costing 61% less than "
            "prompt-heavy RAG calls to frontier models.</p>"
            "<p>The main drawback of fine-tuning is knowledge staleness: updating embedded facts "
            "requires scheduled retraining and regression evaluation cycles.</p>"
            "</main></body></html>"
        ),
    },
    {
        "tags": ["rag", "fine-tuning", "hybrid", "enterprise", "llm", "agent", "evaluation"],
        "title": "Hybrid RAG + LoRA Architectures in Production",
        "url": "https://cloud-arch.example.io/hybrid-rag-lora",
        "published_date": "2026-02-04",
        "raw_html": (
            "<html><body><aside>Subscribe now</aside><section>"
            "<p>Combining lightweight LoRA adapters for domain tone and tool syntax with hybrid "
            "BM25 + vector RAG delivered 94% grounded accuracy across financial QA benchmarks.</p>"
            "<p>Teams report that evaluation harnesses and citation verification matter more than "
            "switching base models, yet standardized multi-hop faithfulness benchmarks remain sparse.</p>"
            "<p>Security teams require row-level ACL filtering before vector search to prevent "
            "cross-tenant data leakage in shared indexes.</p>"
            "</section></body></html>"
        ),
    },
    {
        "tags": ["agent", "harness", "coding", "multi-agent", "llm", "tools", "sandbox"],
        "title": "Minimal Coding Agent Harnesses and Bounded Control Loops",
        "url": "https://systems-eng.example.dev/minimal-agent-harnesses",
        "published_date": "2026-03-01",
        "raw_html": (
            "<html><body><article>"
            "<p>Production coding agents succeed with a small orthogonal toolset—list, read, edit, "
            "and test—paired with deterministic workspace sandboxing and strict step budgets.</p>"
            "<p>Adding more than 15 overlapping tools increases tool-selection error rates by 28% "
            "and inflates prompt token overhead without improving bug-fix pass rates.</p>"
            "<p>Structured JSONL execution traces allow offline replay and automated regression evals.</p>"
            "</article></body></html>"
        ),
    },
    {
        "tags": ["multi-agent", "crew", "critic", "verification", "agent", "enterprise", "llm"],
        "title": "Self-Critique and Specialist Multi-Agent Crews: Reliability Analysis",
        "url": "https://ai-ops.example.net/multi-agent-critic-reliability",
        "published_date": "2026-02-22",
        "raw_html": (
            "<html><body><article>"
            "<p>Separating research, quantitative analysis, writing, and independent verification "
            "reduces unsupported claims by 58% compared to single-shot report generation.</p>"
            "<p>However, unbounded multi-agent debate loops increase token cost by 3.4x and can "
            "amplify sycophantic agreement unless capped at 1-2 structured revision rounds.</p>"
            "<p>Strict Pydantic JSON handoff schemas eliminate brittle free-text parsing failures "
            "between specialist agents.</p>"
            "</article></body></html>"
        ),
    },
    {
        "tags": ["local", "inference", "quantization", "gguf", "vllm", "gpu", "cpu", "llm"],
        "title": "Local LLM Serving Trade-Offs: Quantization, Throughput, and Privacy",
        "url": "https://edge-ml.example.org/local-llm-serving-tradeoffs",
        "published_date": "2026-01-29",
        "raw_html": (
            "<html><body><article>"
            "<p>4-bit Q4_K_M quantization cuts VRAM usage by 68% while preserving over 96% of "
            "FP16 benchmark quality on 7B-14B instruction models.</p>"
            "<p>Local deployments eliminate third-party data egress risk, but require capacity "
            "planning for concurrent batching under bursty enterprise traffic.</p>"
            "</article></body></html>"
        ),
    },
]


class FixtureSearchBackend(SearchBackend):
    """Deterministic offline search backend with rich HTML corpus and intentional duplicates."""

    name = "fixture"

    def __init__(self, corpus: list[dict[str, Any]] | None = None) -> None:
        self.corpus = corpus if corpus is not None else _OFFLINE_FIXTURE_CORPUS

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        q_tokens = self._tokenize(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []

        for idx, doc in enumerate(self.corpus):
            haystack = " ".join(doc.get("tags", [])) + " " + doc["title"] + " " + doc["raw_html"]
            d_tokens = self._tokenize(haystack)
            overlap = len(q_tokens & d_tokens)
            tag_bonus = 2.0 * len(q_tokens & set(doc.get("tags", [])))
            score = overlap + tag_bonus
            scored.append((score, -idx, doc))

        scored.sort(reverse=True)
        # Ensure deterministic non-empty output even for niche queries
        top_docs = [item[2] for item in scored[:max_results]]
        results: list[SearchResult] = []
        for rank, doc in enumerate(top_docs):
            text_preview = BeautifulSoup(doc["raw_html"], "html.parser").get_text(" ", strip=True)[:220]
            results.append(
                SearchResult(
                    title=doc["title"],
                    url=doc["url"],
                    snippet=text_preview,
                    raw_html=doc["raw_html"],
                    source_backend=self.name,
                    published_date=doc.get("published_date", "n/a"),
                    score=round(max(0.1, 1.0 - rank * 0.12), 2),
                )
            )
        return results


def create_search_backend(mode: str = "auto") -> SearchBackend:
    """Factory selecting Tavily, Wikipedia, DuckDuckGo, or deterministic Fixture backend."""
    normalized = mode.strip().lower()
    if normalized == "fixture":
        return FixtureSearchBackend()
    if normalized == "tavily":
        return TavilySearchBackend()
    if normalized == "wikipedia":
        return WikipediaSearchBackend()
    if normalized == "duckduckgo":
        return DuckDuckGoSearchBackend()
    if normalized == "auto":
        if os.getenv("TAVILY_API_KEY"):
            return TavilySearchBackend()
        return WikipediaSearchBackend()
    raise ValueError(f"Unsupported search backend mode: {mode}")
