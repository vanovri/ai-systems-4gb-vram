"""Tests for pluggable search backends (Fixture, Tavily, Wikipedia, DuckDuckGo) with mocked HTTP."""

from __future__ import annotations

import json

import httpx
import pytest

from research_agent.search import (
    DuckDuckGoSearchBackend,
    FixtureSearchBackend,
    TavilySearchBackend,
    WikipediaSearchBackend,
    create_search_backend,
)


def test_fixture_backend_deterministic_and_returns_html() -> None:
    backend = FixtureSearchBackend()
    results = backend.search("RAG vs Fine-Tuning enterprise LLM", max_results=5)
    assert len(results) == 5
    assert all(r.source_backend == "fixture" for r in results)
    assert any("RAG" in r.title for r in results)


def test_tavily_backend_requires_key_and_parses_mock_response() -> None:
    no_key = TavilySearchBackend(api_key="")
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        no_key.search("test")

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Tavily Hit",
                        "url": "https://docs.example/tavily",
                        "content": "Concise summary snippet.",
                        "raw_content": "<article><p>Detailed content from Tavily.</p></article>",
                        "published_date": "2026-02-10",
                        "score": 0.94,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = TavilySearchBackend(api_key="dummy-test-key", client=client)
    hits = backend.search("agent harnesses", max_results=3)

    assert captured["body"]["query"] == "agent harnesses"
    assert len(hits) == 1
    assert hits[0].title == "Tavily Hit"
    assert hits[0].source_backend == "tavily"
    assert hits[0].score == pytest.approx(0.94)


def test_wikipedia_backend_parses_mediawiki_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["list"] == "search"
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "title": "Retrieval-augmented generation",
                            "snippet": "A <span class='searchmatch'>RAG</span> technique grounding LLMs.",
                            "timestamp": "2026-01-15T12:00:00Z",
                        }
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = WikipediaSearchBackend(client=client)
    hits = backend.search("RAG", max_results=2)

    assert len(hits) == 1
    assert hits[0].url == "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    assert hits[0].published_date == "2026-01-15"
    assert "RAG technique" in hits[0].snippet


def test_duckduckgo_backend_parses_instant_answer_and_related_topics() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "Heading": "Python",
                "AbstractText": "Python is a high-level programming language.",
                "AbstractURL": "https://en.wikipedia.org/wiki/Python",
                "RelatedTopics": [
                    {
                        "FirstURL": "https://duckduckgo.com/CPython",
                        "Text": "CPython - Reference implementation of Python.",
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = DuckDuckGoSearchBackend(client=client)
    hits = backend.search("Python", max_results=3)

    assert len(hits) == 2
    assert hits[0].title == "Python"
    assert hits[1].url == "https://duckduckgo.com/CPython"


def test_backend_factory_respects_auto_and_explicit_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert isinstance(create_search_backend("auto"), WikipediaSearchBackend)
    assert isinstance(create_search_backend("fixture"), FixtureSearchBackend)

    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-token")
    assert isinstance(create_search_backend("auto"), TavilySearchBackend)
