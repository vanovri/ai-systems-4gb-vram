"""Tests for Markdown Research Brief synthesis, citation integrity, cache reuse, and CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from research_agent.cache import ResearchCache
from research_agent.cli import main as cli_main
from research_agent.extractor import extract_claims, extract_documents
from research_agent.pipeline import ResearchAgent
from research_agent.search import FixtureSearchBackend
from research_agent.synthesizer import OpenAICompatibleSynthesizer


def test_end_to_end_pipeline_generates_cited_brief_and_valid_citations(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "cache.sqlite3")
    agent = ResearchAgent(backend=FixtureSearchBackend(), cache=cache)
    out_md = tmp_path / "brief.md"

    result = agent.research(
        "RAG vs Fine-Tuning for Enterprise LLMs",
        max_results=6,
        output_path=out_md,
    )

    assert result.cache_hit is False
    assert result.deduped_source_count < result.raw_result_count  # UTM mirror was deduped
    assert out_md.exists()

    md = result.markdown
    for heading in (
        "## Executive Summary",
        "## Key Findings",
        "## Contradictions & Knowledge Gaps",
        "## Sources Table",
    ):
        assert heading in md

    # Verify every inline citation [n] references an existing source_id in the Sources Table
    cited_ids = {int(m) for m in re.findall(r"\[(\d+)\]", md)}
    valid_ids = {s.source_id for s in result.sources}
    assert len(cited_ids) >= 2
    assert cited_ids.issubset(valid_ids)


def test_second_identical_query_hits_sqlite_cache_without_re_querying_backend(tmp_path: Path) -> None:
    calls = {"count": 0}

    class CountingFixtureBackend(FixtureSearchBackend):
        def search(self, query: str, max_results: int = 5):
            calls["count"] += 1
            return super().search(query, max_results=max_results)

    cache = ResearchCache(tmp_path / "cache.sqlite3")
    agent = ResearchAgent(backend=CountingFixtureBackend(), cache=cache)

    r1 = agent.research("RAG vs Fine-Tuning", max_results=4)
    r2 = agent.research("  rag vs fine-tuning ", max_results=4)

    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert calls["count"] == 1


def test_openai_compatible_synthesizer_appends_canonical_sources_table() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gemini-compat"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "## Executive Summary\n\nRAG lowers hallucination by 42% [1].\n\n"
                                "## Key Findings\n\n- Grounded retrieval reduces errors [1].\n\n"
                                "## Contradictions & Knowledge Gaps\n\n- Latency overhead varies [1]."
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    synth = OpenAICompatibleSynthesizer(
        base_url="http://127.0.0.1:20128/v1",
        api_key="dummy-key",
        model="gemini-compat",
        client=client,
    )
    docs = extract_documents(FixtureSearchBackend().search("RAG", max_results=2))
    claims = extract_claims(docs)
    md = synth.synthesize("RAG", docs, claims, backend_used="fixture", cache_hit=False)

    assert "## Executive Summary" in md
    assert "## Sources Table" in md
    assert docs[0].url in md


def test_cli_runs_end_to_end_and_outputs_markdown(tmp_path: Path, capsys) -> None:
    out_file = tmp_path / "reports" / "brief.md"
    cache_db = tmp_path / "cli_cache.sqlite3"

    rc1 = cli_main(
        [
            "RAG vs Fine-Tuning for Enterprise LLMs",
            "--backend",
            "fixture",
            "--out",
            str(out_file),
            "--cache-db",
            str(cache_db),
            "--json-meta",
        ]
    )
    assert rc1 == 0
    meta1 = json.loads(capsys.readouterr().out)
    assert meta1["cache_hit"] is False
    assert out_file.exists()

    rc2 = cli_main(
        [
            "RAG vs Fine-Tuning for Enterprise LLMs",
            "--backend",
            "fixture",
            "--out",
            str(out_file),
            "--cache-db",
            str(cache_db),
            "--json-meta",
        ]
    )
    assert rc2 == 0
    meta2 = json.loads(capsys.readouterr().out)
    assert meta2["cache_hit"] is True
