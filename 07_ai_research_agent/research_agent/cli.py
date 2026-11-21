"""CLI entrypoint for AI Research Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.cache import ResearchCache
from research_agent.pipeline import ResearchAgent
from research_agent.search import create_search_backend
from research_agent.synthesizer import (
    ExtractiveOfflineSynthesizer,
    OpenAICompatibleSynthesizer,
    Synthesizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research_agent.cli",
        description="Search -> Extract & Deduplicate -> Cited Markdown Research Brief CLI.",
    )
    parser.add_argument("query", help="Research question or topic.")
    parser.add_argument(
        "--out",
        default="reports/brief.md",
        help="Output Markdown file path (default: reports/brief.md).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "fixture", "wikipedia", "duckduckgo", "tavily"],
        default="fixture",
        help="Search backend (default: fixture for deterministic offline execution).",
    )
    parser.add_argument(
        "--synthesizer",
        choices=["offline", "openai"],
        default="offline",
        help="Brief synthesizer mode (default: offline extractive).",
    )
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for openai synthesizer.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL for openai synthesizer.")
    parser.add_argument("--max-results", type=int, default=6, help="Max search results to fetch.")
    parser.add_argument(
        "--cache-db",
        default=".cache/research_cache.sqlite3",
        help="SQLite cache database path.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Bypass local SQLite cache.")
    parser.add_argument("--json-meta", action="store_true", help="Print run metadata as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    backend = create_search_backend(args.backend)
    synthesizer: Synthesizer
    if args.synthesizer == "openai":
        synthesizer = OpenAICompatibleSynthesizer(model=args.model, base_url=args.base_url)
    else:
        synthesizer = ExtractiveOfflineSynthesizer()

    cache = None if args.no_cache else ResearchCache(Path(args.cache_db))
    agent = ResearchAgent(
        backend=backend,
        synthesizer=synthesizer,
        cache=cache,
        use_cache=not args.no_cache,
    )
    result = agent.research(args.query, max_results=args.max_results, output_path=args.out)

    meta = {
        "query": result.query,
        "backend_used": result.backend_used,
        "synthesizer_used": result.synthesizer_used,
        "cache_hit": result.cache_hit,
        "raw_result_count": result.raw_result_count,
        "deduped_source_count": result.deduped_source_count,
        "claim_count": result.claim_count,
        "output_path": result.output_path,
    }
    if args.json_meta:
        print(json.dumps(meta, indent=2))
    else:
        print(
            f"Research Brief saved -> {result.output_path} "
            f"(backend={result.backend_used}, cache_hit={result.cache_hit}, "
            f"sources={result.deduped_source_count}/{result.raw_result_count}, claims={result.claim_count})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
