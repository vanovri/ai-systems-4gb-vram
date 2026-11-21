"""End-to-end ResearchAgent orchestrating Cache -> Search -> Extract/Dedup -> Synthesize."""

from __future__ import annotations

from pathlib import Path

from research_agent.cache import ResearchCache
from research_agent.extractor import extract_claims, extract_documents
from research_agent.models import ResearchBriefResult
from research_agent.search import SearchBackend, create_search_backend
from research_agent.synthesizer import ExtractiveOfflineSynthesizer, Synthesizer


class ResearchAgent:
    """Orchestrates pluggable search, local SQLite caching, dedup extraction, and cited synthesis."""

    def __init__(
        self,
        *,
        backend: SearchBackend | None = None,
        synthesizer: Synthesizer | None = None,
        cache: ResearchCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self.backend = backend or create_search_backend("fixture")
        self.synthesizer = synthesizer or ExtractiveOfflineSynthesizer()
        self.use_cache = use_cache
        self.cache = cache if (cache is not None or not use_cache) else ResearchCache()

    def research(
        self,
        query: str,
        *,
        max_results: int = 6,
        output_path: Path | str | None = None,
    ) -> ResearchBriefResult:
        cache_hit = False
        raw_results = None

        if self.use_cache and self.cache is not None:
            raw_results = self.cache.get(self.backend.name, query, max_results)
            if raw_results is not None:
                cache_hit = True

        if raw_results is None:
            raw_results = self.backend.search(query, max_results=max_results)
            if self.use_cache and self.cache is not None:
                self.cache.set(self.backend.name, query, max_results, raw_results)

        sources = extract_documents(raw_results)
        claims = extract_claims(sources)
        markdown = self.synthesizer.synthesize(
            query,
            sources,
            claims,
            backend_used=self.backend.name,
            cache_hit=cache_hit,
        )

        saved_path: str | None = None
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(markdown, encoding="utf-8")
            saved_path = str(out)

        return ResearchBriefResult(
            query=query,
            backend_used=self.backend.name,
            synthesizer_used=self.synthesizer.name,
            cache_hit=cache_hit,
            raw_result_count=len(raw_results),
            deduped_source_count=len(sources),
            claim_count=len(claims),
            markdown=markdown,
            sources=sources,
            claims=claims,
            output_path=saved_path,
        )
