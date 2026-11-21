"""AI Research Agent: Search -> Extract & Dedup -> Synthesize with inline citations."""

from research_agent.cache import ResearchCache
from research_agent.extractor import (
    canonicalize_url,
    deduplicate_documents,
    extract_claims,
    extract_documents,
    html_to_clean_text,
)
from research_agent.models import (
    CleanedDocument,
    ExtractedClaim,
    ResearchBriefResult,
    SearchResult,
)
from research_agent.pipeline import ResearchAgent
from research_agent.search import (
    DuckDuckGoSearchBackend,
    FixtureSearchBackend,
    SearchBackend,
    TavilySearchBackend,
    WikipediaSearchBackend,
    create_search_backend,
)
from research_agent.synthesizer import (
    ExtractiveOfflineSynthesizer,
    OpenAICompatibleSynthesizer,
    Synthesizer,
)

__all__ = [
    "CleanedDocument",
    "DuckDuckGoSearchBackend",
    "ExtractedClaim",
    "ExtractiveOfflineSynthesizer",
    "FixtureSearchBackend",
    "OpenAICompatibleSynthesizer",
    "ResearchAgent",
    "ResearchBriefResult",
    "ResearchCache",
    "SearchBackend",
    "SearchResult",
    "Synthesizer",
    "TavilySearchBackend",
    "WikipediaSearchBackend",
    "canonicalize_url",
    "create_search_backend",
    "deduplicate_documents",
    "extract_claims",
    "extract_documents",
    "html_to_clean_text",
]
