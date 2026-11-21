# Project #7 - AI Research Agent (`07_ai_research_agent`)

End-to-end research pipeline implementing **Search -> Extract & Deduplicate -> Synthesize with Inline Citations** plus local SQLite caching.

## Architecture

- `research_agent/search.py` - Pluggable search backends:
  - `TavilySearchBackend` (live API when `TAVILY_API_KEY` env var is set)
  - `WikipediaSearchBackend` (live public MediaWiki API, zero paid key)
  - `DuckDuckGoSearchBackend` (live public Instant Answer API, zero paid key)
  - `FixtureSearchBackend` (deterministic offline HTML corpus with intentional UTM/content duplicates for reproducible CI & demos)
- `research_agent/extractor.py` - BeautifulSoup HTML-to-clean-text stripping (`script`/`style`/`nav`/`footer`/`aside`), URL canonicalization (tracking-param removal), exact + shingle-Jaccard near-duplicate deduplication, and factual claim + metric extraction.
- `research_agent/cache.py` - Persistent local SQLite cache (`ResearchCache`) with TTL expiry to prevent redundant queries.
- `research_agent/synthesizer.py` - `OpenAICompatibleSynthesizer` (OpenAI / local `9Router` `http://127.0.0.1:20128/v1` / Gemini OpenAI-compat via env vars) + `ExtractiveOfflineSynthesizer` producing structured **Markdown Research Briefs**:
  - `Executive Summary`
  - `Key Findings` with inline `[n]` citations
  - `Contradictions & Knowledge Gaps`
  - `Sources Table`
- `research_agent/pipeline.py` & `research_agent/cli.py` - Orchestrator and CLI.

## Quickstart

```bash
# Run offline deterministic research brief demo
python -m research_agent.cli "RAG vs Fine-Tuning for Enterprise LLMs" --backend fixture --out reports/brief.md

# Run live public Wikipedia search (no API key needed)
python -m research_agent.cli "Retrieval-augmented generation" --backend wikipedia --out reports/wiki_brief.md

# Run live OpenAI-compatible synthesizer (configure OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL via env)
python -m research_agent.cli "Multi-agent verification" --backend fixture --synthesizer openai --out reports/llm_brief.md

# Run unit + integration tests
pytest -v
```
