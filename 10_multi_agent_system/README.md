# Project #10 - Multi-Agent AI System (`10_multi_agent_system`)

Framework-light, pure-Python CrewAI-style multi-agent orchestration engine (`Agent`, `Task`, `Crew`, `kickoff`) with strict Pydantic JSON inter-agent handoffs, a 4-specialist research & verification crew, critic-driven revision loop, and full `execution_trace.json` telemetry.

## Architecture

- `multi_agent/core.py` - Pure-Python `Tool`, `Agent`, `Task`, `Crew`, `kickoff()` supporting both `process="sequential"` and `process="hierarchical"` plus bounded quality-gate revision loops.
- `multi_agent/schemas.py` - Typed Pydantic v2 JSON handoff contracts:
  - `ResearcherHandoff` (`SourceFact[]`, themes, gaps)
  - `AnalystHandoff` (`ComparisonItem[]`, trade-offs, `RiskItem[]`, recommendation)
  - `ReportHandoff` (executive Markdown, cited IDs, sections, revision number)
  - `CriticAuditHandoff` (`quality_score`, `passed`, `ungrounded_claims`, `missing_sections`, `AuditIssue[]`, `revision_instructions`)
  - `ManagerPlanHandoff` (hierarchical delegation plan)
- `multi_agent/workflow.py` - 4 specialized agents wired end-to-end:
  1. **Researcher** - gathers verified facts & sources via `knowledge_search`.
  2. **Analyst** - extracts comparative metrics, trade-offs, and risks via `metric_comparator`.
  3. **ReportWriter** - drafts and revises an executive Markdown report with inline `[Sn]` citations.
  4. **CriticVerifier** - audits ungrounded claims & completeness via `citation_audit`; triggers 1 revision round when `quality_score < threshold`.
- `multi_agent/llm.py` - `DeterministicRoleLLM` (100% offline/reproducible CI & demos) + `OpenAICompatibleLLM` (live OpenAI / local `9Router` `http://127.0.0.1:20128/v1` / Gemini OpenAI-compat via env vars).
- `multi_agent/trace.py` - Writes structured `execution_trace.json` (step timings, schemas, quality scores, revision count).
- `multi_agent/crewai_adapter.py` - Optional bridge to upstream `crewai` when installed.

## Quickstart

```bash
# Run offline end-to-end demo (triggers CriticVerifier revision loop 0.70 -> 1.00)
python -m multi_agent.cli "Evaluating Multi-Agent AI Architectures for Enterprise Production" \
  --process sequential \
  --out reports/sample_multi_agent_report.md \
  --trace reports/execution_trace.json

# Run hierarchical mode with CrewManager delegation
python -m multi_agent.cli "Evaluating Multi-Agent AI Architectures for Enterprise Production" \
  --process hierarchical \
  --out reports/hierarchical_report.md \
  --trace reports/hierarchical_trace.json

# Run live against OpenAI / 9Router / Gemini OpenAI-compat endpoint via env vars
# (OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL)
python -m multi_agent.cli "Enterprise RAG vs Fine-Tuning" --provider openai

# Run unit + integration tests
pytest -v
```
