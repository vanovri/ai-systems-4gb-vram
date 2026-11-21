"""Research brief synthesizers: Extractive offline fallback & OpenAI-compatible live LLM."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

from research_agent.models import CleanedDocument, ExtractedClaim


class Synthesizer(ABC):
    """Generates a cited Markdown Research Brief from cleaned sources and extracted claims."""

    name: str = "base"

    @abstractmethod
    def synthesize(
        self,
        query: str,
        sources: list[CleanedDocument],
        claims: list[ExtractedClaim],
        *,
        backend_used: str,
        cache_hit: bool,
    ) -> str:
        """Return a structured Markdown Research Brief with inline [n] citations."""


def render_sources_table(sources: list[CleanedDocument]) -> str:
    lines = [
        "| # | Title | Domain | Backend | Published | URL |",
        "| - | ----- | ------ | ------- | --------- | --- |",
    ]
    for src in sources:
        safe_title = src.title.replace("|", "\\|")
        lines.append(
            f"| [{src.source_id}] | {safe_title} | `{src.domain}` | {src.source_backend} | "
            f"{src.published_date} | {src.url} |"
        )
    return "\n".join(lines)


class ExtractiveOfflineSynthesizer(Synthesizer):
    """Deterministic zero-network synthesizer producing a cited Markdown Research Brief."""

    name = "extractive-offline"

    @staticmethod
    def _find_contradictions(claims: list[ExtractedClaim]) -> list[str]:
        contradictions: list[str] = []
        pos_or_metric = [c for c in claims if c.stance in {"positive", "metric"}]
        negatives = [c for c in claims if c.stance == "negative"]

        used_pairs: set[tuple[int, int]] = set()
        for neg in negatives:
            neg_kw = set(neg.keywords)
            for pos in pos_or_metric:
                if neg.source_id == pos.source_id:
                    continue
                pair = (min(neg.source_id, pos.source_id), max(neg.source_id, pos.source_id))
                if pair in used_pairs:
                    continue
                shared = sorted(neg_kw & set(pos.keywords))
                if shared:
                    used_pairs.add(pair)
                    topic_label = ", ".join(shared[:3])
                    contradictions.append(
                        f"- **Trade-off / Tension ({topic_label})**: Source [{pos.source_id}] reports "
                        f'"{pos.claim_text}" whereas Source [{neg.source_id}] highlights a caveat or counter-evidence: '
                        f'"{neg.claim_text}"'
                    )
                    break
        return contradictions

    def synthesize(
        self,
        query: str,
        sources: list[CleanedDocument],
        claims: list[ExtractedClaim],
        *,
        backend_used: str,
        cache_hit: bool,
    ) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if not sources:
            return (
                f"# Research Brief: {query}\n\n"
                f"- **Date (UTC):** {date_str}\n"
                f"- **Search Backend:** `{backend_used}` (cache_hit={cache_hit})\n\n"
                "## Executive Summary\n\nNo verifiable sources were returned for this query.\n\n"
                "## Key Findings\n\n- No claims extracted.\n\n"
                "## Contradictions & Knowledge Gaps\n\n- Primary evidence is missing; broaden search terms.\n\n"
                "## Sources Table\n\n_No sources._\n"
            )

        # Rank claims preferring quantitative metrics first, then positive/negative, preserving diversity
        priority = {"metric": 0, "positive": 1, "negative": 2, "neutral": 3}
        ranked = sorted(claims, key=lambda c: (priority[c.stance], c.source_id))

        # Executive Summary: concise 2-3 sentence synthesis with inline citations
        summary_claims: list[ExtractedClaim] = []
        seen_sources: set[int] = set()
        for claim in ranked:
            if claim.source_id not in seen_sources:
                summary_claims.append(claim)
                seen_sources.add(claim.source_id)
            if len(summary_claims) >= 3:
                break
        if not summary_claims and ranked:
            summary_claims = ranked[:2]

        exec_sentences = [
            f"{c.claim_text.rstrip('.')} [{c.source_id}]." for c in summary_claims
        ]
        exec_summary = (
            f"This research brief synthesizes **{len(sources)} deduplicated sources** "
            f"and **{len(claims)} extracted claims** on *{query}*. "
            + " ".join(exec_sentences)
        )

        # Key Findings: up to 8 cited findings covering all sources
        key_findings_lines: list[str] = []
        for idx, claim in enumerate(ranked[:8], start=1):
            metric_badge = f" *(metrics: {', '.join(claim.metrics[:2])})*" if claim.metrics else ""
            key_findings_lines.append(
                f"{idx}. {claim.claim_text.rstrip('.')} [{claim.source_id}].{metric_badge}"
            )

        # Contradictions & Knowledge Gaps
        contradictions = self._find_contradictions(claims)
        gap_lines: list[str] = list(contradictions)
        if not gap_lines:
            gap_lines.append(
                "- No direct factual contradictions were detected across the retrieved subset; "
                "however, single-pass search may under-represent dissenting benchmarks."
            )
        missing_dates = [s.source_id for s in sources if s.published_date == "n/a"]
        if missing_dates:
            refs = ", ".join(f"[{sid}]" for sid in missing_dates)
            gap_lines.append(
                f"- **Temporal Freshness Gap**: Publication dates were unavailable for source(s) {refs}; "
                "verify recency before making production architecture commitments."
            )
        if len(sources) < 4:
            gap_lines.append(
                f"- **Coverage Gap**: Only {len(sources)} unique sources passed deduplication; "
                "consider running an expanded query or live web search backend for broader triangulation."
            )

        sections = [
            f"# Research Brief: {query}",
            "",
            f"- **Generated (UTC):** {date_str}",
            f"- **Search Backend:** `{backend_used}`",
            f"- **Synthesizer:** `{self.name}`",
            f"- **Cache Hit:** `{cache_hit}`",
            f"- **Deduplicated Sources:** {len(sources)}",
            "",
            "## Executive Summary",
            "",
            exec_summary,
            "",
            "## Key Findings",
            "",
            "\n".join(key_findings_lines),
            "",
            "## Contradictions & Knowledge Gaps",
            "",
            "\n".join(gap_lines),
            "",
            "## Sources Table",
            "",
            render_sources_table(sources),
            "",
        ]
        return "\n".join(sections)


class OpenAICompatibleSynthesizer(Synthesizer):
    """Live LLM synthesizer supporting OpenAI, local 9Router (127.0.0.1:20128/v1), or Gemini compat."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout_seconds = timeout_seconds
        self._client = client

    def synthesize(
        self,
        query: str,
        sources: list[CleanedDocument],
        claims: list[ExtractedClaim],
        *,
        backend_used: str,
        cache_hit: bool,
    ) -> str:
        evidence_lines = [
            f"[{c.source_id}] ({c.domain}) {c.claim_text}" for c in claims[:16]
        ]
        prompt = (
            f"Write a concise Markdown Research Brief for the topic: '{query}'.\n"
            "Requirements:\n"
            "1. Include exact headings: '## Executive Summary', '## Key Findings', "
            "'## Contradictions & Knowledge Gaps', and '## Sources Table'.\n"
            "2. Every factual statement MUST include inline citation brackets like [1], [2] matching the evidence IDs.\n"
            "3. Never invent facts or citations outside the provided evidence.\n\n"
            "Evidence:\n" + "\n".join(evidence_lines)
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a rigorous technical research analyst producing strictly cited Markdown briefs.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        url = f"{self.base_url}/chat/completions"
        if self._client is not None:
            resp = self._client.post(url, headers=headers, json=payload)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        markdown = data["choices"][0]["message"]["content"].strip()

        # Guarantee canonical Sources Table is always present and accurate
        if "## Sources Table" not in markdown:
            markdown += "\n\n## Sources Table\n\n" + render_sources_table(sources) + "\n"
        return markdown
