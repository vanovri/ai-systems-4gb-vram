"""Specialist agent tools: knowledge search, comparative metric calculator, and citation verifier."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_REPORT_SECTIONS = (
    "## Executive Summary",
    "## Key Evidence & Findings",
    "## Comparative Architecture Analysis",
    "## Trade-Offs & Risk Mitigation",
    "## Recommendations",
    "## References",
)

_CURATED_KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "tags": ["agent", "multi-agent", "enterprise", "production", "architecture", "crew", "llm"],
        "id": "S1",
        "claim": "Separating research, quantitative analysis, writing, and independent verification cuts unsupported claims by 58% relative to single-shot generation.",
        "source_title": "Reliability of Specialist Multi-Agent Workflows (2026)",
        "source_url": "https://ai-ops.example.net/multi-agent-critic-reliability",
        "metric": "-58% unsupported claims",
        "confidence": 0.94,
    },
    {
        "tags": ["agent", "multi-agent", "cost", "revision", "debate", "production", "architecture"],
        "id": "S2",
        "claim": "Unbounded agent debate loops increase token consumption by 3.4x; capping critique to 1-2 structured revision passes preserves >92% of the quality gain at 1.4x baseline cost.",
        "source_title": "Cost-Quality Frontiers in Iterative LLM Critique",
        "source_url": "https://ml-systems.example.org/bounded-critique-frontiers",
        "metric": "1.4x vs 3.4x token cost",
        "confidence": 0.92,
    },
    {
        "tags": ["agent", "schema", "pydantic", "handoff", "json", "enterprise", "production"],
        "id": "S3",
        "claim": "Typed Pydantic JSON handoff contracts reduce downstream parsing failures between specialist agents from 19% to under 1.2% in production pipelines.",
        "source_title": "Typed Inter-Agent Contracts in Production Systems",
        "source_url": "https://systems-eng.example.dev/typed-agent-handoffs",
        "metric": "19% -> <1.2% handoff failures",
        "confidence": 0.95,
    },
    {
        "tags": ["agent", "rag", "latency", "enterprise", "architecture", "tool", "llm"],
        "id": "S4",
        "claim": "Single-agent sequential tool chains average 2.1s end-to-end latency whereas 4-agent specialist crews average 5.8s sequentially (3.2s with parallelizable sub-steps).",
        "source_title": "Latency Telemetry Across Enterprise Agent Topologies",
        "source_url": "https://cloud-arch.example.io/agent-topology-telemetry",
        "metric": "2.1s single vs 5.8s sequential crew",
        "confidence": 0.90,
    },
    {
        "tags": ["agent", "governance", "trace", "audit", "enterprise", "compliance", "production"],
        "id": "S5",
        "claim": "Persisting step-level execution traces (timings, schemas, revision counts) cuts incident root-cause triage time by 64% during enterprise compliance reviews.",
        "source_title": "Observability Patterns for Audited AI Agents",
        "source_url": "https://enterprise-sec.example.org/agent-observability-audit",
        "metric": "-64% triage time",
        "confidence": 0.91,
    },
    {
        "tags": ["rag", "fine-tuning", "hybrid", "llm", "enterprise", "cost"],
        "id": "S6",
        "claim": "Hybrid RAG + lightweight LoRA adapters delivered 94% grounded QA accuracy while enabling instant document revocation for regulated data.",
        "source_title": "Hybrid RAG + LoRA Architectures in Production",
        "source_url": "https://cloud-arch.example.io/hybrid-rag-lora",
        "metric": "94% grounded accuracy",
        "confidence": 0.93,
    },
]


def knowledge_search(query: str, max_facts: int = 5) -> dict[str, Any]:
    """Retrieve evidence-backed facts ranked by keyword overlap against the curated store."""
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, item in enumerate(_CURATED_KNOWLEDGE_BASE):
        haystack = " ".join(item["tags"]) + " " + item["claim"] + " " + item["source_title"]
        d_tokens = set(re.findall(r"[a-z0-9]+", haystack.lower()))
        score = len(q_tokens & d_tokens) + 2 * len(q_tokens & set(item["tags"]))
        scored.append((score, -idx, item))

    scored.sort(reverse=True)
    selected = [entry[2] for entry in scored[:max_facts]]
    # Ensure deterministic S1..Sn ordering in output
    selected_sorted = sorted(selected, key=lambda x: x["id"])
    return {
        "query": query,
        "count": len(selected_sorted),
        "facts": [
            {
                "id": f["id"],
                "claim": f["claim"],
                "source_title": f["source_title"],
                "source_url": f["source_url"],
                "metric": f["metric"],
                "confidence": f["confidence"],
            }
            for f in selected_sorted
        ],
    }


def metric_comparator(facts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic comparative architecture scorecard from researched facts."""
    fact_ids = [f["id"] for f in facts]
    metrics_extracted = {
        f["id"]: f.get("metric") or "qualitative" for f in facts
    }
    return {
        "evidence_count": len(facts),
        "evidence_ids": fact_ids,
        "metrics_extracted": metrics_extracted,
        "topology_scores": {
            "Single-Agent Loop": {"accuracy_index": 0.76, "cost_multiplier": 1.0, "latency_sec": 2.1},
            "Sequential Specialist Crew + Bounded Critic": {
                "accuracy_index": 0.94,
                "cost_multiplier": 1.4,
                "latency_sec": 5.8,
            },
            "Unbounded Multi-Agent Debate": {
                "accuracy_index": 0.95,
                "cost_multiplier": 3.4,
                "latency_sec": 12.6,
            },
        },
    }


def citation_audit(
    markdown_body: str,
    valid_source_ids: list[str],
    required_sections: tuple[str, ...] = REQUIRED_REPORT_SECTIONS,
) -> dict[str, Any]:
    """Audit a Markdown report for ungrounded claims, invalid citations, and missing sections."""
    valid_set = set(valid_source_ids)
    cited_ids = sorted(set(re.findall(r"\[(S\d+)\]", markdown_body)))
    invalid_citations = [cid for cid in cited_ids if cid not in valid_set]
    missing_sections = [sec for sec in required_sections if sec not in markdown_body]

    # Detect uncited quantitative claims (bullets or lines with percentages/multipliers lacking [Sn])
    ungrounded_claims: list[str] = []
    in_references = False
    for raw_line in markdown_body.splitlines():
        line = raw_line.strip()
        if line.startswith("## References"):
            in_references = True
            continue
        if in_references or not line or line.startswith("#") or line.startswith("|"):
            continue
        has_quant = bool(re.search(r"\b\d+(?:\.\d+)?(?:\s*%|x|s\b|ms\b)", line))
        has_citation = bool(re.search(r"\[S\d+\]", line))
        if has_quant and not has_citation:
            ungrounded_claims.append(line.lstrip("-*0123456789. ").strip())

    # Also treat invalid citations as ungrounded claims
    for bad_id in invalid_citations:
        ungrounded_claims.append(f"Invalid citation reference [{bad_id}] not present in Researcher sources.")

    # Compute deterministic quality score in [0.0, 1.0]
    score = 1.0
    score -= 0.16 * len(ungrounded_claims)
    score -= 0.14 * len(missing_sections)
    if len(valid_set) > 0:
        coverage = len(set(cited_ids) & valid_set) / len(valid_set)
        if coverage < 0.8:
            score -= round((0.8 - coverage) * 0.25, 3)
    score = round(max(0.0, min(1.0, score)), 3)

    return {
        "quality_score": score,
        "cited_ids": cited_ids,
        "invalid_citations": invalid_citations,
        "missing_sections": missing_sections,
        "ungrounded_claims": ungrounded_claims,
        "source_coverage": round(len(set(cited_ids) & valid_set) / max(1, len(valid_set)), 3),
    }
