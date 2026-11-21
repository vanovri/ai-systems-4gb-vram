"""Strict Pydantic JSON handoff schemas exchanged between specialist agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceFact(BaseModel):
    """Single evidence-backed fact collected by the Researcher."""

    id: str = Field(description="Citation handle such as S1, S2.")
    claim: str
    source_title: str
    source_url: str
    metric: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)


class ResearcherHandoff(BaseModel):
    """Structured output of the Researcher agent."""

    topic: str
    facts: list[SourceFact]
    key_themes: list[str]
    data_gaps: list[str] = Field(default_factory=list)


class ComparisonItem(BaseModel):
    """Structured option row in the Analyst comparison matrix."""

    option: str
    pros: list[str]
    cons: list[str]
    estimated_cost: str
    latency_profile: str
    best_for: str


class RiskItem(BaseModel):
    """Risk and mitigation item grounded in Researcher evidence IDs."""

    risk: str
    severity: Literal["low", "medium", "high"]
    mitigation: str
    evidence_ids: list[str]


class AnalystHandoff(BaseModel):
    """Structured output of the Analyst agent."""

    topic: str
    metrics_summary: dict[str, str]
    comparisons: list[ComparisonItem]
    tradeoffs: list[str]
    risks: list[RiskItem]
    recommendation: str


class ReportHandoff(BaseModel):
    """Structured output of the ReportWriter agent."""

    title: str
    markdown_body: str
    cited_source_ids: list[str]
    sections_present: list[str]
    revision_number: int = 0


class AuditIssue(BaseModel):
    """Specific defect identified by the CriticVerifier."""

    severity: Literal["low", "medium", "high"]
    issue_type: Literal["ungrounded_claim", "missing_section", "missing_citation", "clarity"]
    description: str
    suggested_fix: str


class CriticAuditHandoff(BaseModel):
    """Structured verification audit emitted by CriticVerifier."""

    quality_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    threshold: float = 0.85
    ungrounded_claims: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    revision_instructions: str = ""


class ManagerPlanHandoff(BaseModel):
    """Delegation plan emitted by Manager agent in hierarchical process mode."""

    topic: str
    delegation_order: list[str]
    focus_directives: dict[str, str]
    quality_gate_threshold: float = 0.85
