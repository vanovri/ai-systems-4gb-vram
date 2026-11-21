"""Tests for CriticVerifier auditing, ungrounded-claim detection, and bounded revision loop."""

from __future__ import annotations

from multi_agent.tools import citation_audit
from multi_agent.workflow import build_research_crew


def test_citation_audit_flags_ungrounded_metrics_invalid_refs_and_missing_sections() -> None:
    flawed_md = (
        "# Report\n\n"
        "## Executive Summary\n\n"
        "- Autonomous systems cut cost by 99% overnight.\n"
        "- Valid claim with invalid citation [S99].\n"
        "- Grounded claim [S1].\n\n"
        "## References\n\n"
        "| [S1] | Source | 0.9 | https://example.org |\n"
    )
    audit = citation_audit(flawed_md, valid_source_ids=["S1", "S2"])

    assert audit["quality_score"] < 0.85
    assert "S99" in audit["invalid_citations"]
    assert len(audit["ungrounded_claims"]) >= 2
    assert "## Trade-Offs & Risk Mitigation" in audit["missing_sections"]


def test_critic_triggers_single_revision_round_and_repairs_draft() -> None:
    crew = build_research_crew(
        process="sequential",
        quality_threshold=0.85,
        max_revisions=1,
        simulate_first_draft_gaps=True,
    )
    result = crew.kickoff(inputs={"topic": "Evaluating Multi-Agent AI Architectures"})

    assert result.revision_loop_count == 1
    assert len(result.trace.quality_scores) == 2
    assert result.trace.quality_scores[0] < 0.85
    assert result.trace.quality_scores[1] >= 0.85
    assert result.critic_audit is not None
    assert result.critic_audit.passed is True
    assert result.critic_audit.ungrounded_claims == []
    assert result.critic_audit.missing_sections == []
    # 4 initial steps + 1 writer revision + 1 critic re-verification = 6 steps
    assert result.trace.total_steps == 6
    assert "## Trade-Offs & Risk Mitigation" in result.report_markdown
    assert "Unverified Industry Projection" not in result.report_markdown


def test_no_unnecessary_revision_when_first_draft_passes_quality_gate() -> None:
    crew = build_research_crew(
        process="sequential",
        quality_threshold=0.85,
        max_revisions=2,
        simulate_first_draft_gaps=False,
    )
    result = crew.kickoff(inputs={"topic": "Enterprise Multi-Agent Systems"})

    assert result.revision_loop_count == 0
    assert len(result.trace.quality_scores) == 1
    assert result.trace.quality_scores[0] >= 0.85
    assert result.trace.total_steps == 4
    assert result.trace.passed_quality_gate is True


def test_max_revisions_zero_caps_loop_without_revising() -> None:
    crew = build_research_crew(
        process="sequential",
        quality_threshold=0.85,
        max_revisions=0,
        simulate_first_draft_gaps=True,
    )
    result = crew.kickoff(inputs={"topic": "Capped Loop Test"})

    assert result.revision_loop_count == 0
    assert result.trace.total_steps == 4
    assert result.trace.passed_quality_gate is False
    assert result.critic_audit is not None
    assert result.critic_audit.quality_score < 0.85
