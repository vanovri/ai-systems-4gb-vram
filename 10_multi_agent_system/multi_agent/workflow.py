"""Pre-wired 4-agent collaborative research & verification crew (Researcher, Analyst, ReportWriter, CriticVerifier)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from multi_agent.core import Agent, Crew, CrewResult, Task, Tool
from multi_agent.llm import BaseLLM, DeterministicRoleLLM
from multi_agent.schemas import (
    AnalystHandoff,
    CriticAuditHandoff,
    ReportHandoff,
    ResearcherHandoff,
)
from multi_agent.tools import citation_audit, knowledge_search, metric_comparator


def build_research_crew(
    *,
    llm: BaseLLM | None = None,
    process: Literal["sequential", "hierarchical"] = "sequential",
    quality_threshold: float = 0.85,
    max_revisions: int = 1,
    simulate_first_draft_gaps: bool = True,
) -> Crew:
    """Construct the 4-agent specialist crew with typed JSON handoffs and critic revision loop."""
    shared_llm = llm or DeterministicRoleLLM(
        simulate_first_draft_gaps=simulate_first_draft_gaps,
        quality_threshold=quality_threshold,
    )

    researcher = Agent(
        role="Researcher",
        goal="Gather verified, quantitative facts and primary sources on the target topic.",
        backstory="Principal technical researcher focused on empirical benchmarks and primary documentation.",
        tools=[
            Tool(
                name="knowledge_search",
                description="Search curated technical benchmarks and source records.",
                func=knowledge_search,
            )
        ],
        llm=shared_llm,
    )

    analyst = Agent(
        role="Analyst",
        goal="Synthesize quantitative metrics, trade-offs, architectural comparisons, and operational risks.",
        backstory="Staff ML systems architect translating raw evidence into structured decision matrices.",
        tools=[
            Tool(
                name="metric_comparator",
                description="Compute comparative cost, latency, and accuracy metrics across options.",
                func=metric_comparator,
            )
        ],
        llm=shared_llm,
    )

    writer = Agent(
        role="ReportWriter",
        goal="Produce a crisp executive Markdown report with complete sections and inline [Sn] citations.",
        backstory="Executive technical editor who grounds every claim in verified upstream handoffs.",
        tools=[],
        llm=shared_llm,
    )

    critic = Agent(
        role="CriticVerifier",
        goal="Audit drafts for ungrounded claims, invalid citations, and missing sections; trigger revision if quality < threshold.",
        backstory="Independent QA & compliance auditor enforcing strict evidence traceability.",
        tools=[
            Tool(
                name="citation_audit",
                description="Verify every [Sn] citation, detect ungrounded quantitative claims, and score section completeness.",
                func=citation_audit,
            )
        ],
        llm=shared_llm,
    )

    research_task = Task(
        name="research_gathering",
        description="Gather verified facts, quantitative benchmarks, and source URLs for: {topic}",
        expected_output="ResearcherHandoff JSON containing SourceFact items, key themes, and data gaps.",
        agent=researcher,
        output_schema=ResearcherHandoff,
    )

    analysis_task = Task(
        name="comparative_analysis",
        description="Analyze researched facts for '{topic}' to extract metrics, architecture comparisons, trade-offs, and risks.",
        expected_output="AnalystHandoff JSON with ComparisonItem table, RiskItem mitigations, and recommendation.",
        agent=analyst,
        context=[research_task],
        output_schema=AnalystHandoff,
    )

    writing_task = Task(
        name="executive_report_drafting",
        description="Write an executive Markdown report on '{topic}' grounded strictly in Researcher and Analyst handoffs.",
        expected_output="ReportHandoff JSON containing complete Markdown body and cited source IDs.",
        agent=writer,
        context=[research_task, analysis_task],
        output_schema=ReportHandoff,
    )

    critic_task = Task(
        name="critic_verification_audit",
        description="Audit the drafted report on '{topic}' for ungrounded claims, citation validity, and section completeness.",
        expected_output="CriticAuditHandoff JSON with quality_score, issues, and revision instructions if needed.",
        agent=critic,
        context=[research_task, analysis_task, writing_task],
        output_schema=CriticAuditHandoff,
        revision_target=writing_task,
    )

    return Crew(
        agents=[researcher, analyst, writer, critic],
        tasks=[research_task, analysis_task, writing_task, critic_task],
        process=process,
        quality_threshold=quality_threshold,
        max_revisions=max_revisions,
    )


def run_research_workflow(
    topic: str,
    *,
    llm: BaseLLM | None = None,
    process: Literal["sequential", "hierarchical"] = "sequential",
    quality_threshold: float = 0.85,
    max_revisions: int = 1,
    simulate_first_draft_gaps: bool = True,
    report_path: Path | str | None = None,
    trace_path: Path | str | None = None,
) -> CrewResult:
    """Build and run the 4-agent workflow end-to-end, optionally saving report Markdown and trace JSON."""
    crew = build_research_crew(
        llm=llm,
        process=process,
        quality_threshold=quality_threshold,
        max_revisions=max_revisions,
        simulate_first_draft_gaps=simulate_first_draft_gaps,
    )
    result = crew.kickoff(inputs={"topic": topic}, trace_path=trace_path)

    if report_path is not None and result.report_markdown:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.report_markdown, encoding="utf-8")

    return result
