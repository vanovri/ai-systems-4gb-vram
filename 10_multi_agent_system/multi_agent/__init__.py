"""CrewAI-style pure-Python multi-agent engine with structured JSON handoffs and critic revision loop."""

from multi_agent.core import Agent, Crew, CrewResult, Task, Tool
from multi_agent.crewai_adapter import is_crewai_available, to_crewai_crew
from multi_agent.llm import BaseLLM, DeterministicRoleLLM, OpenAICompatibleLLM
from multi_agent.schemas import (
    AnalystHandoff,
    AuditIssue,
    ComparisonItem,
    CriticAuditHandoff,
    ManagerPlanHandoff,
    ReportHandoff,
    ResearcherHandoff,
    RiskItem,
    SourceFact,
)
from multi_agent.trace import ExecutionTrace, TraceStep
from multi_agent.workflow import build_research_crew, run_research_workflow

__all__ = [
    "Agent",
    "AnalystHandoff",
    "AuditIssue",
    "BaseLLM",
    "ComparisonItem",
    "Crew",
    "CrewResult",
    "CriticAuditHandoff",
    "DeterministicRoleLLM",
    "ExecutionTrace",
    "ManagerPlanHandoff",
    "OpenAICompatibleLLM",
    "ReportHandoff",
    "ResearcherHandoff",
    "RiskItem",
    "SourceFact",
    "Task",
    "Tool",
    "TraceStep",
    "build_research_crew",
    "is_crewai_available",
    "run_research_workflow",
    "to_crewai_crew",
]
