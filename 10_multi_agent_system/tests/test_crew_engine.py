"""Unit tests for the pure-Python CrewAI-style engine, schemas, LLM adapter, and CrewAI bridge."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from multi_agent.core import Agent, Crew, Task, Tool
from multi_agent.crewai_adapter import is_crewai_available, to_crewai_crew
from multi_agent.llm import BaseLLM, OpenAICompatibleLLM
from multi_agent.schemas import CriticAuditHandoff, ResearcherHandoff, SourceFact
from multi_agent.workflow import build_research_crew


class EchoSchema(BaseModel):
    topic: str
    doubled: int


class EchoLLM(BaseLLM):
    def generate_structured(self, *, tool_outputs, **_kwargs):
        return {"topic": "custom", "doubled": int(tool_outputs.get("double", 0))}


def test_custom_agent_task_crew_sequential_kickoff() -> None:
    tool = Tool(name="double", description="Double a fixed metric", func=lambda: 42)
    agent = Agent(
        role="Calculator",
        goal="Compute doubled value",
        backstory="Math specialist",
        tools=[tool],
        llm=EchoLLM(),
    )
    task = Task(
        name="calc_task",
        description="Double value for {topic}",
        expected_output="EchoSchema JSON",
        agent=agent,
        output_schema=EchoSchema,
    )
    crew = Crew(agents=[agent], tasks=[task], process="sequential")
    result = crew.kickoff(inputs={"topic": "math"})

    assert result.final_output == {"topic": "custom", "doubled": 42}
    assert result.trace.total_steps == 1
    assert result.trace.steps[0].tools_called == ["double"]
    assert result.trace.steps[0].handoff_schema == "EchoSchema"


def test_hierarchical_process_executes_manager_delegation_first() -> None:
    crew = build_research_crew(process="hierarchical", simulate_first_draft_gaps=False)
    result = crew.kickoff(inputs={"topic": "Enterprise Multi-Agent Architecture"})

    assert result.process == "hierarchical"
    assert result.trace.steps[0].agent_role == "CrewManager"
    assert result.trace.steps[0].handoff_schema == "ManagerPlanHandoff"
    assert result.trace.total_steps == 5  # Manager + Researcher + Analyst + ReportWriter + CriticVerifier
    assert result.trace.passed_quality_gate is True


def test_pydantic_handoff_schema_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        SourceFact(
            id="S1",
            claim="Claim",
            source_title="Title",
            source_url="https://example.org",
            confidence=1.5,
        )

    valid = ResearcherHandoff(
        topic="Test",
        facts=[
            SourceFact(
                id="S1",
                claim="Valid fact",
                source_title="Bench",
                source_url="https://example.org",
                confidence=0.9,
            )
        ],
        key_themes=["reliability"],
    )
    assert valid.facts[0].id == "S1"


def test_openai_compatible_llm_parses_structured_json_with_mock_transport() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "quality_score": 0.96,
                                    "passed": True,
                                    "threshold": 0.85,
                                    "ungrounded_claims": [],
                                    "missing_sections": [],
                                    "issues": [],
                                    "revision_instructions": "Approved.",
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatibleLLM(
        base_url="http://127.0.0.1:20128/v1",
        api_key="dummy-key",
        model="local-json-model",
        client=client,
    )
    data = llm.generate_structured(
        role="CriticVerifier",
        goal="Audit draft",
        backstory="QA auditor",
        task_description="Audit report",
        expected_output="CriticAuditHandoff JSON",
        context_payloads={},
        tool_outputs={},
        output_schema=CriticAuditHandoff,
    )
    validated = CriticAuditHandoff.model_validate(data)
    assert captured["url"] == "http://127.0.0.1:20128/v1/chat/completions"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert validated.passed is True
    assert validated.quality_score == pytest.approx(0.96)


def test_optional_crewai_adapter_reports_clean_error_when_not_installed() -> None:
    crew = build_research_crew()
    if not is_crewai_available():
        with pytest.raises(RuntimeError, match="crewai"):
            to_crewai_crew(crew)
    else:
        adapted = to_crewai_crew(crew)
        assert adapted is not None
