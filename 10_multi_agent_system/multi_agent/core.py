"""CrewAI-style pure-Python multi-agent orchestration engine (Agent, Task, Crew, kickoff)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

from multi_agent.llm import BaseLLM, DeterministicRoleLLM
from multi_agent.schemas import CriticAuditHandoff, ManagerPlanHandoff, ReportHandoff
from multi_agent.trace import ExecutionTrace, TraceStep


@dataclass(slots=True)
class Tool:
    """Callable tool attached to an Agent."""

    name: str
    description: str
    func: Callable[..., Any]

    def run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


@dataclass
class Agent:
    """CrewAI-style specialist Agent definition."""

    role: str
    goal: str
    backstory: str
    tools: list[Tool] = field(default_factory=list)
    llm: BaseLLM = field(default_factory=DeterministicRoleLLM)
    verbose: bool = False

    def execute_task(
        self,
        task: Task,
        *,
        inputs: dict[str, Any],
        context_payloads: dict[str, dict[str, Any]],
        revision_round: int = 0,
    ) -> tuple[dict[str, Any], list[str]]:
        rendered_desc = task.render_description(inputs)
        tool_outputs: dict[str, Any] = {}
        tools_called: list[str] = []

        for tool in self.tools:
            kwargs = self._prepare_tool_kwargs(tool.name, inputs, context_payloads)
            if kwargs is not None:
                tool_outputs[tool.name] = tool.run(**kwargs)
                tools_called.append(tool.name)

        raw_output = self.llm.generate_structured(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            task_description=rendered_desc,
            expected_output=task.expected_output,
            context_payloads=context_payloads,
            tool_outputs=tool_outputs,
            output_schema=task.output_schema,
            revision_round=revision_round,
        )

        if task.output_schema is not None:
            validated = task.output_schema.model_validate(raw_output)
            return validated.model_dump(), tools_called

        return raw_output, tools_called

    @staticmethod
    def _prepare_tool_kwargs(
        tool_name: str,
        inputs: dict[str, Any],
        context_payloads: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if tool_name == "knowledge_search":
            return {"query": str(inputs.get("topic", "")), "max_facts": int(inputs.get("max_facts", 5))}

        if tool_name == "metric_comparator":
            for payload in context_payloads.values():
                if isinstance(payload, dict) and "facts" in payload:
                    return {"facts": payload["facts"]}
            return {"facts": []}

        if tool_name == "citation_audit":
            markdown_body = ""
            valid_ids: list[str] = []
            for payload in context_payloads.values():
                if isinstance(payload, dict):
                    if "markdown_body" in payload:
                        markdown_body = str(payload["markdown_body"])
                    if "facts" in payload:
                        valid_ids = [f["id"] for f in payload["facts"]]
            return {"markdown_body": markdown_body, "valid_source_ids": valid_ids}

        return {}


@dataclass
class Task:
    """CrewAI-style Task bound to a specialist Agent with optional context dependencies and schema."""

    description: str
    expected_output: str
    agent: Agent
    context: list[Task] = field(default_factory=list)
    output_schema: type[BaseModel] | None = None
    name: str | None = None
    revision_target: Task | None = None

    @property
    def task_name(self) -> str:
        return self.name or f"{self.agent.role.lower().replace(' ', '_')}_task"

    def render_description(self, inputs: dict[str, Any]) -> str:
        try:
            return self.description.format(**inputs)
        except KeyError:
            return self.description


@dataclass
class CrewResult:
    """Final structured result of Crew.kickoff()."""

    run_id: str
    topic: str
    process: str
    final_output: dict[str, Any]
    task_outputs: dict[str, dict[str, Any]]
    report_markdown: str
    critic_audit: CriticAuditHandoff | None
    revision_loop_count: int
    trace: ExecutionTrace


class Crew:
    """CrewAI-style multi-agent orchestrator supporting sequential & hierarchical processes + revision loop."""

    def __init__(
        self,
        *,
        agents: list[Agent],
        tasks: list[Task],
        process: Literal["sequential", "hierarchical"] = "sequential",
        manager_agent: Agent | None = None,
        quality_threshold: float = 0.85,
        max_revisions: int = 1,
    ) -> None:
        if process not in {"sequential", "hierarchical"}:
            raise ValueError(f"Unsupported process mode: {process}")
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.manager_agent = manager_agent
        self.quality_threshold = quality_threshold
        self.max_revisions = max_revisions

    def _default_manager(self) -> Agent:
        base_llm = self.agents[0].llm if self.agents else DeterministicRoleLLM(
            quality_threshold=self.quality_threshold
        )
        return Agent(
            role="CrewManager",
            goal="Coordinate specialist delegation and enforce output quality gates.",
            backstory="Principal AI program manager overseeing multi-agent execution and audit standards.",
            tools=[],
            llm=base_llm,
        )

    @staticmethod
    def _summarize_handoff(schema_name: str, payload: dict[str, Any]) -> str:
        if schema_name == "ResearcherHandoff":
            return f"Collected {len(payload.get('facts', []))} verified facts."
        if schema_name == "AnalystHandoff":
            return (
                f"Compared {len(payload.get('comparisons', []))} options and "
                f"identified {len(payload.get('risks', []))} risks."
            )
        if schema_name == "ReportHandoff":
            return (
                f"Drafted '{payload.get('title', 'Report')}' "
                f"(rev={payload.get('revision_number', 0)}, sections={len(payload.get('sections_present', []))})."
            )
        if schema_name == "CriticAuditHandoff":
            return (
                f"score={payload.get('quality_score')} passed={payload.get('passed')} "
                f"ungrounded={len(payload.get('ungrounded_claims', []))} "
                f"missing_sections={len(payload.get('missing_sections', []))}"
            )
        if schema_name == "ManagerPlanHandoff":
            return f"Delegated sequence: {', '.join(payload.get('delegation_order', []))}."
        return f"Completed {schema_name}"

    def _run_single_task(
        self,
        task: Task,
        *,
        inputs: dict[str, Any],
        task_outputs: dict[str, dict[str, Any]],
        extra_context: dict[str, dict[str, Any]] | None,
        trace: ExecutionTrace,
        revision_round: int,
    ) -> dict[str, Any]:
        context_payloads: dict[str, dict[str, Any]] = {}
        for dep in task.context:
            if dep.task_name in task_outputs:
                context_payloads[dep.task_name] = task_outputs[dep.task_name]
        if extra_context:
            context_payloads.update(extra_context)

        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        output, tools_called = task.agent.execute_task(
            task,
            inputs=inputs,
            context_payloads=context_payloads,
            revision_round=revision_round,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        schema_name = task.output_schema.__name__ if task.output_schema else "dict"
        step = TraceStep(
            step_index=len(trace.steps) + 1,
            agent_role=task.agent.role,
            task_name=task.task_name,
            revision_round=revision_round,
            started_at=started_at,
            duration_ms=duration_ms,
            tools_called=tools_called,
            handoff_schema=schema_name,
            handoff_valid=True,
            handoff_summary=self._summarize_handoff(schema_name, output),
            handoff_payload=output,
        )
        trace.steps.append(step)
        task_outputs[task.task_name] = output
        return output

    def kickoff(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        trace_path: Path | str | None = None,
        run_id: str | None = None,
    ) -> CrewResult:
        inputs = dict(inputs or {})
        topic = str(inputs.get("topic", "Untitled Research Topic"))
        run_id = run_id or f"crew-{uuid.uuid4().hex[:10]}"

        run_t0 = time.perf_counter()
        trace = ExecutionTrace(
            run_id=run_id,
            topic=topic,
            process=self.process,
            quality_threshold=self.quality_threshold,
        )
        task_outputs: dict[str, dict[str, Any]] = {}
        extra_shared: dict[str, dict[str, Any]] = {}

        # In hierarchical mode, manager plans delegation before specialists run
        if self.process == "hierarchical":
            manager = self.manager_agent or self._default_manager()
            plan_task = Task(
                name="manager_delegation_plan",
                description="Create delegation plan and focus directives for '{topic}'.",
                expected_output="ManagerPlanHandoff JSON",
                agent=manager,
                output_schema=ManagerPlanHandoff,
            )
            plan_output = self._run_single_task(
                plan_task,
                inputs=inputs,
                task_outputs=task_outputs,
                extra_context=None,
                trace=trace,
                revision_round=0,
            )
            extra_shared["manager_delegation_plan"] = plan_output

        revision_loop_count = 0
        latest_critic: CriticAuditHandoff | None = None

        for idx, task in enumerate(self.tasks):
            output = self._run_single_task(
                task,
                inputs=inputs,
                task_outputs=task_outputs,
                extra_context=extra_shared or None,
                trace=trace,
                revision_round=0,
            )

            # Check if this task is a CriticVerifier quality gate
            if task.output_schema is CriticAuditHandoff:
                audit = CriticAuditHandoff.model_validate(output)
                latest_critic = audit
                trace.quality_scores.append(audit.quality_score)

                target_task = task.revision_target or (self.tasks[idx - 1] if idx > 0 else None)
                while (
                    not audit.passed
                    and audit.quality_score < self.quality_threshold
                    and revision_loop_count < self.max_revisions
                    and target_task is not None
                ):
                    revision_loop_count += 1
                    # Trigger targeted revision on ReportWriter with critic feedback attached
                    revision_context = dict(extra_shared)
                    revision_context["critic_feedback"] = audit.model_dump()
                    self._run_single_task(
                        target_task,
                        inputs=inputs,
                        task_outputs=task_outputs,
                        extra_context=revision_context,
                        trace=trace,
                        revision_round=revision_loop_count,
                    )

                    # Re-verify revised output with CriticVerifier
                    re_audit_output = self._run_single_task(
                        task,
                        inputs=inputs,
                        task_outputs=task_outputs,
                        extra_context=extra_shared or None,
                        trace=trace,
                        revision_round=revision_loop_count,
                    )
                    audit = CriticAuditHandoff.model_validate(re_audit_output)
                    latest_critic = audit
                    trace.quality_scores.append(audit.quality_score)

        # Resolve final Markdown report from ReportHandoff output if present
        report_markdown = ""
        for payload in task_outputs.values():
            if isinstance(payload, dict) and "markdown_body" in payload:
                report_markdown = str(payload["markdown_body"])

        trace.finished_at = datetime.now(timezone.utc).isoformat()
        trace.total_duration_ms = round((time.perf_counter() - run_t0) * 1000.0, 3)
        trace.total_steps = len(trace.steps)
        trace.revision_loop_count = revision_loop_count
        trace.final_quality_score = latest_critic.quality_score if latest_critic else None
        trace.passed_quality_gate = bool(latest_critic.passed) if latest_critic else True

        if trace_path is not None:
            trace.save(trace_path)

        final_task_name = self.tasks[-1].task_name if self.tasks else ""
        final_output = task_outputs.get(final_task_name, {})

        return CrewResult(
            run_id=run_id,
            topic=topic,
            process=self.process,
            final_output=final_output,
            task_outputs=task_outputs,
            report_markdown=report_markdown,
            critic_audit=latest_critic,
            revision_loop_count=revision_loop_count,
            trace=trace,
        )
