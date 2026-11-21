"""LLM backends for multi-agent execution: Deterministic offline role engine & OpenAI-compatible live client."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

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
from multi_agent.tools import REQUIRED_REPORT_SECTIONS


class BaseLLM(ABC):
    """Minimal structured generation interface consumed by Agent."""

    @abstractmethod
    def generate_structured(
        self,
        *,
        role: str,
        goal: str,
        backstory: str,
        task_description: str,
        expected_output: str,
        context_payloads: dict[str, dict[str, Any]],
        tool_outputs: dict[str, Any],
        output_schema: type[BaseModel] | None,
        revision_round: int = 0,
    ) -> dict[str, Any]:
        """Return a JSON-serializable dict conforming to output_schema (if provided)."""


class OpenAICompatibleLLM(BaseLLM):
    """Live OpenAI-compatible structured JSON LLM (OpenAI / 9Router 127.0.0.1:20128/v1 / Gemini compat)."""

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

    def generate_structured(
        self,
        *,
        role: str,
        goal: str,
        backstory: str,
        task_description: str,
        expected_output: str,
        context_payloads: dict[str, dict[str, Any]],
        tool_outputs: dict[str, Any],
        output_schema: type[BaseModel] | None,
        revision_round: int = 0,
    ) -> dict[str, Any]:
        schema_hint = (
            json.dumps(output_schema.model_json_schema(), indent=2)
            if output_schema is not None
            else '{"result": "string"}'
        )
        system_msg = (
            f"You are acting as {role}. Goal: {goal}. Backstory: {backstory}. "
            "Respond ONLY with a valid JSON object matching the required JSON Schema."
        )
        user_msg = (
            f"Task (revision_round={revision_round}): {task_description}\n"
            f"Expected Output: {expected_output}\n\n"
            f"Upstream Agent Handoffs:\n{json.dumps(context_payloads, indent=2)}\n\n"
            f"Tool Observations:\n{json.dumps(tool_outputs, indent=2)}\n\n"
            f"Required JSON Schema:\n{schema_hint}"
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }

        url = f"{self.base_url}/chat/completions"
        if self._client is not None:
            resp = self._client.post(url, headers=headers, json=payload)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip accidental markdown code fences if a gateway adds them
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        return json.loads(content)


class DeterministicRoleLLM(BaseLLM):
    """Offline deterministic role synthesizer producing validated structured handoffs.

    When `simulate_first_draft_gaps=True` (default), ReportWriter draft v0 intentionally
    includes an uncited quantitative claim and omits the Risk Mitigation heading so that
    CriticVerifier detects both defects, scores below threshold, triggers 1 revision loop,
    and verifies the repaired draft v1 passes the quality gate.
    """

    def __init__(self, *, simulate_first_draft_gaps: bool = True, quality_threshold: float = 0.85) -> None:
        self.simulate_first_draft_gaps = simulate_first_draft_gaps
        self.quality_threshold = quality_threshold

    def generate_structured(
        self,
        *,
        role: str,
        goal: str,
        backstory: str,
        task_description: str,
        expected_output: str,
        context_payloads: dict[str, dict[str, Any]],
        tool_outputs: dict[str, Any],
        output_schema: type[BaseModel] | None,
        revision_round: int = 0,
    ) -> dict[str, Any]:
        topic = self._extract_topic(task_description, context_payloads, tool_outputs)

        if output_schema is ManagerPlanHandoff or "manager" in role.lower():
            return ManagerPlanHandoff(
                topic=topic,
                delegation_order=["Researcher", "Analyst", "ReportWriter", "CriticVerifier"],
                focus_directives={
                    "Researcher": f"Collect primary metrics and verified sources for '{topic}'.",
                    "Analyst": "Compare topologies across accuracy, latency, cost, and operational risk.",
                    "ReportWriter": "Draft executive Markdown with strict [Sn] citations and full sections.",
                    "CriticVerifier": f"Audit grounding and enforce quality_score >= {self.quality_threshold}.",
                },
                quality_gate_threshold=self.quality_threshold,
            ).model_dump()

        if output_schema is ResearcherHandoff or "researcher" in role.lower():
            search_data = tool_outputs.get("knowledge_search") or {}
            raw_facts = search_data.get("facts") or []
            facts = [SourceFact.model_validate(f) for f in raw_facts]
            return ResearcherHandoff(
                topic=topic,
                facts=facts,
                key_themes=[
                    "Specialist role separation vs single-agent baselines",
                    "Bounded critic revision cost-quality frontier",
                    "Typed JSON handoff reliability and observability traces",
                ],
                data_gaps=[
                    "Long-horizon 30-day drift telemetry across heterogeneous open-weight models remains limited."
                ],
            ).model_dump()

        if output_schema is AnalystHandoff or "analyst" in role.lower():
            research = self._find_context(context_payloads, "facts")
            facts = research.get("facts", [])
            comp_data = tool_outputs.get("metric_comparator") or {}
            metrics_summary = comp_data.get("metrics_extracted") or {
                f["id"]: f.get("metric") or "qualitative" for f in facts
            }
            ev_ids = [f["id"] for f in facts]
            s1 = ev_ids[0] if len(ev_ids) > 0 else "S1"
            s2 = ev_ids[1] if len(ev_ids) > 1 else s1
            s3 = ev_ids[2] if len(ev_ids) > 2 else s1
            s4 = ev_ids[3] if len(ev_ids) > 3 else s1
            s5 = ev_ids[4] if len(ev_ids) > 4 else s1

            return AnalystHandoff(
                topic=topic,
                metrics_summary=metrics_summary,
                comparisons=[
                    ComparisonItem(
                        option="Single-Agent Tool Loop",
                        pros=["Lowest median latency (2.1s)", "Simplest deployment footprint"],
                        cons=["Higher unverified claim rate", "Context bloat on complex reports"],
                        estimated_cost="1.0x baseline",
                        latency_profile="2.1s avg",
                        best_for="Narrow coding fixes and rapid interactive lookups",
                    ),
                    ComparisonItem(
                        option="Sequential Specialist Crew + Bounded Critic (1 Revision)",
                        pros=[
                            "Cuts unsupported claims by 58%",
                            "Typed Pydantic handoffs keep parsing failures <1.2%",
                            "Full step-level audit traces cut triage time by 64%",
                        ],
                        cons=["Higher sequential latency (5.8s) vs single-agent"],
                        estimated_cost="1.4x baseline",
                        latency_profile="5.8s sequential (3.2s hybrid parallel)",
                        best_for="Enterprise research reports, architecture reviews, compliance workflows",
                    ),
                    ComparisonItem(
                        option="Unbounded Multi-Agent Debate",
                        pros=["Marginal extra error discovery on adversarial prompts"],
                        cons=["3.4x token cost inflation", "Risk of sycophantic convergence loops"],
                        estimated_cost="3.4x baseline",
                        latency_profile="12.6s+ avg",
                        best_for="Offline red-teaming with generous latency budgets",
                    ),
                ],
                tradeoffs=[
                    f"Accuracy vs Latency: Specialist crews add ~3.7s over single agents [{s4}] but cut unsupported claims by 58% [{s1}].",
                    f"Revision Depth vs Token Cost: Capping critique at 1 revision captures >92% of quality lift at 1.4x cost instead of 3.4x [{s2}].",
                    f"Strict Schemas vs Flexibility: Typed JSON handoffs cut inter-agent failure from 19% to <1.2% [{s3}].",
                ],
                risks=[
                    RiskItem(
                        risk="Unbounded critic loops causing runaway token spend",
                        severity="high",
                        mitigation="Hard-cap Crew max_revisions=1..2 and enforce step/token trace monitoring.",
                        evidence_ids=[s2, s5],
                    ),
                    RiskItem(
                        risk="Free-text handoff drift breaking downstream parsers",
                        severity="high",
                        mitigation="Validate every inter-agent payload against strict Pydantic v2 JSON schemas.",
                        evidence_ids=[s3],
                    ),
                    RiskItem(
                        risk="Sequential latency exceeding interactive SLA budgets",
                        severity="medium",
                        mitigation="Cache search results locally and parallelize independent evidence sub-queries.",
                        evidence_ids=[s4],
                    ),
                ],
                recommendation=(
                    f"Adopt a 4-role specialist crew (Researcher, Analyst, ReportWriter, CriticVerifier) "
                    f"with strict Pydantic JSON handoffs [{s3}], step-level execution tracing [{s5}], "
                    f"and a hard-capped 1-round critic revision gate [{s2}]."
                ),
            ).model_dump()

        if output_schema is ReportHandoff or "writer" in role.lower():
            return self._build_report_handoff(
                topic=topic,
                context_payloads=context_payloads,
                revision_round=revision_round,
            ).model_dump()

        if output_schema is CriticAuditHandoff or "critic" in role.lower():
            return self._build_critic_handoff(
                tool_outputs=tool_outputs,
                threshold=self.quality_threshold,
            ).model_dump()

        return {"role": role, "topic": topic, "summary": f"Completed task for {role}."}

    @staticmethod
    def _extract_topic(
        task_description: str,
        context_payloads: dict[str, dict[str, Any]],
        tool_outputs: dict[str, Any],
    ) -> str:
        for payload in context_payloads.values():
            if isinstance(payload, dict) and payload.get("topic"):
                return str(payload["topic"])
        ks = tool_outputs.get("knowledge_search")
        if isinstance(ks, dict) and ks.get("query"):
            return str(ks["query"])
        return task_description.splitlines()[0][:100]

    @staticmethod
    def _find_context(context_payloads: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        for payload in context_payloads.values():
            if isinstance(payload, dict) and key in payload:
                return payload
        return {}

    def _build_report_handoff(
        self,
        *,
        topic: str,
        context_payloads: dict[str, dict[str, Any]],
        revision_round: int,
    ) -> ReportHandoff:
        research = self._find_context(context_payloads, "facts")
        analysis = self._find_context(context_payloads, "comparisons")
        facts = research.get("facts", [])
        comparisons = analysis.get("comparisons", [])
        tradeoffs = analysis.get("tradeoffs", [])
        risks = analysis.get("risks", [])
        recommendation = analysis.get("recommendation", "Use bounded multi-agent verification.")

        all_ids = [f["id"] for f in facts]
        # On draft v0 with simulate_first_draft_gaps=True, emit two realistic defects
        # so CriticVerifier catches them and triggers revision_round=1.
        inject_gaps = self.simulate_first_draft_gaps and revision_round == 0

        evidence_bullets = [
            f"- **{f['id']} ({f.get('metric') or 'Finding'}):** {f['claim']} [{f['id']}]"
            for f in facts
        ]
        if inject_gaps:
            # Defect 1: uncited quantitative claim (no [Sn] handle)
            evidence_bullets.append(
                "- **Unverified Industry Projection:** Autonomous agent deployments will cut cloud spend by 85% within 90 days."
            )

        table_lines = [
            "| Architecture Option | Estimated Cost | Latency Profile | Best For |",
            "| ------------------- | -------------- | --------------- | -------- |",
        ]
        for item in comparisons:
            table_lines.append(
                f"| {item['option']} | {item['estimated_cost']} | {item['latency_profile']} | {item['best_for']} |"
            )

        risk_lines = [
            f"- **{r['risk']}** (*{r['severity']}*): {r['mitigation']} "
            + " ".join(f"[{eid}]" for eid in r.get("evidence_ids", []))
            for r in risks
        ]
        tradeoff_lines = [f"- {t}" for t in tradeoffs]

        ref_lines = [
            "| ID | Source Title | Confidence | URL |",
            "| -- | ------------ | ---------- | --- |",
        ]
        for f in facts:
            ref_lines.append(
                f"| [{f['id']}] | {f['source_title']} | {f['confidence']:.2f} | {f['source_url']} |"
            )

        title = f"Executive Technical Report: {topic}"
        first_id = all_ids[0] if all_ids else "S1"
        second_id = all_ids[1] if len(all_ids) > 1 else first_id
        third_id = all_ids[2] if len(all_ids) > 2 else first_id

        parts = [
            f"# {title}",
            "",
            f"> **Draft Revision:** v{revision_round + 1} | **Verification Mode:** Multi-Agent Crew (Researcher -> Analyst -> ReportWriter -> CriticVerifier)",
            "",
            "## Executive Summary",
            "",
            (
                f"This executive report evaluates **{topic}** using evidence gathered across "
                f"**{len(facts)} verified sources**. Separating specialist responsibilities with typed "
                f"JSON handoffs [{third_id}] and a bounded critic verification loop [{second_id}] "
                f"significantly lowers unsupported claims [{first_id}] while controlling token overhead."
            ),
            "",
            "## Key Evidence & Findings",
            "",
            "\n".join(evidence_bullets),
            "",
            "## Comparative Architecture Analysis",
            "",
            "\n".join(table_lines),
            "",
        ]

        sections_present = [
            "## Executive Summary",
            "## Key Evidence & Findings",
            "## Comparative Architecture Analysis",
        ]

        if not inject_gaps:
            # Defect 2 omitted on v0, restored cleanly on revision v1!
            parts.extend(
                [
                    "## Trade-Offs & Risk Mitigation",
                    "",
                    "### Key Trade-Offs",
                    "\n".join(tradeoff_lines),
                    "",
                    "### Operational Risks & Mitigations",
                    "\n".join(risk_lines),
                    "",
                ]
            )
            sections_present.append("## Trade-Offs & Risk Mitigation")

        parts.extend(
            [
                "## Recommendations",
                "",
                recommendation,
                "",
                "## References",
                "",
                "\n".join(ref_lines),
                "",
            ]
        )
        sections_present.extend(["## Recommendations", "## References"])

        return ReportHandoff(
            title=title,
            markdown_body="\n".join(parts),
            cited_source_ids=all_ids,
            sections_present=sections_present,
            revision_number=revision_round,
        )

    @staticmethod
    def _build_critic_handoff(
        *,
        tool_outputs: dict[str, Any],
        threshold: float,
    ) -> CriticAuditHandoff:
        audit = tool_outputs.get("citation_audit") or {}
        score = float(audit.get("quality_score", 0.0))
        ungrounded = list(audit.get("ungrounded_claims", []))
        missing = list(audit.get("missing_sections", []))

        issues: list[AuditIssue] = []
        for claim in ungrounded:
            issues.append(
                AuditIssue(
                    severity="high",
                    issue_type="ungrounded_claim",
                    description=f"Unsupported or uncited quantitative claim: {claim}",
                    suggested_fix="Remove the unverified claim or attach a valid [Sn] citation from ResearcherHandoff.",
                )
            )
        for sec in missing:
            issues.append(
                AuditIssue(
                    severity="high",
                    issue_type="missing_section",
                    description=f"Required report section missing: '{sec}'.",
                    suggested_fix=f"Add '{sec}' incorporating AnalystHandoff trade-offs and risk mitigations with citations.",
                )
            )

        passed = score >= threshold and not ungrounded and not missing
        if passed:
            instructions = "All quantitative claims are grounded in Researcher sources and all required sections are present. Approved."
        else:
            instructions = (
                f"Revise report to raise quality_score from {score:.2f} to >= {threshold:.2f}: "
                + "; ".join(i.suggested_fix for i in issues)
            )

        return CriticAuditHandoff(
            quality_score=score,
            passed=passed,
            threshold=threshold,
            ungrounded_claims=ungrounded,
            missing_sections=missing,
            issues=issues,
            revision_instructions=instructions,
        )
