"""Optional bridge adapter translating pure-Python Crew definitions to upstream `crewai` if installed."""

from __future__ import annotations

import importlib.util
from typing import Any

from multi_agent.core import Crew


def is_crewai_available() -> bool:
    """Return True if optional external `crewai` package is installed."""
    return importlib.util.find_spec("crewai") is not None


def to_crewai_crew(crew: Crew) -> Any:
    """Convert a pure-Python `Crew` into an external `crewai.Crew` instance.

    Raises RuntimeError with installation hint if `crewai` is not installed.
    """
    if not is_crewai_available():
        raise RuntimeError(
            "Optional dependency 'crewai' is not installed. "
            "Use the built-in pure-Python Crew.kickoff() engine or install with `pip install '.[crewai]'`."
        )

    import crewai  # type: ignore[import-not-found]

    agent_map = {}
    for ag in crew.agents:
        agent_map[ag.role] = crewai.Agent(
            role=ag.role,
            goal=ag.goal,
            backstory=ag.backstory,
            verbose=ag.verbose,
        )

    crewai_tasks = []
    for task in crew.tasks:
        crewai_tasks.append(
            crewai.Task(
                description=task.description,
                expected_output=task.expected_output,
                agent=agent_map[task.agent.role],
            )
        )

    proc = (
        crewai.Process.hierarchical
        if crew.process == "hierarchical"
        else crewai.Process.sequential
    )
    return crewai.Crew(
        agents=list(agent_map.values()),
        tasks=crewai_tasks,
        process=proc,
    )
