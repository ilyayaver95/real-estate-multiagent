"""Graph state.

Two schemas: the overall conversation-turn state, and the per-task state that ``Send`` hands to
each specialist. Lists that several parallel specialists append to use ``operator.add`` reducers.
Everything is JSON-serialisable (pydantic models / plain dicts) so the checkpointer can persist
it between turns.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from rem_agent.schemas import (
    ExtractedSlots,
    ResolvedTask,
    RouterOutput,
    SpecialistResult,
    TraceEvent,
)


class GraphState(TypedDict, total=False):
    # inputs
    question: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}, ...] before this turn
    dedupe: bool
    # pipeline
    cleaned_question: str
    guard_message: str | None
    router: RouterOutput | None
    slots: dict[str, ExtractedSlots]
    tasks: list[ResolvedTask]
    clarification: str | None
    results: Annotated[list[SpecialistResult], operator.add]
    # outputs
    answer: str
    verification: dict | None
    revision: int
    feedback: str | None
    trace: Annotated[list[TraceEvent], operator.add]


class TaskState(TypedDict):
    """What a single specialist receives via ``Send``."""

    task: ResolvedTask
    question: str
    dedupe: bool
