"""Facade used by the UI, the CLI and the evaluation script.

Hides LangGraph details: builds the graph once, runs one conversation turn per call, and
surfaces interrupts (clarification questions) as a normal result so the caller can simply ask
the user and call ``resume``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from rem_agent.config import Settings, get_settings
from rem_agent.data import DataCatalog, LedgerData, load_ledger
from rem_agent.graph import build_graph
from rem_agent.llm import get_chat_model
from rem_agent.schemas import ResolvedTask, SpecialistResult, TraceEvent


@dataclass
class TurnResult:
    answer: str
    thread_id: str
    interrupted: bool = False
    clarification: str | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    tasks: list[ResolvedTask] = field(default_factory=list)
    results: list[SpecialistResult] = field(default_factory=list)
    verification: dict | None = None
    elapsed_s: float = 0.0


class Assistant:
    def __init__(self, settings: Settings | None = None, llm=None):
        self.settings = settings or get_settings()
        self.ledger: LedgerData = load_ledger(self.settings.data_path)
        self.catalog = DataCatalog(self.ledger)
        self.llm = llm or get_chat_model(self.settings)
        self.checkpointer = InMemorySaver()
        self.graph = build_graph(
            self.ledger,
            self.catalog,
            self.llm,
            self.checkpointer,
            max_tool_rounds=self.settings.max_tool_rounds,
        )

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        dedupe: bool = False,
        thread_id: str | None = None,
    ) -> TurnResult:
        """Run one turn. A fresh thread per turn; conversation memory is passed as ``history``."""
        thread_id = thread_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        t0 = time.perf_counter()
        state = self.graph.invoke(
            {"question": question, "history": history or [], "dedupe": dedupe}, config
        )
        return self._to_result(state, thread_id, time.perf_counter() - t0)

    def resume(self, thread_id: str, reply: str) -> TurnResult:
        """Continue a turn that paused for clarification."""
        config = {"configurable": {"thread_id": thread_id}}
        t0 = time.perf_counter()
        state = self.graph.invoke(Command(resume=reply), config)
        return self._to_result(state, thread_id, time.perf_counter() - t0)

    def _to_result(self, state: dict, thread_id: str, elapsed_s: float = 0.0) -> TurnResult:
        interrupts = state.get("__interrupt__") or []
        if interrupts:
            payload = interrupts[0].value
            question = payload.get("question") if isinstance(payload, dict) else str(payload)
            return TurnResult(
                answer=question,
                thread_id=thread_id,
                interrupted=True,
                clarification=question,
                trace=list(state.get("trace", [])),
                tasks=list(state.get("tasks", [])),
                elapsed_s=elapsed_s,
            )
        return TurnResult(
            answer=state.get("answer") or "I couldn't produce an answer.",
            thread_id=thread_id,
            trace=list(state.get("trace", [])),
            tasks=list(state.get("tasks", [])),
            results=list(state.get("results", [])),
            verification=state.get("verification"),
            elapsed_s=elapsed_s,
        )
