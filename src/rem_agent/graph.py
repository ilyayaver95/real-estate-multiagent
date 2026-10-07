"""LangGraph workflow.

    START -> guard -> router -> extractor -> resolver --(ok)--> [Send] specialists -> synthesize
              |                                   |                                     |
              +--(bad input)--> END               +--(ambiguous)--> clarify -> router    verify
                                                   (interrupt: waits for the user)        |
                                                                        (unverified numbers, once)
                                                                                          v
                                                                                    synthesize / END

* ``Send`` fans compound questions out to specialists in parallel; results are reduced into
  ``results`` with an ``operator.add`` reducer.
* ``clarify`` uses ``interrupt`` so the graph pauses with a question and resumes on the user's
  reply with the same thread id (needs the checkpointer).
* Every node appends a ``TraceEvent`` so the UI can show the step-by-step reasoning.
"""

from __future__ import annotations

import time
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from rem_agent.agents import extractor as extractor_agent
from rem_agent.agents import router as router_agent
from rem_agent.agents.guard import check_input
from rem_agent.agents.resolver import has_concrete_mention, resolve_all
from rem_agent.agents.specialists import (
    run_fallback_specialist,
    run_knowledge_specialist,
    run_no_data_specialist,
    run_not_found_specialist,
    run_tool_specialist,
)
from rem_agent.agents.synthesizer import synthesize
from rem_agent.agents.verifier import verify
from rem_agent.data.catalog import DataCatalog
from rem_agent.data.loader import LedgerData
from rem_agent.schemas import ExtractedSlots, Intent, SpecialistResult, TraceEvent
from rem_agent.state import GraphState, TaskState

MAX_REVISIONS = 1
NO_SLOTS_INTENTS = {
    Intent.OUT_OF_SCOPE,
    Intent.GENERAL_KNOWLEDGE,
    Intent.DATA_SCOPE,
    Intent.CLARIFICATION,
}


def build_graph(
    ledger: LedgerData,
    catalog: DataCatalog,
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    max_tool_rounds: int = 6,
):
    """Compile the workflow. Dependencies are injected so tests can pass a fake model."""

    def _trace(node: str, summary: str, detail=None, t0: float | None = None) -> TraceEvent:
        return TraceEvent(
            node=node,
            summary=summary,
            detail=detail,
            duration_ms=int((time.perf_counter() - t0) * 1000) if t0 else None,
        )

    # ---- nodes ---------------------------------------------------------------------------------
    def guard(state: GraphState) -> dict:
        t0 = time.perf_counter()
        res = check_input(state.get("question"))
        if not res.ok:
            return {
                "cleaned_question": res.cleaned,
                "guard_message": res.message,
                "answer": res.message,
                "trace": [_trace("guard", f"Rejected input ({res.reason})", res.message, t0)],
            }
        return {
            "cleaned_question": res.cleaned,
            "guard_message": None,
            "trace": [_trace("guard", "Input accepted", None, t0)],
        }

    def router(state: GraphState) -> dict:
        t0 = time.perf_counter()
        out = router_agent.route(llm, catalog, state["cleaned_question"], state.get("history", []))
        summary = "; ".join(f"{sq.id}: {sq.intent.value}" for sq in out.sub_questions)
        if out.needs_clarification:
            summary += " (needs clarification)"
        return {
            "router": out,
            "results": [],
            "trace": [_trace("router", summary, out.model_dump(mode="json"), t0)],
        }

    def extractor(state: GraphState) -> dict:
        t0 = time.perf_counter()
        sub_questions = state["router"].sub_questions
        if all(sq.intent in NO_SLOTS_INTENTS for sq in sub_questions):
            slots = {sq.id: ExtractedSlots(sub_question_id=sq.id) for sq in sub_questions}
            return {
                "slots": slots,
                "trace": [
                    _trace("extractor", "Skipped: no entities needed for these intents", None, t0)
                ],
            }
        slots = extractor_agent.extract(llm, catalog, sub_questions, state["cleaned_question"])
        detail = {k: v.model_dump(mode="json", exclude_defaults=True) for k, v in slots.items()}
        return {
            "slots": slots,
            "trace": [
                _trace("extractor", f"Extracted slots for {len(slots)} sub-question(s)", detail, t0)
            ],
        }

    def resolver(state: GraphState) -> dict:
        t0 = time.perf_counter()
        tasks = resolve_all(
            state["router"].sub_questions, state["slots"], catalog, state["cleaned_question"]
        )
        bits = []
        for t in tasks:
            desc = t.specialist
            if t.properties:
                desc += f" props={t.properties}"
            if t.tenants:
                desc += f" tenants={t.tenants}"
            if t.period:
                desc += f" period={t.period.label}"
            if t.comparison_period:
                desc += f" vs {t.comparison_period.label}"
            if t.issues:
                desc += f" issues={[i.kind for i in t.issues]}"
            bits.append(f"{t.id}: {desc}")
        return {
            "tasks": tasks,
            "trace": [
                _trace(
                    "resolver",
                    "; ".join(bits),
                    [t.model_dump(mode="json", exclude_defaults=True) for t in tasks],
                    t0,
                )
            ],
        }

    def clarify(state: GraphState) -> dict:
        question = _clarification_text(state)
        reply = interrupt({"question": question})  # pauses here until the user answers
        merged = f"{state['cleaned_question']} (clarification from user: {reply})"
        return {
            "question": merged,
            "cleaned_question": merged,
            "clarification": question,
            "trace": [
                _trace(
                    "clarify",
                    "Asked the user for clarification and received a reply",
                    {"question": question, "reply": reply},
                )
            ],
        }

    def specialist(state: TaskState) -> dict:
        t0 = time.perf_counter()
        task = state["task"]
        try:
            if task.specialist == "fallback":
                res = run_fallback_specialist(catalog, task)
            elif task.unresolved_entities and task.specialist != "knowledge":
                res = run_not_found_specialist(catalog, task)
            elif task.out_of_coverage and task.specialist != "knowledge":
                res = run_no_data_specialist(catalog, task)
            elif task.specialist == "knowledge":
                res = run_knowledge_specialist(llm, catalog, task)
            else:
                res = run_tool_specialist(
                    llm,
                    ledger,
                    catalog,
                    task,
                    dedupe=state.get("dedupe", False),
                    max_rounds=max_tool_rounds,
                )
        except Exception as exc:  # never let one specialist kill the turn
            res = SpecialistResult(
                task_id=task.id,
                specialist=task.specialist,
                answer="I ran into a problem computing this part of the answer.",
                error=f"{type(exc).__name__}: {exc}",
            )
        calls = [
            f"{c.tool}({', '.join(f'{k}={v}' for k, v in c.args.items())})" for c in res.tool_calls
        ]
        return {
            "results": [res],
            "trace": [
                _trace(
                    f"specialist:{task.specialist}",
                    f"{task.id}: {len(res.tool_calls)} tool call(s)"
                    + (f" - {res.error}" if res.error else ""),
                    {"tool_calls": calls, "answer": res.answer, "caveats": res.caveats},
                    t0,
                )
            ],
        }

    def synthesizer(state: GraphState) -> dict:
        t0 = time.perf_counter()
        feedback = state.get("feedback")
        answer = synthesize(
            llm, state["cleaned_question"], state["tasks"], state["results"], feedback
        )
        how = (
            "revised after verification"
            if feedback
            else (
                "single specialist answer passed through"
                if len(state["results"]) == 1
                else f"merged {len(state['results'])} specialist answers"
            )
        )
        return {"answer": answer, "trace": [_trace("synthesizer", how, None, t0)]}

    def verifier(state: GraphState) -> dict:
        t0 = time.perf_counter()
        if _only_non_numeric(state):
            report = {
                "ok": True,
                "checked": 0,
                "verified": 0,
                "unverified": [],
                "skipped": "no tool-computed figures in this answer",
            }
        else:
            report = verify(
                state["answer"], state["results"], ignore_names=catalog.properties + catalog.tenants
            )
        revision = state.get("revision", 0)
        update: dict = {"verification": report, "revision": revision}
        if report["ok"] or revision >= MAX_REVISIONS:
            summary = (
                f"{report['verified']}/{report['checked']} figures traced to tool outputs"
                if report["checked"]
                else "no tool-computed figures to verify"
            )
            update["feedback"] = None
        else:
            update["revision"] = revision + 1
            update["feedback"] = (
                "These figures do not appear in any specialist answer or tool result: "
                f"{report['unverified']}. Remove or replace them with figures that do."
            )
            summary = f"{len(report['unverified'])} unverified figure(s); requesting one revision"
        update["trace"] = [_trace("verifier", summary, report, t0)]
        return update

    # ---- routing functions ---------------------------------------------------------------------
    def after_guard(state: GraphState) -> Literal["router", END]:
        return END if state.get("guard_message") else "router"

    def after_resolver(state: GraphState):
        tasks = state["tasks"]
        blocking = any(t.blocking_issues for t in tasks)
        clarification_only = all(t.intent == Intent.CLARIFICATION for t in tasks)
        # Router asked for clarification AND nothing concrete was extracted anywhere -> ask.
        # Otherwise act with explicit assumptions (fallback intents are always answered directly).
        # Vague = the user's own words contain nothing concrete to act on (no period, entity,
        # account or metric word) while the intent needs data. Deterministic on purpose: the
        # router's needs_clarification flag varies between runs for the same input.
        vague = not has_concrete_mention(state["cleaned_question"]) and not all(
            t.specialist in ("fallback", "knowledge") for t in tasks
        )
        if (blocking or clarification_only or vague) and not state.get("clarification"):
            return "clarify"
        return [
            Send(
                "specialist",
                {
                    "task": t,
                    "question": state["cleaned_question"],
                    "dedupe": state.get("dedupe", False),
                },
            )
            for t in state["tasks"]
        ]

    def after_verifier(state: GraphState) -> Literal["synthesizer", END]:
        return "synthesizer" if state.get("feedback") else END

    # ---- wiring --------------------------------------------------------------------------------
    g = StateGraph(GraphState)
    g.add_node("guard", guard)
    g.add_node("router", router)
    g.add_node("extractor", extractor)
    g.add_node("resolver", resolver)
    g.add_node("clarify", clarify)
    g.add_node("specialist", specialist)
    g.add_node("synthesizer", synthesizer)
    g.add_node("verifier", verifier)

    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", after_guard, ["router", END])
    g.add_edge("router", "extractor")
    g.add_edge("extractor", "resolver")
    g.add_conditional_edges("resolver", after_resolver, ["clarify", "specialist"])
    g.add_edge("clarify", "router")
    g.add_edge("specialist", "synthesizer")
    g.add_edge("synthesizer", "verifier")
    g.add_conditional_edges("verifier", after_verifier, ["synthesizer", END])

    return g.compile(checkpointer=checkpointer or InMemorySaver())


def _clarification_text(state: GraphState) -> str:
    rt = state["router"]
    for t in state["tasks"]:
        for issue in t.blocking_issues:
            opts = ", ".join(issue.suggestions)
            return f"{issue.message} Which one do you mean: {opts}?"
    if rt.clarification_question:
        return rt.clarification_question
    return (
        "Could you be more specific? For example: a property (Building 17, 120, 140, 160, 180), "
        "a tenant, a period (2024, Q1 2025), or what to compute (P&L, comparison, top tenants, "
        "anomalies)."
    )


def _only_non_numeric(state: GraphState) -> bool:
    """Knowledge / fallback answers legitimately contain numbers that no tool produced."""
    return all(r.specialist in ("knowledge", "fallback") for r in state["results"])
