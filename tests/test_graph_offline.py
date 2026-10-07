"""Graph wiring tests with the LLM agents replaced by deterministic fakes.

They prove: guard short-circuit, Send fan-out and reduction, the clarification interrupt +
resume cycle, pass-through synthesis for single tasks, and the verifier revision loop.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.types import Command

from rem_agent import graph as graph_module
from rem_agent.graph import build_graph
from rem_agent.schemas import (
    ExtractedSlots,
    Intent,
    RouterOutput,
    SpecialistResult,
    SubQuestion,
    TimeSpec,
)


class Fakes:
    """Mutable container the monkeypatched agents read from."""

    router: RouterOutput
    slots: dict[str, ExtractedSlots]
    specialist_answers: dict[str, str]
    synth_calls: int = 0


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()
    f.synth_calls = 0

    def fake_route(llm, catalog, question, history=None):
        return f.router

    def fake_extract(llm, catalog, sub_questions, original_message=None):
        return {
            sq.id: f.slots.get(sq.id, ExtractedSlots(sub_question_id=sq.id)) for sq in sub_questions
        }

    def fake_tool_specialist(llm, ledger, catalog, task, dedupe=False, max_rounds=6):
        return SpecialistResult(
            task_id=task.id,
            specialist=task.specialist,
            answer=f.specialist_answers.get(task.id, f"answer for {task.id}"),
            numbers_used=[1171521.55, 592124.15],
        )

    real_synth = graph_module.synthesize

    def counting_synth(llm, question, tasks, results, feedback=None):
        f.synth_calls += 1
        return real_synth(llm, question, tasks, results, feedback)

    monkeypatch.setattr(graph_module.router_agent, "route", fake_route)
    monkeypatch.setattr(graph_module.extractor_agent, "extract", fake_extract)
    monkeypatch.setattr(graph_module, "run_tool_specialist", fake_tool_specialist)
    monkeypatch.setattr(graph_module, "synthesize", counting_synth)
    return f


@pytest.fixture
def graph(ledger, catalog):
    llm = FakeListChatModel(responses=["merged answer: €1,171,521.55 and €592,124.15"])
    return build_graph(ledger, catalog, llm)


def _run(graph, question, **kw):
    cfg = {"configurable": {"thread_id": question[:20]}}
    return graph.invoke({"question": question, "history": [], **kw}, cfg), cfg


def test_guard_short_circuits_without_llm(graph, fakes):
    state, _ = _run(graph, "   ")
    assert "empty" in state["answer"].lower() or "Your message is empty" in state["answer"]
    assert [t.node for t in state["trace"]] == ["guard"]
    assert "router" not in state


def test_single_task_passthrough(graph, fakes):
    fakes.router = RouterOutput(
        sub_questions=[
            SubQuestion(id="q1", text="total pnl 2024", intent=Intent.PNL, rationale="r")
        ],
        needs_clarification=False,
    )
    fakes.slots = {
        "q1": ExtractedSlots(
            sub_question_id="q1", timeframes=[TimeSpec(raw="2024", kind="year", year=2024)]
        )
    }
    fakes.specialist_answers = {"q1": "Net P&L for 2024 was €1,171,521.55."}
    state, _ = _run(graph, "What is the total P&L for 2024?")
    assert state["answer"] == "Net P&L for 2024 was €1,171,521.55."
    assert state["tasks"][0].period.label == "2024"
    nodes = [t.node for t in state["trace"]]
    assert nodes == [
        "guard",
        "router",
        "extractor",
        "resolver",
        "specialist:finance",
        "synthesizer",
        "verifier",
    ]
    assert state["verification"]["ok"] and state["verification"]["checked"] == 1


def test_compound_fans_out_and_merges(graph, fakes):
    fakes.router = RouterOutput(
        sub_questions=[
            SubQuestion(id="q1", text="top tenants", intent=Intent.TENANT_ANALYSIS, rationale="r"),
            SubQuestion(
                id="q2", text="anything unusual", intent=Intent.ANOMALY_AUDIT, rationale="r"
            ),
        ],
        needs_clarification=False,
    )
    fakes.slots = {}
    fakes.specialist_answers = {"q1": "Tenant 7 leads with €592,124.15.", "q2": "Found duplicates."}
    state, _ = _run(graph, "Who are my top tenants, and is anything unusual?")
    specialists = sorted(t.node for t in state["trace"] if t.node.startswith("specialist:"))
    assert specialists == ["specialist:audit", "specialist:portfolio"]
    assert len(state["results"]) == 2
    assert state["answer"].startswith("merged answer")  # LLM synthesizer used for 2 results
    assert fakes.synth_calls == 1


def test_fallback_for_valuation_is_deterministic(graph, fakes):
    fakes.router = RouterOutput(
        sub_questions=[
            SubQuestion(
                id="q1",
                text="price of 123 Main St vs 456 Oak Ave",
                intent=Intent.VALUATION_UNSUPPORTED,
                rationale="r",
            )
        ],
        needs_clarification=False,
    )
    fakes.slots = {
        "q1": ExtractedSlots(sub_question_id="q1", properties=["123 Main St", "456 Oak Ave"])
    }
    state, _ = _run(graph, "What is the price of my asset at 123 Main St compared to 456 Oak Ave?")
    ans = state["answer"]
    assert "no prices" in ans or "valuations" in ans
    assert "123 Main St" in ans and "Building 17" in ans
    assert state["results"][0].specialist == "fallback"


def test_clarification_interrupt_and_resume(graph, fakes):
    fakes.router = RouterOutput(
        sub_questions=[
            SubQuestion(id="q1", text="numbers please", intent=Intent.CLARIFICATION, rationale="r")
        ],
        needs_clarification=True,
        clarification_question="Which numbers: P&L, revenue or something else, and which period?",
    )
    fakes.slots = {}
    state, cfg = _run(graph, "numbers please")
    assert "__interrupt__" in state
    assert "Which numbers" in state["__interrupt__"][0].value["question"]

    # user answers; router now (fake) classifies as PNL
    fakes.router = RouterOutput(
        sub_questions=[SubQuestion(id="q1", text="P&L 2024", intent=Intent.PNL, rationale="r")],
        needs_clarification=False,
    )
    fakes.specialist_answers = {"q1": "Net P&L for 2024 was €1,171,521.55."}
    state2 = graph.invoke(Command(resume="P&L for 2024"), cfg)
    assert "__interrupt__" not in state2
    assert state2["answer"].startswith("Net P&L")
    assert "clarification from user: P&L for 2024" in state2["cleaned_question"]
    nodes = [t.node for t in state2["trace"]]
    assert "clarify" in nodes and nodes.count("router") == 2


def test_verifier_requests_one_revision(graph, fakes):
    fakes.router = RouterOutput(
        sub_questions=[SubQuestion(id="q1", text="pnl", intent=Intent.PNL, rationale="r")],
        needs_clarification=False,
    )
    fakes.slots = {}
    # specialist answer contains a figure that no tool produced
    fakes.specialist_answers = {"q1": "Net P&L was €9,999,999.99."}
    state, _ = _run(graph, "pnl please")
    nodes = [t.node for t in state["trace"]]
    assert nodes.count("synthesizer") == 2 and nodes.count("verifier") == 2
    assert state["revision"] == 1
    assert state["answer"].startswith("merged answer")  # revised via the LLM synthesizer


def test_prefetch_selection():
    from rem_agent.agents.specialists import prefetch_call
    from rem_agent.schemas import ResolvedTask

    def task(intent, **kw):
        return ResolvedTask(id="q", text="t", intent=intent, specialist="x", **kw)

    assert prefetch_call(task(Intent.PNL))[0] == "get_pnl"
    assert prefetch_call(task(Intent.PERIOD_COMPARISON))[0] == "compare_periods"
    assert prefetch_call(task(Intent.TREND, granularity="quarter")) == (
        "get_trend",
        {"granularity": "quarter"},
    )
    assert prefetch_call(task(Intent.PROPERTY_DETAILS, properties=["Building 17"])) == (
        "get_property_details",
        {"property_name": "Building 17"},
    )
    assert prefetch_call(task(Intent.PROPERTY_DETAILS))[0] == "get_portfolio_overview"
    assert prefetch_call(task(Intent.TENANT_ANALYSIS, top_n=3)) == ("get_top_tenants", {"n": 3})
    assert (
        prefetch_call(task(Intent.TENANT_ANALYSIS, tenants=["Tenant 7"]))[0] == "get_tenant_details"
    )
    assert prefetch_call(task(Intent.ANOMALY_AUDIT))[0] == "run_anomaly_audit"
    assert prefetch_call(task(Intent.GENERAL_KNOWLEDGE)) is None


def test_merge_duplicate_tasks(catalog):
    from rem_agent.agents.resolver import resolve_all

    sqs = [
        SubQuestion(
            id="q1",
            text="Monthly net trend for Building 140 in 2024",
            intent=Intent.TREND,
            rationale="r",
        ),
        SubQuestion(
            id="q2",
            text="Which month was worst for Building 140 in 2024?",
            intent=Intent.TREND,
            rationale="r",
        ),
        SubQuestion(
            id="q3", text="Is anything unusual?", intent=Intent.ANOMALY_AUDIT, rationale="r"
        ),
    ]
    slots = {
        "q1": ExtractedSlots(
            sub_question_id="q1",
            properties=["Building 140"],
            timeframes=[TimeSpec(raw="2024", kind="year", year=2024)],
        ),
        "q2": ExtractedSlots(
            sub_question_id="q2",
            properties=["Building 140"],
            timeframes=[TimeSpec(raw="2024", kind="year", year=2024)],
        ),
    }
    tasks = resolve_all(
        sqs,
        slots,
        catalog,
        "Monthly net trend for Building 140 in 2024 - which month was worst? Anything unusual?",
    )
    assert [t.id for t in tasks] == ["q1", "q3"]
    assert "Also:" in tasks[0].text


def test_verifier_regex_does_not_read_main_as_millions():
    from rem_agent.agents.verifier import extract_numbers

    vals = [v for v, _ in extract_numbers("No data for 123 Main St; €1.2M revenue and 5k rows")]
    assert 123.0 in vals and 1_200_000.0 in vals and 5000.0 in vals
    assert 123_000_000.0 not in vals
