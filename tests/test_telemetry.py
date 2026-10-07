"""Telemetry: metrics flattening, store round-trip and KPI aggregation (offline)."""

from rem_agent.assistant import TurnResult
from rem_agent.schemas import SpecialistResult, TraceEvent
from rem_agent.telemetry import LLMUsageHandler, MetricsStore, build_metrics, price_for, summarize


def _result(**kw) -> TurnResult:
    base = dict(
        answer="ok",
        thread_id="t1",
        elapsed_s=5.0,
        trace=[
            TraceEvent(node="guard", summary="Input accepted"),
            TraceEvent(
                node="router",
                summary="q1: pnl",
                duration_ms=1200,
                detail={"sub_questions": [{"intent": "pnl"}, {"intent": "anomaly_audit"}]},
            ),
            TraceEvent(
                node="specialist:finance",
                summary="q1",
                duration_ms=2500,
                detail={"tool_calls": ["get_pnl()", "get_trend()"]},
            ),
            TraceEvent(node="synthesizer", summary="revised after verification", duration_ms=900),
        ],
        results=[SpecialistResult(task_id="q1", specialist="finance", answer="x")],
        verification={"checked": 4, "verified": 4, "ok": True},
    )
    base.update(kw)
    return TurnResult(**base)


def test_build_metrics_flattens_trace_and_usage():
    h = LLMUsageHandler()
    h.llm_calls, h.input_tokens, h.output_tokens, h.llm_time_s = 3, 4000, 300, 4.2
    m = build_metrics(_result(), h, "gpt-4o-mini", "What is the P&L?", dedupe=False)
    assert m.intents == ["pnl", "anomaly_audit"]
    assert m.specialists == ["finance"] and m.tool_calls == 2
    assert m.nodes["specialist:finance"] == 2500 and m.revision is True
    assert m.verification_ok is True and m.figures_verified == 4
    assert m.total_tokens == 4300
    pin, pout = price_for("gpt-4o-mini")
    assert m.cost_usd == round((4000 * pin + 300 * pout) / 1e6, 6)


def test_build_metrics_flags_guard_and_interrupt():
    h = LLMUsageHandler()
    guard = _result(
        trace=[TraceEvent(node="guard", summary="Rejected input (empty)")],
        results=[],
        verification=None,
    )
    m = build_metrics(guard, h, "gpt-4o-mini", "", False)
    assert m.guard_rejected and m.verification_ok is None and m.llm_calls == 0
    m2 = build_metrics(_result(interrupted=True, verification=None), h, "gpt-4o-mini", "q", False)
    assert m2.interrupted


def test_store_roundtrip_and_summary(tmp_path):
    store = MetricsStore(tmp_path / "m.jsonl")
    h = LLMUsageHandler()
    h.llm_calls, h.input_tokens, h.output_tokens = 2, 1000, 100
    for i, el in enumerate((2.0, 4.0, 6.0, 20.0)):
        store.record(
            build_metrics(_result(elapsed_s=el, thread_id=f"t{i}"), h, "gpt-4o-mini", "q", False)
        )
    rows = store.load()
    assert len(rows) == 4 and rows[-1].elapsed_s == 20.0
    s = summarize(rows)
    assert s["requests"] == 4 and s["avg_latency_s"] == 8.0 and s["p95_latency_s"] == 20.0
    assert s["verification_pass_pct"] == 100.0 and s["revision_pct"] == 100.0
    assert s["node_avg_ms"]["router"] == 1200 and s["total_input_tokens"] == 4000
    store.clear()
    assert store.load() == [] and summarize([]) == {}


def test_price_lookup_prefix_and_unknown(monkeypatch):
    assert price_for("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
    assert price_for("some-other-model") == (0.0, 0.0)
    monkeypatch.setenv("REM_PRICE_INPUT", "1")
    monkeypatch.setenv("REM_PRICE_OUTPUT", "2")
    assert price_for("anything") == (1.0, 2.0)
