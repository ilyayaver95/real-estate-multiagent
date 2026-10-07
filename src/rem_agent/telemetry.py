"""Per-request telemetry: latency, LLM usage, cost, quality signals.

* ``LLMUsageHandler`` is a LangChain callback that counts model calls and tokens for one graph
  run (callbacks passed in the invoke config propagate to every model call inside the nodes).
* ``TurnMetrics`` is the flat record written per request; ``MetricsStore`` appends it to a JSONL
  file (ephemeral on Streamlit Cloud, durable locally) and loads it back for the dashboard.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from rem_agent.config import PROJECT_ROOT

# USD per 1M tokens (input, output). Override with REM_PRICE_INPUT / REM_PRICE_OUTPUT.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
}


def price_for(model: str) -> tuple[float, float]:
    env_in, env_out = os.getenv("REM_PRICE_INPUT"), os.getenv("REM_PRICE_OUTPUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    for name, prices in PRICES_PER_M.items():
        if model.startswith(name):
            return prices
    return (0.0, 0.0)


class LLMUsageHandler(BaseCallbackHandler):
    """Counts LLM calls, tokens and model latency for one request."""

    def __init__(self) -> None:
        super().__init__()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_time_s = 0.0
        self.errors = 0
        self._starts: dict[Any, float] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self.llm_calls += 1
        self._starts[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self.llm_calls += 1
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        t0 = self._starts.pop(run_id, None)
        if t0 is not None:
            self.llm_time_s += time.perf_counter() - t0
        for gens in response.generations:
            for gen in gens:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage.get("output_tokens", 0) or 0)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self.errors += 1
        self._starts.pop(run_id, None)


@dataclass
class TurnMetrics:
    ts: float
    thread_id: str
    question: str
    model: str
    elapsed_s: float
    llm_calls: int
    llm_time_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    intents: list[str] = field(default_factory=list)
    specialists: list[str] = field(default_factory=list)
    tool_calls: int = 0
    nodes: dict[str, int] = field(default_factory=dict)  # node -> ms
    interrupted: bool = False
    guard_rejected: bool = False
    revision: bool = False
    verification_ok: bool | None = None
    figures_checked: int = 0
    figures_verified: int = 0
    error: bool = False
    dedupe: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def build_metrics(
    result, handler: LLMUsageHandler, model: str, question: str, dedupe: bool
) -> TurnMetrics:
    """Flatten a TurnResult + usage handler into one metrics record."""
    intents: list[str] = []
    specialists: list[str] = []
    nodes: dict[str, int] = {}
    tool_calls = 0
    revision = False
    guard_rejected = False
    for ev in result.trace:
        nodes[ev.node] = nodes.get(ev.node, 0) + (ev.duration_ms or 0)
        if ev.node == "router" and isinstance(ev.detail, dict):
            intents = [sq.get("intent") for sq in ev.detail.get("sub_questions", [])]
        elif ev.node.startswith("specialist:"):
            specialists.append(ev.node.split(":", 1)[1])
            if isinstance(ev.detail, dict):
                tool_calls += len(ev.detail.get("tool_calls", []))
        elif ev.node == "guard" and ev.summary.startswith("Rejected"):
            guard_rejected = True
        elif ev.node == "synthesizer" and "revised" in ev.summary:
            revision = True
    v = result.verification or {}
    pin, pout = price_for(model)
    cost = (handler.input_tokens * pin + handler.output_tokens * pout) / 1_000_000
    return TurnMetrics(
        ts=time.time(),
        thread_id=result.thread_id,
        question=question[:300],
        model=model,
        elapsed_s=round(result.elapsed_s, 3),
        llm_calls=handler.llm_calls,
        llm_time_s=round(handler.llm_time_s, 3),
        input_tokens=handler.input_tokens,
        output_tokens=handler.output_tokens,
        cost_usd=round(cost, 6),
        intents=[i for i in intents if i],
        specialists=specialists,
        tool_calls=tool_calls,
        nodes=nodes,
        interrupted=result.interrupted,
        guard_rejected=guard_rejected,
        revision=revision,
        verification_ok=(
            None if not v or v.get("skipped") or not v.get("checked") else bool(v.get("ok"))
        ),
        figures_checked=int(v.get("checked", 0) or 0),
        figures_verified=int(v.get("verified", 0) or 0),
        error=any(r.error for r in result.results) or handler.errors > 0,
        dedupe=dedupe,
    )


class MetricsStore:
    """Append-only JSONL store. Safe to call from several Streamlit sessions."""

    def __init__(self, path: str | Path | None = None):
        p = Path(path or os.getenv("REM_METRICS_PATH", "outputs/metrics.jsonl"))
        self.path = p if p.is_absolute() else PROJECT_ROOT / p

    def record(self, m: TurnMetrics) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
        except OSError:
            pass  # telemetry must never break a request

    def load(self) -> list[TurnMetrics]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(TurnMetrics(**json.loads(line)))
                except (ValueError, TypeError):
                    continue
        return out

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def summarize(rows: list[TurnMetrics]) -> dict:
    """Aggregate KPIs for the dashboard."""
    if not rows:
        return {}
    lat = sorted(r.elapsed_s for r in rows)
    answered = [r for r in rows if not r.guard_rejected and not r.interrupted]
    verified_rows = [r for r in rows if r.verification_ok is not None]
    n = len(rows)

    def pct(k: int) -> float:
        return round(100 * k / n, 1) if n else 0.0

    def p(q: float) -> float:
        if not lat:
            return 0.0
        idx = min(len(lat) - 1, max(0, int(round(q * (len(lat) - 1)))))
        return lat[idx]

    node_totals: dict[str, list[int]] = {}
    for r in rows:
        for node, ms in r.nodes.items():
            node_totals.setdefault(node, []).append(ms)
    return {
        "requests": n,
        "avg_latency_s": round(statistics.mean(lat), 2),
        "p50_latency_s": round(p(0.5), 2),
        "p95_latency_s": round(p(0.95), 2),
        "max_latency_s": round(lat[-1], 2),
        "avg_llm_calls": round(statistics.mean(r.llm_calls for r in rows), 2),
        "avg_llm_time_s": round(statistics.mean(r.llm_time_s for r in rows), 2),
        "total_input_tokens": sum(r.input_tokens for r in rows),
        "total_output_tokens": sum(r.output_tokens for r in rows),
        "avg_tokens": round(statistics.mean(r.total_tokens for r in rows)),
        "total_cost_usd": round(sum(r.cost_usd for r in rows), 4),
        "avg_cost_usd": round(statistics.mean(r.cost_usd for r in rows), 5),
        "avg_tool_calls": round(statistics.mean(r.tool_calls for r in rows), 2),
        "verification_pass_pct": (
            round(100 * sum(1 for r in verified_rows if r.verification_ok) / len(verified_rows), 1)
            if verified_rows
            else None
        ),
        "figures_checked": sum(r.figures_checked for r in rows),
        "figures_verified": sum(r.figures_verified for r in rows),
        "revision_pct": pct(sum(1 for r in rows if r.revision)),
        "clarification_pct": pct(sum(1 for r in rows if r.interrupted)),
        "guard_reject_pct": pct(sum(1 for r in rows if r.guard_rejected)),
        "error_pct": pct(sum(1 for r in rows if r.error)),
        "answered": len(answered),
        "node_avg_ms": {k: round(statistics.mean(v)) for k, v in node_totals.items()},
    }
