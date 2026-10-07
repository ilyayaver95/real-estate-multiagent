"""Monitoring dashboard: latency, LLM usage and cost, quality and workload KPIs.

Reads the JSONL metrics the Assistant writes for every request (see rem_agent.telemetry).
On Streamlit Community Cloud the file lives on the app container, so it resets on redeploy;
locally it persists in outputs/metrics.jsonl.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rem_agent.telemetry import MetricsStore, summarize  # noqa: E402

ACCENT = "#1f6f8b"  # single hue: magnitudes only, identity comes from axis labels

st.title("📊 Monitoring")
st.caption(
    "Per-request telemetry for the multi-agent assistant: latency, model calls, tokens, cost, "
    "verification and routing. Every question asked in the chat page appears here."
)

store = MetricsStore()
rows = store.load()

top = st.columns([1, 1, 4])
if top[0].button("Refresh"):
    st.rerun()
if top[1].button("Clear metrics"):
    store.clear()
    st.rerun()

if not rows:
    st.info("No requests recorded yet. Ask a few questions on the chat page, then come back.")
    st.stop()

s = summarize(rows)
df = pd.DataFrame([r.__dict__ for r in rows])
df["time"] = pd.to_datetime(df["ts"], unit="s")
df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
df["request"] = range(1, len(df) + 1)

# ---- Headline KPIs -----------------------------------------------------------------------------
st.subheader("Headline")
c = st.columns(6)
c[0].metric("Requests", s["requests"])
c[1].metric("Avg latency", f"{s['avg_latency_s']:.1f}s", help="Wall time per request")
c[2].metric("p95 latency", f"{s['p95_latency_s']:.1f}s")
c[3].metric("Avg LLM calls", f"{s['avg_llm_calls']:.1f}")
c[4].metric("Avg tokens", f"{s['avg_tokens']:,}")
c[5].metric(
    "Total cost", f"${s['total_cost_usd']:.4f}", help="Estimated from list prices per model"
)

c = st.columns(6)
vp = s["verification_pass_pct"]
c[0].metric(
    "Figures verified",
    f"{s['figures_verified']}/{s['figures_checked']}",
    help="Numbers in answers traced back to tool outputs",
)
c[1].metric(
    "Verification pass",
    "n/a" if vp is None else f"{vp:.0f}%",
    help="Share of answers with tool-computed figures where every figure was verified",
)
c[2].metric(
    "Revision rate", f"{s['revision_pct']:.0f}%", help="Answers rewritten once by the verifier loop"
)
c[3].metric("Clarification rate", f"{s['clarification_pct']:.0f}%")
c[4].metric("Guard rejections", f"{s['guard_reject_pct']:.0f}%")
c[5].metric("Error rate", f"{s['error_pct']:.0f}%", help="Specialist exceptions or model errors")

st.divider()

# ---- Latency -----------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Latency per request (s)")
    st.line_chart(
        df.set_index("request")[["elapsed_s"]].rename(columns={"elapsed_s": "wall time"}),
        color=ACCENT,
        height=260,
    )
    st.caption(
        f"median {s['p50_latency_s']:.1f}s · p95 {s['p95_latency_s']:.1f}s · "
        f"max {s['max_latency_s']:.1f}s · "
        f"model time {s['avg_llm_time_s']:.1f}s of {s['avg_latency_s']:.1f}s on average"
    )
with right:
    st.subheader("Average time per graph node (ms)")
    node_df = pd.Series(s["node_avg_ms"]).sort_values(ascending=False).rename("avg ms").to_frame()
    st.bar_chart(node_df, color=ACCENT, height=260, horizontal=True)

# ---- LLM usage ---------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Tokens per request")
    tok = df.set_index("request")[["input_tokens", "output_tokens"]]
    st.bar_chart(tok, color=[ACCENT, "#9ec9d8"], height=260, stack=True)
    st.caption(
        f"total {s['total_input_tokens']:,} in / {s['total_output_tokens']:,} out · "
        f"avg cost ${s['avg_cost_usd']:.5f} per request"
    )
with right:
    st.subheader("Cumulative cost (USD)")
    cost = df.set_index("request")[["cost_usd"]].cumsum().rename(columns={"cost_usd": "cumulative"})
    st.line_chart(cost, color=ACCENT, height=260)

# ---- Workload ----------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Intent mix")
    intents = df["intents"].explode().dropna().value_counts().rename("requests").to_frame()
    if len(intents):
        st.bar_chart(intents, color=ACCENT, height=260, horizontal=True)
with right:
    st.subheader("Specialists used")
    specs = df["specialists"].explode().dropna().value_counts().rename("runs").to_frame()
    if len(specs):
        st.bar_chart(specs, color=ACCENT, height=260, horizontal=True)
    st.caption(f"avg {s['avg_tool_calls']:.1f} tool calls per request")

st.divider()

# ---- Recent requests ---------------------------------------------------------------------------
st.subheader("Recent requests")
show = df.sort_values("ts", ascending=False)[
    [
        "time",
        "question",
        "intents",
        "elapsed_s",
        "llm_calls",
        "total_tokens",
        "cost_usd",
        "tool_calls",
        "figures_verified",
        "figures_checked",
        "verification_ok",
        "interrupted",
        "revision",
        "guard_rejected",
        "error",
        "dedupe",
        "model",
    ]
].copy()
show["verified"] = show.apply(
    lambda r: (
        "n/a"
        if r["verification_ok"] is None or pd.isna(r["verification_ok"])
        else f"{int(r['figures_verified'])}/{int(r['figures_checked'])}"
    ),
    axis=1,
)
show["intents"] = show["intents"].apply(lambda v: ", ".join(v) if isinstance(v, list) else v)
show = show.drop(columns=["figures_verified", "figures_checked", "verification_ok"])
st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss"),
        "question": st.column_config.TextColumn("Question", width="large"),
        "elapsed_s": st.column_config.NumberColumn("Latency (s)", format="%.1f"),
        "llm_calls": "LLM calls",
        "total_tokens": st.column_config.NumberColumn("Tokens", format="%d"),
        "cost_usd": st.column_config.NumberColumn("Cost ($)", format="%.5f"),
        "tool_calls": "Tools",
        "interrupted": "Clarified",
        "guard_rejected": "Guard",
    },
)
with st.expander("Raw data"):
    st.dataframe(df.drop(columns=["nodes"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download metrics.jsonl",
        data=store.path.read_bytes(),
        file_name="metrics.jsonl",
        mime="application/json",
    )
st.caption(
    f"Store: `{store.path}` · last updated {time.strftime('%H:%M:%S', time.localtime(rows[-1].ts))}"
)
