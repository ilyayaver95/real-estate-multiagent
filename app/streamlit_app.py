"""Streamlit chat UI for the real-estate asset-management multi-agent assistant.

Run locally:  streamlit run app/streamlit_app.py
On Streamlit Community Cloud the OPENAI_API_KEY comes from st.secrets and is mirrored into the
environment before the agent package reads its settings.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Streamlit secrets -> environment (no-op locally where .env is used).
try:
    for key in ("OPENAI_API_KEY", "REM_MODEL"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
except Exception:  # no secrets file locally
    pass

from rem_agent.assistant import Assistant, TurnResult  # noqa: E402
from rem_agent.config import get_settings  # noqa: E402
from rem_agent.llm import LLMUnavailable  # noqa: E402

st.set_page_config(page_title="Asset Manager Assistant", page_icon="🏢", layout="wide")

EXAMPLES = [
    "What is the total P&L for all my properties this year?",
    "How does this quarter compare to the same period last year?",
    "Who are my top tenants, and is anything unusual in the numbers?",
    "What is the price of my asset at 123 Main St compared to the one at 456 Oak Ave?",
    "Show me details for Building 17",
    "Which property performed best last quarter?",
    "How much mortgage interest did we pay in 2024, and what were parking revenues in Q3 2024?",
    "Monthly net trend for Building 140 in 2024 - which month was worst?",
]


@st.cache_resource(show_spinner="Loading data and agents...")
def get_assistant() -> Assistant:
    return Assistant()


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])  # [{"role", "content", "turn": TurnResult | None}]
    ss.setdefault("pending_thread", None)  # thread id of an interrupted (clarifying) turn
    ss.setdefault("queued_question", None)


def history_for_agent() -> list[dict]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]


def render_trace(turn: TurnResult) -> None:
    with st.expander("Agent trace: how this answer was produced", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Graph nodes run", len(turn.trace))
        cols[1].metric("Wall time", f"{turn.elapsed_s:.1f}s")
        if turn.verification:
            v = turn.verification
            label = "n/a" if v.get("skipped") else f"{v['verified']}/{v['checked']}"
            cols[2].metric("Figures verified", label)
        for ev in turn.trace:
            dur = f" · {ev.duration_ms} ms" if ev.duration_ms else ""
            st.markdown(f"**{ev.node}**{dur} — {ev.summary}")
            if ev.node.startswith("specialist") and isinstance(ev.detail, dict):
                for call in ev.detail.get("tool_calls", []):
                    st.code(call, language="text")
            elif ev.node in ("router", "extractor", "resolver", "verifier") and ev.detail:
                with st.popover("details"):
                    st.json(ev.detail, expanded=False)


def render_sidebar(assistant: Assistant) -> bool:
    cat = assistant.catalog
    with st.sidebar:
        st.title("🏢 Asset Manager Assistant")
        st.caption(
            "Multi-agent system orchestrated with LangGraph. Ask about P&L, period comparisons, "
            "property or tenant details, trends and anomalies in the ledger."
        )
        dedupe = st.toggle(
            "Exclude exact duplicate rows",
            value=False,
            help=f"The ledger contains {assistant.ledger.duplicate_rows:,} exact duplicate rows. "
            "Off = compute on the data as delivered (duplicates are flagged in answers). "
            "On = drop them before calculating.",
        )
        st.divider()
        st.subheader("Dataset")
        st.markdown(
            f"**{', '.join(cat.entities)}** · general ledger\n\n"
            f"- {len(assistant.ledger.df):,} ledger lines\n"
            f"- {cat.min_period} → {cat.max_period} (as-of {cat.as_of})\n"
            f"- {len(cat.properties)} properties, {len(cat.tenants)} tenants\n"
            f"- Amounts in EUR"
        )
        with st.expander("Properties"):
            for p in cat.properties:
                tenants = cat.property_tenants.get(p, [])
                st.markdown(f"- **{p}** · {len(tenants)} tenant(s)")
        with st.expander("Not in the data"):
            st.markdown(
                "Prices / valuations, appraisal dates, addresses, floor areas, occupancy, "
                "lease terms, debt balances. The assistant says so instead of guessing."
            )
        st.divider()
        st.caption(f"Model: `{assistant.settings.model}`")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_thread = None
            st.rerun()
    return dedupe


def run_turn(assistant: Assistant, text: str, dedupe: bool) -> None:
    st.session_state.messages.append({"role": "user", "content": text, "turn": None})
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Routing → extracting → resolving → specialists → verifying..."):
            t0 = time.perf_counter()
            try:
                pending = st.session_state.pending_thread
                if pending:
                    turn = assistant.resume(pending, text)
                else:
                    turn = assistant.ask(text, history=history_for_agent()[:-1], dedupe=dedupe)
            except Exception as exc:  # surface, never crash the app
                turn = TurnResult(
                    answer="Something went wrong while processing that: "
                    f"{type(exc).__name__}: {exc}",
                    thread_id="",
                )
            elapsed = time.perf_counter() - t0
        st.session_state.pending_thread = turn.thread_id if turn.interrupted else None
        st.markdown(turn.answer)
        st.caption(
            f"{elapsed:.1f}s" + (" · waiting for your clarification" if turn.interrupted else "")
        )
        if turn.trace:
            render_trace(turn)
    st.session_state.messages.append({"role": "assistant", "content": turn.answer, "turn": turn})


def main() -> None:
    init_state()
    settings = get_settings()
    if not settings.llm_available:
        st.error(
            "OPENAI_API_KEY is not configured. Locally: copy .env.example to .env. "
            "On Streamlit Cloud: add it under App settings → Secrets."
        )
        st.stop()
    try:
        assistant = get_assistant()
    except LLMUnavailable as exc:
        st.error(str(exc))
        st.stop()

    dedupe = render_sidebar(assistant)

    st.markdown("#### Ask about your portfolio")
    if not st.session_state.messages:
        st.caption("Try one of these, or type your own question below.")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.queued_question = ex

    # replay history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("turn") and m["turn"].trace:
                render_trace(m["turn"])

    placeholder = (
        "Answer the clarification question above..."
        if st.session_state.pending_thread
        else "e.g. What is the total P&L for 2024?"
    )
    prompt = st.chat_input(placeholder)
    queued = st.session_state.pop("queued_question", None)
    text = prompt or queued
    if text:
        run_turn(assistant, text, dedupe)
        st.rerun()  # re-render from history so placeholder/state are consistent


main()
