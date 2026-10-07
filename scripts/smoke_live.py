"""Quick live smoke test: run a few questions through the full graph and print traces."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

from rem_agent.assistant import Assistant  # noqa: E402

QUESTIONS = sys.argv[1:] or [
    "What is the price of my asset at 123 Main St compared to the one at 456 Oak Ave?",
    "What is the total P&L for all my properties this year?",
    "How does this quarter compare to the same period last year?",
    "Who are my top tenants, and is anything unusual in the numbers?",
]

a = Assistant()
for q in QUESTIONS:
    t0 = time.perf_counter()
    res = a.ask(q)
    dt = time.perf_counter() - t0
    print("=" * 100)
    print("Q:", q)
    print(f"[{dt:.1f}s] interrupted={res.interrupted} verification={res.verification}")
    for ev in res.trace:
        print(f"  - {ev.node:22s} {ev.duration_ms or 0:6d}ms  {ev.summary}")
        if ev.node.startswith("specialist") and isinstance(ev.detail, dict):
            for c in ev.detail.get("tool_calls", []):
                print(f"        tool: {c}")
    print("-" * 100)
    print(res.answer)
