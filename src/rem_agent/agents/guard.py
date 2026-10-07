"""Input guard: deterministic sanity checks that run before any LLM call.

Catches the cheap failure modes (empty input, pasted binary/JSON blobs, absurd length,
no letters at all) with an instant, friendly message instead of spending tokens on them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

MAX_CHARS = 2000
MIN_LETTERS = 2


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    cleaned: str
    message: str | None = None
    reason: str | None = None


def check_input(text: str | None) -> GuardResult:
    if text is None or not isinstance(text, str):
        return GuardResult(False, "", "I need a text question to work with.", "non_text")
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return GuardResult(
            False,
            "",
            "Your message is empty. Ask me about P&L, properties, tenants or "
            "anomalies in the ledger.",
            "empty",
        )
    if len(cleaned) > MAX_CHARS:
        return GuardResult(
            False,
            cleaned[:MAX_CHARS],
            f"That message is {len(cleaned):,} characters long. Please keep questions under "
            f"{MAX_CHARS:,} characters - one or two questions at a time works best.",
            "too_long",
        )
    letters = sum(ch.isalpha() for ch in cleaned)
    if letters < MIN_LETTERS:
        return GuardResult(
            False,
            cleaned,
            "I couldn't find a question in that. Try something like 'Who are my top tenants?'",
            "no_letters",
        )
    if _looks_like_structured_blob(text):  # raw text: line structure matters for CSV detection
        return GuardResult(
            False,
            cleaned,
            "That looks like raw data or code rather than a question. I work from the ledger "
            "dataset already loaded; ask me a question about it in plain language, e.g. "
            "'What was the total P&L in 2024?'",
            "structured_blob",
        )
    return GuardResult(True, cleaned)


def _looks_like_structured_blob(s: str) -> bool:
    stripped = s.strip()
    if stripped[:1] in "{[" and stripped[-1:] in "}]":
        try:
            json.loads(stripped)
            return True
        except ValueError:
            pass
    # CSV-ish: several lines with the same comma count, or many semicolons/pipes
    lines = [ln for ln in s.split("\n") if ln.strip()]
    if len(lines) >= 3:
        counts = {ln.count(",") for ln in lines}
        if len(counts) == 1 and counts.pop() >= 3:
            return True
    symbol_ratio = sum(not (ch.isalnum() or ch.isspace()) for ch in s) / max(len(s), 1)
    return symbol_ratio > 0.4
