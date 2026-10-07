"""Verifier: checks that the figures in the final answer trace back to tool outputs.

Deterministic. Every money/percentage-looking number in the answer is matched against the set of
numbers the tools returned (with tolerance for rounding and for % vs ratio). Years, small counts
and ids are ignored. The result is attached to the trace; if unverified figures remain the graph
asks the synthesizer for one revision.
"""

from __future__ import annotations

import re

from rem_agent.schemas import SpecialistResult

_NUM = re.compile(
    r"(?<![\w.])([€$]?)\s?(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s?(%|(?:k|K|M|m)(?![A-Za-z]))?"
)


def extract_numbers(text: str) -> list[tuple[float, str]]:
    out = []
    for m in _NUM.finditer(text):
        cur, num, suffix = m.group(1), m.group(2), m.group(3) or ""
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        if suffix in ("k", "K"):
            val *= 1_000
        elif suffix == "M":
            val *= 1_000_000
        out.append((val, "%" if suffix == "%" else ("€" if cur else "")))
    return out


def _is_ignorable(val: float, kind: str, raw_has_decimal: bool) -> bool:
    if kind in ("%", "€"):
        return False
    if 1900 <= val <= 2100 and float(val).is_integer():  # years
        return True
    if abs(val) <= 100 and float(val).is_integer():  # counts, "top 5", quarter numbers
        return True
    return False


def verify(
    answer: str,
    results: list[SpecialistResult],
    tolerance: float = 0.006,
    ignore_names: list[str] | None = None,
) -> dict:
    text = answer
    for name in ignore_names or []:  # "Building 120" must not be read as the number 120
        text = text.replace(name, " ")
    pool = [n for r in results for n in r.numbers_used]
    checked, unverified, derived = [], [], []
    for val, kind in extract_numbers(text):
        if _is_ignorable(val, kind, "." in str(val)):
            continue
        if kind == "%":
            ok = _match(val / 100, pool, tolerance) or _match(val, pool, tolerance)
            if not ok and _match_derived_ratio(val / 100, pool, tolerance):
                ok = True
                derived.append(val)
        else:
            ok = _match(val, pool, tolerance)
            if not ok and _match_derived_diff(val, pool, tolerance):
                ok = True
                derived.append(val)
        checked.append(val)
        if not ok:
            unverified.append(val)
    return {
        "checked": len(checked),
        "verified": len(checked) - len(unverified),
        "derived": derived[:10],
        "unverified": unverified[:10],
        "ok": not unverified,
        "pool_size": len(pool),
    }


def _match_derived_diff(val: float, pool: list[float], tol: float) -> bool:
    """Accept a correct difference or sum of two tool numbers (the LLM did the arithmetic)."""
    uniq = sorted(set(pool))
    if len(uniq) > 400:
        return False
    for i, a in enumerate(uniq):
        for b in uniq[i + 1 :]:
            for cand in (a - b, b - a, a + b):
                if abs(cand - val) <= tol * max(abs(cand), abs(val), 1.0):
                    return True
    return False


def _match_derived_ratio(ratio: float, pool: list[float], tol: float) -> bool:
    """Accept a correct percentage change or share computed from two tool numbers."""
    uniq = sorted(set(abs(p) for p in pool if p))
    if len(uniq) > 400:
        return False
    for a in uniq:
        for b in uniq:
            if a == b:
                continue
            for cand in (a / b, a / b - 1):
                if abs(cand - ratio) <= max(tol * abs(cand), 0.0006):
                    return True
    return False


def _match(val: float, pool: list[float], tol: float) -> bool:
    for p in pool:
        if p == val:
            return True
        scale = max(abs(p), abs(val), 1.0)
        if abs(p - val) <= tol * scale:
            return True
        # answer rounded to thousands/millions ("€1.17M", "€592k")
        for unit in (1_000, 1_000_000):
            if abs(round(p / unit, 2) * unit - val) <= 0.005 * unit + 1e-6:
                return True
    return False
