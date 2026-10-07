"""Number formatting shared by tools and the UI."""

from __future__ import annotations

import math

CURRENCY = "EUR"


def money(x: float | None, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}€{abs(x):,.{decimals}f}"


def pct(x: float | None, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x * 100:+.{decimals}f}%"


def r2(x: float) -> float:
    """Round for JSON payloads without turning -0.0 into an eyesore."""
    v = round(float(x), 2)
    return 0.0 if v == 0 else v


def r4(x: float | None) -> float | None:
    """Ratios and shares keep four decimals (0.0943 = 9.43%)."""
    if x is None:
        return None
    v = round(float(x), 4)
    return 0.0 if v == 0 else v
