"""Time-period helpers.

The ledger is monthly, so every timeframe a user mentions ("2024", "Q1 2025", "last quarter",
"this year", "June 2024") is normalised to an inclusive ``PeriodRange`` of months. Relative
phrases are anchored to the dataset's latest month (``as_of``) rather than the wall clock,
because the data ends in 2025-03 and a literal "this year" would otherwise be empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})


@dataclass(frozen=True)
class PeriodRange:
    start: pd.Period
    end: pd.Period
    label: str

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} after end {self.end}")

    @property
    def months(self) -> int:
        return (self.end - self.start).n + 1

    def contains(self, p: pd.Period) -> bool:
        return self.start <= p <= self.end

    def shift_years(self, years: int) -> PeriodRange:
        s, e = self.start + 12 * years, self.end + 12 * years
        return PeriodRange(s, e, natural_label(s, e))

    def previous(self) -> PeriodRange:
        """The immediately preceding range of equal length."""
        n = self.months
        s, e = self.start - n, self.end - n
        return PeriodRange(s, e, natural_label(s, e))

    def clip(self, lo: pd.Period, hi: pd.Period) -> tuple[PeriodRange | None, bool]:
        """Clip to coverage. Returns (clipped or None if disjoint, was_clipped)."""
        s, e = max(self.start, lo), min(self.end, hi)
        if s > e:
            return None, True
        clipped = s != self.start or e != self.end
        return PeriodRange(s, e, self.label), clipped

    def __str__(self) -> str:
        if self.start == self.end:
            return str(self.start)
        return f"{self.start} to {self.end}"


def natural_label(start: pd.Period, end: pd.Period) -> str:
    """'2024', '2024-Q3', '2024-06' or 'YYYY-MM to YYYY-MM' depending on alignment."""
    if start == end:
        return str(start)
    if start.year == end.year and start.month == 1 and end.month == 12:
        return str(start.year)
    if (end - start).n == 2 and start.month in (1, 4, 7, 10) and start.year == end.year:
        return f"{start.year}-Q{start.quarter}"
    return f"{start} to {end}"


def year_range(year: int) -> PeriodRange:
    return PeriodRange(pd.Period(f"{year}-01", "M"), pd.Period(f"{year}-12", "M"), str(year))


def quarter_range(year: int, q: int) -> PeriodRange:
    if q not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1-4, got {q}")
    start = pd.Period(f"{year}-{3 * (q - 1) + 1:02d}", "M")
    return PeriodRange(start, start + 2, f"{year}-Q{q}")


def month_range(year: int, m: int) -> PeriodRange:
    p = pd.Period(f"{year}-{m:02d}", "M")
    return PeriodRange(p, p, str(p))


def range_from_periods(start: pd.Period, end: pd.Period, label: str | None = None) -> PeriodRange:
    return PeriodRange(start, end, label or f"{start} to {end}")


def ytd_range(as_of: pd.Period) -> PeriodRange:
    return PeriodRange(pd.Period(f"{as_of.year}-01", "M"), as_of, f"{as_of.year} YTD")


def current_quarter_range(as_of: pd.Period) -> PeriodRange:
    return quarter_range(as_of.year, as_of.quarter)


def last_n_months(as_of: pd.Period, n: int) -> PeriodRange:
    return PeriodRange(as_of - (n - 1), as_of, f"last {n} months")


def parse_period_text(text: str, as_of: pd.Period) -> PeriodRange | None:
    """Best-effort parse of a short timeframe string into a PeriodRange.

    Handles: '2024', 'FY2024', 'Q1 2025', '2025-Q1', '2025Q1', '2024-M06', 'June 2024',
    'Jun 2024', '2024-06', 'this year', 'last year', 'this quarter', 'last quarter',
    'last 6 months', 'ytd', 'all'. Returns None if nothing matched.
    """
    t = text.strip().lower()
    if not t:
        return None
    if t in {"all", "all time", "entire period", "whole period", "everything", "overall"}:
        return None  # caller treats None as "no time filter"
    if t in {"this year", "current year", "ytd", "year to date"}:
        return ytd_range(as_of)
    if t in {"last year", "previous year", "prior year"}:
        return year_range(as_of.year - 1)
    if t in {"this quarter", "current quarter", "qtd", "latest quarter", "most recent quarter"}:
        return current_quarter_range(as_of)
    if t in {"last quarter", "previous quarter", "prior quarter"}:
        return current_quarter_range(as_of).previous()
    if t in {"this month", "current month", "latest month", "most recent month"}:
        return month_range(as_of.year, as_of.month)
    if t in {"last month", "previous month"}:
        prev = as_of - 1
        return month_range(prev.year, prev.month)
    m = re.fullmatch(r"(?:last|past|previous|trailing)\s+(\d{1,2})\s+months?", t)
    if m:
        return last_n_months(as_of, int(m.group(1)))
    m = re.fullmatch(r"(?:fy\s?)?(\d{4})", t)
    if m:
        return year_range(int(m.group(1)))
    m = re.fullmatch(r"q([1-4])\s*[-/ ]?\s*(\d{4})", t) or re.fullmatch(
        r"(\d{4})\s*[-/ ]?\s*q([1-4])", t
    )
    if m:
        a, b = m.groups()
        year, q = (int(b), int(a)) if len(a) == 1 else (int(a), int(b))
        return quarter_range(year, q)
    m = re.fullmatch(r"(\d{4})-m?(\d{1,2})", t)
    if m:
        return month_range(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"([a-z]+)\.?\s+(\d{4})", t) or re.fullmatch(r"(\d{4})\s+([a-z]+)\.?", t)
    if m:
        a, b = m.groups()
        mon, year = (a, b) if a.isalpha() else (b, a)
        if mon in _MONTHS:
            return month_range(int(year), _MONTHS[mon])
    return None
