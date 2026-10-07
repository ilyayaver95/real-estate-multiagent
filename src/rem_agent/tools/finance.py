"""P&L style calculations.

Conventions
-----------
* ``profit`` is signed: revenue lines are positive, expense lines negative. "Expenses" in the
  outputs are reported as a positive magnitude for readability, with ``net`` = sum(profit).
* Every result carries ``caveats`` so the LLM layer can surface data limitations honestly
  (partial coverage, unallocated expenses, duplicates).
"""

from __future__ import annotations

import pandas as pd

from rem_agent.data.loader import LedgerData
from rem_agent.tools.filters import LedgerFilter, apply_filter
from rem_agent.tools.formatting import r2, r4
from rem_agent.tools.periods import PeriodRange, natural_label


class FinanceTools:
    def __init__(self, ledger: LedgerData, dedupe: bool = False):
        self.ledger = ledger
        self.dedupe = dedupe
        self.df = ledger.view(dedupe=dedupe)

    # ---- helpers ----------------------------------------------------------------------------
    def _coverage_caveats(self, f: LedgerFilter, sub: pd.DataFrame) -> list[str]:
        caveats: list[str] = []
        lo, hi = self.ledger.min_period, self.ledger.max_period
        if f.period is not None:
            clipped, was_clipped = f.period.clip(lo, hi)
            if clipped is None:
                caveats.append(f"No data for {f.period}: the ledger covers {lo} to {hi} only.")
            elif was_clipped:
                caveats.append(
                    f"Requested {f.period} but the ledger only covers {clipped} "
                    f"({clipped.months} of {f.period.months} months). Figures are partial."
                )
        if f.properties or f.tenants:
            unalloc = self.df[
                self.df["property_name"].isna() & (self.df["ledger_type"] == "expenses")
            ]
            if f.period is not None:
                unalloc = apply_filter(unalloc, LedgerFilter(period=f.period))
            if len(unalloc):
                caveats.append(
                    "Most expenses (interest, management fees, taxes, insurance) are booked at "
                    f"entity level without a property ({r2(-unalloc['profit'].sum()):,.2f} EUR "
                    "in this period) and are NOT included in property/tenant-level P&L."
                )
        if not self.dedupe:
            dups = int(sub["is_duplicate"].sum()) if "is_duplicate" in sub else 0
            if dups:
                caveats.append(
                    f"{dups} exact duplicate ledger rows are included in these figures "
                    "(dataset quality issue; toggle de-duplication to exclude them)."
                )
        if len(sub) == 0:
            caveats.append("No ledger rows matched the requested filters.")
        return caveats

    @staticmethod
    def _split(sub: pd.DataFrame) -> dict:
        rev = sub.loc[sub["ledger_type"] == "revenue", "profit"].sum()
        exp = sub.loc[sub["ledger_type"] == "expenses", "profit"].sum()
        return {
            "revenue": r2(rev),
            "expenses": r2(-exp),  # positive magnitude
            "net": r2(sub["profit"].sum()),
            "rows": int(len(sub)),
        }

    # ---- public tools -----------------------------------------------------------------------
    def pnl(self, f: LedgerFilter, breakdown_by: str | None = "ledger_group") -> dict:
        """Revenue, expenses and net for a filter, with an optional breakdown."""
        sub = apply_filter(self.df, f)
        split = self._split(sub)
        caveats = self._coverage_caveats(f, sub)
        if set(f.ledger_types) == {"revenue"}:
            split["expenses"] = None
            split["net"] = None
            caveats.insert(0, "Revenue-only view: expenses and net are not computed here.")
        elif set(f.ledger_types) == {"expenses"}:
            split["revenue"] = None
            split["net"] = None
            caveats.insert(0, "Expenses-only view: revenue and net are not computed here.")
        out = {
            "filters": f.describe(),
            "coverage": f"{self.ledger.min_period} to {self.ledger.max_period}",
            **split,
            "caveats": caveats,
        }
        if f.ledger_groups or f.ledger_categories:
            # "How much / what share is parking?" needs the unfiltered scope as denominator.
            scope = LedgerFilter(
                properties=f.properties,
                tenants=f.tenants,
                period=f.period,
                ledger_types=f.ledger_types,
            )
            scope_split = self._split(apply_filter(self.df, scope))
            out["scope_totals"] = {
                "description": "Same properties/tenants/period without the account filter",
                "revenue": scope_split["revenue"],
                "expenses": scope_split["expenses"],
            }
            rev_sel = split["revenue"] or 0.0
            exp_sel = split["expenses"] or 0.0
            out["share_of_scope"] = {
                "revenue": (
                    r4(rev_sel / scope_split["revenue"]) if scope_split["revenue"] else None
                ),
                "expenses": (
                    r4(exp_sel / scope_split["expenses"]) if scope_split["expenses"] else None
                ),
            }
        if breakdown_by and len(sub):
            col = _resolve_dim(breakdown_by)
            grp = (
                sub.groupby(col, dropna=False)["profit"].sum().sort_values(key=abs, ascending=False)
            )
            out["breakdown_by"] = breakdown_by
            out["breakdown"] = [
                {"name": _label(k), "amount": r2(v)} for k, v in grp.head(12).items()
            ]
        return out

    def compare_periods(
        self, f: LedgerFilter, period_a: PeriodRange, period_b: PeriodRange
    ) -> dict:
        """P&L for two periods plus absolute and relative deltas (a vs b)."""
        a = self.pnl(f.with_period(period_a), breakdown_by=None)
        b = self.pnl(f.with_period(period_b), breakdown_by=None)
        deltas = {}
        for key in ("revenue", "expenses", "net"):
            if a[key] is None or b[key] is None:
                continue
            da = a[key] - b[key]
            deltas[key] = {
                "absolute": r2(da),
                "relative": (r4(da / abs(b[key])) if b[key] else None),
            }
        # Which categories moved most (helps "what drove the change" follow-ups)
        sa = (
            apply_filter(self.df, f.with_period(period_a))
            .groupby("ledger_category")["profit"]
            .sum()
        )
        sb = (
            apply_filter(self.df, f.with_period(period_b))
            .groupby("ledger_category")["profit"]
            .sum()
        )
        movers = (sa.sub(sb, fill_value=0)).sort_values(key=abs, ascending=False).head(6)
        out = {
            "filters": f.describe(),
            "period_a": {"label": period_a.label, "range": str(period_a), **_strip(a)},
            "period_b": {"label": period_b.label, "range": str(period_b), **_strip(b)},
            "delta_a_minus_b": deltas,
            "largest_category_moves": [{"category": k, "change": r2(v)} for k, v in movers.items()],
            "caveats": _merge_caveats(a, b, period_a, period_b),
        }
        lfl = self._like_for_like(f, period_a, period_b)
        if lfl:
            out["like_for_like"] = lfl
            # Percent changes between a full and a partial period are meaningless; drop them.
            for d in out["delta_a_minus_b"].values():
                d["relative"] = None
            out["caveats"].append(
                "The two periods are not equally covered by the data, so full-period percentage "
                "changes are omitted; 'like_for_like' compares only the months both periods have. "
                "Use it for growth rates."
            )
        return out

    def _like_for_like(self, f: LedgerFilter, pa: PeriodRange, pb: PeriodRange) -> dict | None:
        """When one period is clipped by coverage (e.g. 2025 = Jan-Mar), compare the same
        calendar months of both periods."""
        lo, hi = self.ledger.min_period, self.ledger.max_period
        ca, clipped_a = pa.clip(lo, hi)
        cb, clipped_b = pb.clip(lo, hi)
        if ca is None or cb is None or not (clipped_a or clipped_b):
            return None
        if pa.months != pb.months:
            return None
        # months-of-year available in both clipped ranges
        months_a = {(p.month) for p in pd.period_range(ca.start, ca.end, freq="M")}
        months_b = {(p.month) for p in pd.period_range(cb.start, cb.end, freq="M")}
        common = sorted(months_a & months_b)
        if not common or len(common) == pa.months:
            return None
        a_start = pd.Period(f"{ca.start.year}-{common[0]:02d}", "M")
        a_end = pd.Period(f"{ca.start.year}-{common[-1]:02d}", "M")
        b_start = pd.Period(f"{cb.start.year}-{common[0]:02d}", "M")
        b_end = pd.Period(f"{cb.start.year}-{common[-1]:02d}", "M")
        ra = PeriodRange(a_start, a_end, natural_label(a_start, a_end))
        rb = PeriodRange(b_start, b_end, natural_label(b_start, b_end))
        a = self.pnl(f.with_period(ra), breakdown_by=None)
        b = self.pnl(f.with_period(rb), breakdown_by=None)
        deltas = {}
        for key in ("revenue", "expenses", "net"):
            if a[key] is None or b[key] is None:
                continue
            da = a[key] - b[key]
            deltas[key] = {"absolute": r2(da), "relative": r4(da / abs(b[key])) if b[key] else None}
        return {
            "months_compared": len(common),
            "period_a": {"label": ra.label, "range": str(ra), **_strip(a)},
            "period_b": {"label": rb.label, "range": str(rb), **_strip(b)},
            "delta_a_minus_b": deltas,
        }

    def trend(self, f: LedgerFilter, granularity: str = "month") -> dict:
        """Net/revenue/expenses time series by month or quarter."""
        sub = apply_filter(self.df, f)
        if granularity not in ("month", "quarter", "year"):
            raise ValueError("granularity must be month, quarter or year")
        key = {"month": "period", "quarter": "quarter", "year": "year_num"}[granularity]
        rows = []
        for k, g in sub.groupby(key, sort=True):
            rows.append(
                {"period": str(k), **{kk: vv for kk, vv in self._split(g).items() if kk != "rows"}}
            )
        nets = [r["net"] for r in rows]
        summary = {}
        if nets:
            best = max(rows, key=lambda r: r["net"])
            worst = min(rows, key=lambda r: r["net"])
            summary = {
                "best": {"period": best["period"], "net": best["net"]},
                "worst": {"period": worst["period"], "net": worst["net"]},
                "average_net": r2(sum(nets) / len(nets)),
            }
        return {
            "filters": f.describe(),
            "granularity": granularity,
            "series": rows,
            "summary": summary,
            "caveats": self._coverage_caveats(f, sub),
        }

    def breakdown(self, f: LedgerFilter, by: str, top: int = 10) -> dict:
        """Sum of profit grouped by a dimension (property, tenant, ledger_group, ...)."""
        sub = apply_filter(self.df, f)
        col = _resolve_dim(by)
        grp = sub.groupby(col, dropna=False)["profit"].sum()
        total = grp.sum()
        ordered = grp.sort_values(key=abs, ascending=False)
        items = [
            {
                "name": _label(k),
                "amount": r2(v),
                "share": (r4(v / total) if total else None),
            }
            for k, v in ordered.head(top).items()
        ]
        return {
            "filters": f.describe(),
            "by": by,
            "total": r2(total),
            "items": items,
            "other_count": int(max(len(grp) - top, 0)),
            "caveats": self._coverage_caveats(f, sub),
        }


_DIMS = {
    "property": "property_name",
    "properties": "property_name",
    "property_name": "property_name",
    "tenant": "tenant_name",
    "tenants": "tenant_name",
    "tenant_name": "tenant_name",
    "ledger_type": "ledger_type",
    "type": "ledger_type",
    "ledger_group": "ledger_group",
    "group": "ledger_group",
    "ledger_category": "ledger_category",
    "category": "ledger_category",
    "month": "period",
    "quarter": "quarter",
    "year": "year_num",
}


def _resolve_dim(name: str) -> str:
    try:
        return _DIMS[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dimension '{name}'. Use one of {sorted(set(_DIMS))}") from exc


def _label(key) -> str:
    if key is None or (isinstance(key, float) and pd.isna(key)) or key is pd.NA:
        return "(unallocated / entity level)"
    return str(key)


def _merge_caveats(a: dict, b: dict, pa: PeriodRange, pb: PeriodRange) -> list[str]:
    """One combined duplicate-row caveat instead of one per period; keep the rest unique."""
    out: list[str] = []
    dup_counts = []
    for res, per in ((a, pa), (b, pb)):
        for c in res["caveats"]:
            if "exact duplicate ledger rows" in c:
                dup_counts.append(f"{c.split()[0]} in {per.label}")
            elif c not in out:
                out.append(c)
    if dup_counts:
        out.append(
            "Exact duplicate ledger rows are included in these figures ("
            + ", ".join(dup_counts)
            + "); toggle de-duplication to exclude them."
        )
    return out


def _strip(pnl: dict) -> dict:
    return {k: pnl[k] for k in ("revenue", "expenses", "net", "rows")}
