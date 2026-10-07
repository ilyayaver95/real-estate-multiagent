"""Data-quality and anomaly detection ("is anything unusual in the numbers?").

Each check returns a Finding with a severity, a one-line description, the financial impact
where meaningful, and a few example rows so the answer can be concrete. All checks are pure
pandas and run in well under a second on this dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from rem_agent.data.loader import REQUIRED_COLUMNS, LedgerData
from rem_agent.tools.filters import LedgerFilter, apply_filter
from rem_agent.tools.formatting import r2

EXAMPLE_COLS = [
    "property_name",
    "tenant_name",
    "ledger_category",
    "ledger_code",
    "month",
    "profit",
]


@dataclass
class Finding:
    check: str
    severity: str  # info | warning | high
    title: str
    detail: str
    count: int = 0
    amount: float | None = None
    examples: list[dict] = field(default_factory=list)


class AuditTools:
    def __init__(self, ledger: LedgerData, dedupe: bool = False):
        self.ledger = ledger
        self.dedupe = dedupe
        self.df = ledger.view(dedupe=dedupe)

    def run_all(self, f: LedgerFilter | None = None, max_examples: int = 3) -> dict:
        f = f or LedgerFilter()
        sub = apply_filter(self.df, f)
        findings: list[Finding] = []
        for check in (
            self.duplicate_rows,
            self.double_mapped_codes,
            self.reversal_pairs,
            self.sign_anomalies,
            self.zero_amount_rows,
            self.monthly_outliers,
            self.tenant_gaps,
            self.unallocated_expenses,
            self.partial_period_coverage,
        ):
            try:
                findings.extend(check(sub, max_examples))
            except Exception as exc:  # a failing check must not kill the whole audit
                findings.append(Finding(check.__name__, "info", "check failed", str(exc)))
        order = {"high": 0, "warning": 1, "info": 2}
        findings.sort(key=lambda x: (order[x.severity], -(x.count or 0)))
        return {
            "filters": f.describe(),
            "rows_examined": int(len(sub)),
            "dedupe_applied": self.dedupe,
            "findings": [asdict(x) for x in findings],
            "summary": {
                "high": sum(1 for x in findings if x.severity == "high"),
                "warning": sum(1 for x in findings if x.severity == "warning"),
                "info": sum(1 for x in findings if x.severity == "info"),
            },
        }

    # ---- individual checks -----------------------------------------------------------------
    def duplicate_rows(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        dups = sub[sub.duplicated(subset=sorted(REQUIRED_COLUMNS), keep="first")]
        if dups.empty:
            return []
        return [
            Finding(
                "duplicate_rows",
                "high",
                f"{len(dups)} exact duplicate ledger rows",
                "Identical rows (same property, tenant, account, month and amount) appear more "
                "than once. If these are load errors rather than genuine repeated postings, "
                f"net P&L is overstated by {r2(dups['profit'].sum()):,.2f} EUR.",
                count=int(len(dups)),
                amount=r2(dups["profit"].sum()),
                examples=_examples(dups, k),
            )
        ]

    def double_mapped_codes(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        m = sub.groupby("ledger_code")["ledger_category"].nunique()
        codes = m[m > 1].index.tolist()
        out = []
        for code in codes:
            rows = sub[sub["ledger_code"] == code]
            cats = sorted(rows["ledger_category"].dropna().unique())
            out.append(
                Finding(
                    "double_mapped_code",
                    "high",
                    f"Ledger code {code} is mapped to {len(cats)} categories",
                    f"Code {code} ('{rows['ledger_description'].iloc[0]}') appears under "
                    f"{', '.join(cats)}. The rows are otherwise identical, which looks like the "
                    "same postings counted once per category (double counting of "
                    f"{r2(rows['profit'].sum() / len(cats)):,.2f} EUR).",
                    count=int(len(rows)),
                    amount=r2(rows["profit"].sum() / len(cats)),
                    examples=_examples(rows, k),
                )
            )
        return out

    def reversal_pairs(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        """Same key, same month, amounts that cancel exactly (x and -x)."""
        keys = ["property_name", "tenant_name", "ledger_code", "month"]
        s = sub[sub["profit"] != 0].copy()
        s["abs"] = s["profit"].abs()
        g = s.groupby(keys + ["abs"], dropna=False)["profit"].agg(["count", "sum", "max"])
        pairs = g[(g["count"] >= 2) & (g["sum"].abs() < 0.005)]
        if pairs.empty:
            return []
        big = pairs.sort_values("max", ascending=False).head(k).reset_index()
        return [
            Finding(
                "reversal_pairs",
                "warning",
                f"{len(pairs)} groups of postings that cancel out (booking and reversal)",
                "Amounts booked and reversed in the same month net to zero. They do not change "
                "totals but inflate gross revenue/expense figures and row counts; the largest "
                f"is {r2(big['max'].iloc[0]):,.2f} EUR.",
                count=int(len(pairs)),
                amount=r2(pairs["max"].sum()),
                examples=big[keys + ["max"]].astype(str).to_dict(orient="records"),
            )
        ]

    def sign_anomalies(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        out = []
        pos_exp = sub[(sub["ledger_type"] == "expenses") & (sub["profit"] > 0)]
        if len(pos_exp):
            out.append(
                Finding(
                    "positive_expenses",
                    "warning",
                    f"{len(pos_exp)} expense rows with a positive amount",
                    "Expenses are normally negative. Positive values are usually credit notes or "
                    "reversals; they reduce reported expenses by "
                    f"{r2(pos_exp['profit'].sum()):,.2f} EUR.",
                    count=int(len(pos_exp)),
                    amount=r2(pos_exp["profit"].sum()),
                    examples=_examples(pos_exp, k),
                )
            )
        neg_rev = sub[
            (sub["ledger_type"] == "revenue")
            & (sub["profit"] < 0)
            & (sub["ledger_group"] != "sales_discounts")
        ]
        if len(neg_rev):
            out.append(
                Finding(
                    "negative_revenue",
                    "warning",
                    f"{len(neg_rev)} non-discount revenue rows with a negative amount",
                    "Negative rent/parking revenue outside the discount accounts indicates "
                    f"reversals or corrections totalling {r2(neg_rev['profit'].sum()):,.2f} EUR.",
                    count=int(len(neg_rev)),
                    amount=r2(neg_rev["profit"].sum()),
                    examples=_examples(neg_rev, k),
                )
            )
        return out

    def zero_amount_rows(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        z = sub[sub["profit"] == 0]
        if z.empty:
            return []
        return [
            Finding(
                "zero_amount_rows",
                "info",
                f"{len(z)} rows with a zero amount",
                "Zero lines carry no financial information; they are harmless for totals but "
                "noisy for row counts and 'active month' statistics.",
                count=int(len(z)),
                amount=0.0,
                examples=_examples(z, k),
            )
        ]

    def monthly_outliers(
        self, sub: pd.DataFrame, k: int, z_threshold: float = 3.5
    ) -> list[Finding]:
        """Months where a property's net is far from its own typical month (robust z-score)."""
        out = []
        rows = []
        for prop, g in sub.dropna(subset=["property_name"]).groupby("property_name"):
            monthly = g.groupby("period")["profit"].sum()
            if len(monthly) < 6:
                continue
            med = monthly.median()
            mad = (monthly - med).abs().median()
            if mad == 0:
                continue
            z = 0.6745 * (monthly - med) / mad
            for p, zz in z[z.abs() > z_threshold].items():
                rows.append(
                    {
                        "property": prop,
                        "month": str(p),
                        "net": r2(monthly[p]),
                        "typical_month": r2(med),
                        "robust_z": round(float(zz), 1),
                    }
                )
        if rows:
            rows.sort(key=lambda r: -abs(r["robust_z"]))
            out.append(
                Finding(
                    "monthly_outliers",
                    "warning",
                    f"{len(rows)} property-months far from their typical level",
                    "Monthly net for these properties deviates strongly from the property's "
                    "median month (robust z-score > 3.5).",
                    count=len(rows),
                    examples=rows[: max(k, 5)],
                )
            )
        return out

    def tenant_gaps(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        """Tenants whose revenue stream stops before the end of coverage or has holes."""
        hi = sub["period"].max()
        rev = sub[(sub["ledger_type"] == "revenue") & (sub["profit"] != 0)].dropna(
            subset=["tenant_name"]
        )
        rows = []
        for tenant, g in rev.groupby("tenant_name"):
            months = sorted(g["period"].unique())
            if not months:
                continue
            span = (months[-1] - months[0]).n + 1
            missing = span - len(months)
            ended_early = months[-1] < hi
            if missing or ended_early:
                rows.append(
                    {
                        "tenant": tenant,
                        "first": str(months[0]),
                        "last": str(months[-1]),
                        "missing_months_within_span": int(missing),
                        "stopped_before_end_of_data": bool(ended_early),
                    }
                )
        if not rows:
            return []
        return [
            Finding(
                "tenant_gaps",
                "info",
                f"{len(rows)} tenants with gaps or an early end in their revenue stream",
                "Possible vacancies, lease ends or missing postings.",
                count=len(rows),
                examples=rows[: max(k, 5)],
            )
        ]

    def unallocated_expenses(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        exp = sub[sub["ledger_type"] == "expenses"]
        if exp.empty:
            return []
        unalloc = exp[exp["property_name"].isna()]
        share = unalloc["profit"].sum() / exp["profit"].sum() if exp["profit"].sum() else 0
        return [
            Finding(
                "unallocated_expenses",
                "info",
                f"{share:.0%} of expenses are not allocated to a property",
                "Mortgage interest, management/success fees, taxes and insurance are booked at "
                "entity level. Property-level P&L therefore overstates property profitability.",
                count=int(len(unalloc)),
                amount=r2(-unalloc["profit"].sum()),
            )
        ]

    def partial_period_coverage(self, sub: pd.DataFrame, k: int) -> list[Finding]:
        years = sub.groupby("year_num")["period"].nunique()
        partial = years[years < 12]
        if partial.empty:
            return []
        txt = ", ".join(f"{y}: {n} months" for y, n in partial.items())
        return [
            Finding(
                "partial_year",
                "info",
                "Some years are only partially covered",
                f"Year totals are not comparable like-for-like ({txt}). "
                "Compare equal periods (e.g. Q1 vs Q1) instead.",
                count=int(len(partial)),
            )
        ]


def _examples(df: pd.DataFrame, k: int) -> list[dict]:
    cols = [c for c in EXAMPLE_COLS if c in df.columns]
    return df[cols].head(k).astype(str).to_dict(orient="records")
