"""Property- and tenant-level descriptive tools (the 'asset details' side of the task).

Because the ledger has no valuations or addresses, "details" means: what the data knows about
an asset - its tenants, revenue and direct costs, active months and revenue mix.
"""

from __future__ import annotations

from rem_agent.data.catalog import DataCatalog
from rem_agent.data.loader import LedgerData
from rem_agent.tools.filters import LedgerFilter, apply_filter
from rem_agent.tools.formatting import r2, r4


class PortfolioTools:
    def __init__(self, ledger: LedgerData, catalog: DataCatalog, dedupe: bool = False):
        self.ledger = ledger
        self.catalog = catalog
        self.dedupe = dedupe
        self.df = ledger.view(dedupe=dedupe)

    def property_details(self, name: str, period=None) -> dict:
        if name not in self.catalog.properties:
            return {
                "error": f"Unknown property '{name}'",
                "known_properties": self.catalog.properties,
            }
        f = LedgerFilter(properties=(name,), period=period)
        sub = apply_filter(self.df, f)
        rev = sub[sub["ledger_type"] == "revenue"]
        exp = sub[sub["ledger_type"] == "expenses"]
        portfolio_rev = apply_filter(self.df, LedgerFilter(period=period))
        portfolio_rev = portfolio_rev.loc[portfolio_rev["ledger_type"] == "revenue", "profit"].sum()
        active = sub["period"]
        tenants = (
            rev.dropna(subset=["tenant_name"])
            .groupby("tenant_name")["profit"]
            .sum()
            .sort_values(ascending=False)
        )
        mix = rev.groupby("ledger_category")["profit"].sum().sort_values(key=abs, ascending=False)
        return {
            "property": name,
            "entity": self.catalog.entities[0] if self.catalog.entities else None,
            "period": str(period)
            if period is not None
            else f"{self.ledger.min_period} to {self.ledger.max_period}",
            "active_months": int(active.nunique()),
            "first_month": str(active.min()) if len(active) else None,
            "last_month": str(active.max()) if len(active) else None,
            "revenue": r2(rev["profit"].sum()),
            "direct_expenses": r2(-exp["profit"].sum()),
            "net_before_unallocated_costs": r2(sub["profit"].sum()),
            "share_of_portfolio_revenue": r4(rev["profit"].sum() / portfolio_rev)
            if portfolio_rev
            else None,
            "tenants": [{"tenant": t, "revenue": r2(v)} for t, v in tenants.items()],
            "revenue_mix": [{"category": k, "amount": r2(v)} for k, v in mix.items()],
            "not_available": [
                "address",
                "valuation / price",
                "appraisal date",
                "square metres",
                "occupancy rate",
            ],
            "caveats": [
                "Entity-level expenses (mortgage interest, management fees, taxes, insurance) "
                "are not allocated to properties, so this is not a full property P&L.",
            ],
        }

    def portfolio_overview(self, period=None) -> dict:
        f = LedgerFilter(period=period)
        sub = apply_filter(self.df, f)
        rows = []
        for prop in self.catalog.properties:
            p = sub[sub["property_name"] == prop]
            rows.append(
                {
                    "property": prop,
                    "revenue": r2(p.loc[p["ledger_type"] == "revenue", "profit"].sum()),
                    "direct_expenses": r2(-p.loc[p["ledger_type"] == "expenses", "profit"].sum()),
                    "net_before_unallocated_costs": r2(p["profit"].sum()),
                    "tenants": len(self.catalog.property_tenants.get(prop, [])),
                }
            )
        rows.sort(key=lambda r: r["revenue"], reverse=True)
        unalloc = sub[sub["property_name"].isna()]
        return {
            "period": str(period)
            if period is not None
            else f"{self.ledger.min_period} to {self.ledger.max_period}",
            "properties": rows,
            "unallocated_expenses": r2(
                -unalloc.loc[unalloc["ledger_type"] == "expenses", "profit"].sum()
            ),
            "portfolio_revenue": r2(sub.loc[sub["ledger_type"] == "revenue", "profit"].sum()),
            "portfolio_expenses": r2(-sub.loc[sub["ledger_type"] == "expenses", "profit"].sum()),
            "portfolio_net": r2(sub["profit"].sum()),
            "caveats": [
                "Property ranking uses revenue because most expenses are unallocated.",
            ],
        }

    def top_tenants(self, n: int = 5, period=None, properties: tuple[str, ...] = ()) -> dict:
        f = LedgerFilter(period=period, properties=properties, ledger_types=("revenue",))
        sub = apply_filter(self.df, f).dropna(subset=["tenant_name"])
        total = sub["profit"].sum()
        grp = sub.groupby("tenant_name").agg(
            revenue=("profit", "sum"), months=("period", "nunique")
        )
        grp = grp.sort_values("revenue", ascending=False)
        items = []
        for tenant, row in grp.head(n).iterrows():
            items.append(
                {
                    "tenant": tenant,
                    "properties": self.catalog.tenant_properties.get(tenant, []),
                    "revenue": r2(row["revenue"]),
                    "share": r4(row["revenue"] / total) if total else None,
                    "active_months": int(row["months"]),
                }
            )
        top_share = sum(i["share"] or 0 for i in items)
        return {
            "period": str(period)
            if period is not None
            else f"{self.ledger.min_period} to {self.ledger.max_period}",
            "filters": f.describe(),
            "metric": "net rental revenue (incl. discounts) attributed to the tenant",
            "total_tenant_revenue": r2(total),
            "tenants_count": int(len(grp)),
            "top": items,
            "top_n_share": r4(top_share),
            "concentration_note": (
                f"Top {len(items)} tenants generate {top_share:.0%} of tenant revenue."
                if items
                else "No tenant revenue in this period."
            ),
        }

    def tenant_details(self, name: str, period=None) -> dict:
        if name not in self.catalog.tenants:
            return {"error": f"Unknown tenant '{name}'", "known_tenants": self.catalog.tenants}
        f = LedgerFilter(tenants=(name,), period=period)
        sub = apply_filter(self.df, f)
        rev = sub[sub["ledger_type"] == "revenue"]
        monthly = rev.groupby("period")["profit"].sum()
        mix = rev.groupby("ledger_category")["profit"].sum().sort_values(key=abs, ascending=False)
        return {
            "tenant": name,
            "properties": self.catalog.tenant_properties.get(name, []),
            "period": str(period)
            if period is not None
            else f"{self.ledger.min_period} to {self.ledger.max_period}",
            "revenue": r2(rev["profit"].sum()),
            "active_months": int(monthly.index.nunique()),
            "first_month": str(monthly.index.min()) if len(monthly) else None,
            "last_month": str(monthly.index.max()) if len(monthly) else None,
            "average_monthly_revenue": r2(monthly.mean()) if len(monthly) else None,
            "revenue_mix": [{"category": k, "amount": r2(v)} for k, v in mix.items()],
            "monthly": [{"period": str(k), "revenue": r2(v)} for k, v in monthly.items()],
        }

    def data_dictionary(self) -> dict:
        tax = self.catalog.taxonomy
        return {
            "columns": {
                "entity_name": "legal entity owning the portfolio",
                "property_name": "building (null = entity-level line)",
                "tenant_name": "tenant (null = not tenant specific)",
                "ledger_type/group/category/code/description": "chart-of-accounts hierarchy",
                "month/quarter/year": "accounting period (monthly granularity)",
                "profit": "signed amount in EUR; revenue +, expenses -",
            },
            "taxonomy": tax.astype(str).to_dict(orient="records"),
            "coverage": f"{self.ledger.min_period} to {self.ledger.max_period}",
            "rows": int(len(self.ledger.df)),
        }
