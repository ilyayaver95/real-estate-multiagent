"""Filter model shared by all analysis tools, plus the single function that applies it."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rem_agent.tools.periods import PeriodRange


@dataclass(frozen=True)
class LedgerFilter:
    """Resolved (already validated) filter. Empty lists mean 'no restriction'."""

    properties: tuple[str, ...] = ()
    tenants: tuple[str, ...] = ()
    period: PeriodRange | None = None
    ledger_types: tuple[str, ...] = ()
    ledger_groups: tuple[str, ...] = ()
    ledger_categories: tuple[str, ...] = ()

    def describe(self) -> dict:
        out: dict = {}
        if self.properties:
            out["properties"] = list(self.properties)
        if self.tenants:
            out["tenants"] = list(self.tenants)
        if self.period is not None:
            out["period"] = str(self.period)
        if self.ledger_types:
            out["ledger_types"] = list(self.ledger_types)
        if self.ledger_groups:
            out["ledger_groups"] = list(self.ledger_groups)
        if self.ledger_categories:
            out["ledger_categories"] = list(self.ledger_categories)
        return out

    def with_period(self, period: PeriodRange | None) -> LedgerFilter:
        return LedgerFilter(
            self.properties,
            self.tenants,
            period,
            self.ledger_types,
            self.ledger_groups,
            self.ledger_categories,
        )


def apply_filter(df: pd.DataFrame, f: LedgerFilter) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f.properties:
        mask &= df["property_name"].isin(f.properties).fillna(False)
    if f.tenants:
        mask &= df["tenant_name"].isin(f.tenants).fillna(False)
    if f.period is not None:
        mask &= (df["period"] >= f.period.start) & (df["period"] <= f.period.end)
    if f.ledger_types:
        mask &= df["ledger_type"].isin(f.ledger_types).fillna(False)
    if f.ledger_groups:
        mask &= df["ledger_group"].isin(f.ledger_groups).fillna(False)
    if f.ledger_categories:
        mask &= df["ledger_category"].isin(f.ledger_categories).fillna(False)
    return df[mask]
