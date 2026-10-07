"""Catalog of the entities that exist in the dataset.

The catalog is the single source of truth for *what can be asked about*: property names,
tenant names, ledger taxonomy and the covered time range. It is handed to the LLM agents as
grounding context (so they do not invent "123 Main St") and used by the deterministic
resolver to fuzzy-match user wording onto real names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from rem_agent.data.loader import LedgerData


@dataclass(frozen=True)
class DataCatalog:
    ledger: LedgerData
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_df", self.ledger.df)

    # ---- entities -------------------------------------------------------------------------
    @cached_property
    def entities(self) -> list[str]:
        return sorted(self._df["entity_name"].dropna().unique().tolist())

    @cached_property
    def properties(self) -> list[str]:
        names = self._df["property_name"].dropna().unique().tolist()
        return sorted(names, key=_natural_key)

    @cached_property
    def tenants(self) -> list[str]:
        names = self._df["tenant_name"].dropna().unique().tolist()
        return sorted(names, key=_natural_key)

    @cached_property
    def tenant_properties(self) -> dict[str, list[str]]:
        """Tenant -> properties it appears under (ignoring rows without a property)."""
        sub = self._df.dropna(subset=["tenant_name", "property_name"])
        grouped = sub.groupby("tenant_name")["property_name"].unique()
        return {t: sorted(v.tolist(), key=_natural_key) for t, v in grouped.items()}

    @cached_property
    def property_tenants(self) -> dict[str, list[str]]:
        sub = self._df.dropna(subset=["tenant_name", "property_name"])
        grouped = sub.groupby("property_name")["tenant_name"].unique()
        return {p: sorted(v.tolist(), key=_natural_key) for p, v in grouped.items()}

    # ---- ledger taxonomy ------------------------------------------------------------------
    @cached_property
    def ledger_types(self) -> list[str]:
        return sorted(self._df["ledger_type"].dropna().unique().tolist())

    @cached_property
    def ledger_groups(self) -> list[str]:
        return sorted(self._df["ledger_group"].dropna().unique().tolist())

    @cached_property
    def ledger_categories(self) -> list[str]:
        return sorted(self._df["ledger_category"].dropna().unique().tolist())

    @cached_property
    def taxonomy(self) -> pd.DataFrame:
        """Distinct (type, group, category, code, description) combinations."""
        cols = [
            "ledger_type",
            "ledger_group",
            "ledger_category",
            "ledger_code",
            "ledger_description",
        ]
        return self._df[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)

    # ---- time coverage --------------------------------------------------------------------
    @property
    def min_period(self) -> pd.Period:
        return self.ledger.min_period

    @property
    def max_period(self) -> pd.Period:
        return self.ledger.max_period

    @cached_property
    def years(self) -> list[int]:
        return sorted(self._df["year_num"].unique().tolist())

    @cached_property
    def quarters(self) -> list[str]:
        return sorted(self._df["quarter"].unique().tolist())

    @property
    def as_of(self) -> pd.Period:
        """The dataset's 'today': relative phrases like 'this year' are anchored here."""
        return self.max_period

    # ---- prompt helpers -------------------------------------------------------------------
    def describe_for_prompt(self) -> str:
        """Compact, LLM-friendly description of what the data contains."""
        lines = [
            f"Entity: {', '.join(self.entities)}",
            f"Properties ({len(self.properties)}): {', '.join(self.properties)}",
            f"Tenants ({len(self.tenants)}): {', '.join(self.tenants)}",
            f"Ledger types: {', '.join(self.ledger_types)}",
            f"Ledger groups: {', '.join(self.ledger_groups)}",
            f"Ledger categories: {', '.join(self.ledger_categories)}",
            f"Time coverage: {self.min_period} to {self.max_period} (monthly ledger lines; "
            f"years {self.years}; quarters {', '.join(self.quarters)})",
            "Amounts: signed 'profit' in EUR (revenue positive, expenses negative).",
            "NOT in the data: addresses, purchase prices, valuations, appraisals, square metres,"
            " occupancy, lease terms, debt balances.",
        ]
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        return {
            "entities": self.entities,
            "properties": self.properties,
            "tenants": self.tenants,
            "ledger_types": self.ledger_types,
            "ledger_groups": self.ledger_groups,
            "ledger_categories": self.ledger_categories,
            "min_period": str(self.min_period),
            "max_period": str(self.max_period),
            "rows": int(len(self._df)),
            "duplicate_rows": self.ledger.duplicate_rows,
        }


def _natural_key(name: str) -> tuple:
    """Sort 'Building 17' before 'Building 120'."""
    parts = []
    for token in str(name).split():
        parts.append((0, int(token)) if token.isdigit() else (1, token.lower()))
    return tuple(parts)
