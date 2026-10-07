"""Load and normalise the general-ledger dataset.

The source file is a parquet (a CSV twin is kept next to it) with one row per ledger line:
entity, property, tenant, ledger taxonomy (type/group/category/code/description), month,
quarter, year and a signed ``profit`` amount (revenue positive, expenses negative, in EUR).

Normalisation adds typed period columns and a duplicate flag, and never mutates the raw
numbers. Whether duplicate rows are *excluded* from calculations is a runtime choice
(``LedgerData.view(dedupe=True)``), because we cannot know from the file alone whether four
identical bank charges in one month are four accounts or one row loaded four times.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "entity_name",
    "property_name",
    "tenant_name",
    "ledger_type",
    "ledger_group",
    "ledger_category",
    "ledger_code",
    "ledger_description",
    "month",
    "quarter",
    "year",
    "profit",
}


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be loaded or does not have the expected shape."""


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(f"Dataset not found at {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise DatasetError(f"Unsupported dataset format: {path.suffix}")


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and add typed helper columns. Returns a new DataFrame."""
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise DatasetError(f"Dataset is missing columns: {sorted(missing)}")

    df = raw.copy()
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0).astype(float)
    df["ledger_code"] = pd.to_numeric(df["ledger_code"], errors="coerce").astype("Int64")

    # month is "2024-M06" -> year_num=2024, month_num=6, period=2024-06
    month_parts = df["month"].astype(str).str.extract(r"^(\d{4})-M(\d{1,2})$")
    if month_parts.isna().any().any():
        bad = df.loc[month_parts.isna().any(axis=1), "month"].unique()[:5]
        raise DatasetError(f"Unparseable month values, e.g. {list(bad)}")
    df["year_num"] = month_parts[0].astype(int)
    df["month_num"] = month_parts[1].astype(int)
    df["quarter_num"] = (df["month_num"] - 1) // 3 + 1
    period_str = df["year_num"].astype(str) + "-" + df["month_num"].astype(str).str.zfill(2)
    df["period"] = pd.PeriodIndex(period_str, freq="M")

    # Duplicate detection on the original business columns only.
    business_cols = sorted(REQUIRED_COLUMNS)
    df["is_duplicate"] = df.duplicated(subset=business_cols, keep="first")

    # Consistent string columns (keep NaN for missing property/tenant - it is meaningful).
    for col in (
        "entity_name",
        "property_name",
        "tenant_name",
        "ledger_type",
        "ledger_group",
        "ledger_category",
        "ledger_description",
    ):
        df[col] = df[col].astype("string").str.strip()
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class LedgerData:
    """Immutable handle on the normalised ledger plus convenience views."""

    df: pd.DataFrame
    source: Path

    def view(self, dedupe: bool = False) -> pd.DataFrame:
        """Return the frame used for calculations. ``dedupe`` drops exact duplicate rows."""
        if dedupe:
            return self.df[~self.df["is_duplicate"]]
        return self.df

    @property
    def min_period(self) -> pd.Period:
        return self.df["period"].min()

    @property
    def max_period(self) -> pd.Period:
        return self.df["period"].max()

    @property
    def duplicate_rows(self) -> int:
        return int(self.df["is_duplicate"].sum())


@lru_cache(maxsize=4)
def load_ledger(path: str | Path) -> LedgerData:
    """Load, normalise and cache the dataset. Cached per path so Streamlit reruns are free."""
    path = Path(path)
    return LedgerData(df=normalise(_read_any(path)), source=path)
