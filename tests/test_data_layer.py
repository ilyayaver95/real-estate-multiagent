"""Data-layer checks. Expected values were computed independently with pandas on the raw file."""

import pandas as pd
import pytest

from rem_agent.data.loader import DatasetError, normalise


def test_shape_and_columns(ledger):
    assert len(ledger.df) == 3924
    for col in ("period", "year_num", "month_num", "quarter_num", "is_duplicate"):
        assert col in ledger.df.columns


def test_period_parsing(ledger):
    df = ledger.df
    assert str(ledger.min_period) == "2024-01"
    assert str(ledger.max_period) == "2025-03"
    # month/quarter/year columns in the file must agree with the derived ones
    assert (df["year_num"].astype(str) == df["year"]).all()
    derived_q = df["year_num"].astype(str) + "-Q" + df["quarter_num"].astype(str)
    assert (derived_q == df["quarter"]).all()


def test_duplicate_flagging(ledger):
    assert ledger.duplicate_rows == 1747
    assert len(ledger.view(dedupe=True)) == 3924 - 1747
    assert len(ledger.view(dedupe=False)) == 3924


def test_raw_numbers_untouched(ledger):
    assert ledger.df["profit"].sum() == pytest.approx(1_533_331.87, abs=0.01)


def test_catalog_entities(catalog):
    assert catalog.properties == [
        "Building 17",
        "Building 120",
        "Building 140",
        "Building 160",
        "Building 180",
    ]
    assert len(catalog.tenants) == 18
    assert catalog.tenants[0] == "Tenant 1" and catalog.tenants[-1] == "Tenant 18"
    assert catalog.ledger_types == ["expenses", "revenue"]
    assert catalog.years == [2024, 2025]
    assert str(catalog.as_of) == "2025-03"
    assert catalog.tenant_properties["Tenant 7"] == ["Building 120"]
    assert "Tenant 8" in catalog.property_tenants["Building 17"]


def test_describe_for_prompt_mentions_limits(catalog):
    text = catalog.describe_for_prompt()
    assert "Building 17" in text and "NOT in the data" in text


def test_normalise_rejects_missing_columns():
    with pytest.raises(DatasetError):
        normalise(pd.DataFrame({"foo": [1]}))


def test_normalise_rejects_bad_month():
    row = {
        "entity_name": "X",
        "property_name": None,
        "tenant_name": None,
        "ledger_type": "revenue",
        "ledger_group": "g",
        "ledger_category": "c",
        "ledger_code": 1,
        "ledger_description": "d",
        "month": "June 2024",
        "quarter": "2024-Q2",
        "year": "2024",
        "profit": 1.0,
    }
    with pytest.raises(DatasetError):
        normalise(pd.DataFrame([row]))
