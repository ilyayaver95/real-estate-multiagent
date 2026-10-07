"""Tool tests. Expected figures were computed independently with raw pandas pivots on the file:

revenue 2024 = 2,295,528.74   expenses 2024 = -1,124,007.19
revenue 2025 = 592,124.15     expenses 2025 = -230,313.83
2024-Q1 revenue 541,122.00, expenses -278,812.93 ; 2025-Q1 revenue 592,124.15
Building 120 revenue 880,535.66, direct expenses -29,968.24 ; Tenant 7 revenue 880,535.66
"""

import pandas as pd
import pytest

from rem_agent.tools.audit import AuditTools
from rem_agent.tools.filters import LedgerFilter, apply_filter
from rem_agent.tools.finance import FinanceTools
from rem_agent.tools.periods import (
    PeriodRange,
    parse_period_text,
    quarter_range,
    year_range,
)
from rem_agent.tools.portfolio import PortfolioTools

AS_OF = pd.Period("2025-03", "M")


@pytest.fixture(scope="module")
def fin(ledger):
    return FinanceTools(ledger)


@pytest.fixture(scope="module")
def port(ledger, catalog):
    return PortfolioTools(ledger, catalog)


@pytest.fixture(scope="module")
def audit(ledger):
    return AuditTools(ledger)


# ---- periods ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("2024", "2024-01 to 2024-12"),
        ("FY2024", "2024-01 to 2024-12"),
        ("Q1 2025", "2025-01 to 2025-03"),
        ("2025-Q1", "2025-01 to 2025-03"),
        ("2024q3", "2024-07 to 2024-09"),
        ("June 2024", "2024-06"),
        ("jun 2024", "2024-06"),
        ("2024-M06", "2024-06"),
        ("2024-06", "2024-06"),
        ("this year", "2025-01 to 2025-03"),
        ("last year", "2024-01 to 2024-12"),
        ("this quarter", "2025-01 to 2025-03"),
        ("last quarter", "2024-10 to 2024-12"),
        ("last 6 months", "2024-10 to 2025-03"),
        ("last month", "2025-02"),
    ],
)
def test_parse_period_text(text, expected):
    assert str(parse_period_text(text, AS_OF)) == expected


def test_parse_period_unknown_and_all():
    assert parse_period_text("whenever", AS_OF) is None
    assert parse_period_text("all time", AS_OF) is None


def test_period_range_helpers():
    q = quarter_range(2025, 1)
    assert str(q.shift_years(-1)) == "2024-01 to 2024-03"
    assert str(q.previous()) == "2024-10 to 2024-12"
    clipped, was = year_range(2025).clip(pd.Period("2024-01", "M"), AS_OF)
    assert was and str(clipped) == "2025-01 to 2025-03" and clipped.months == 3
    none, was = year_range(2023).clip(pd.Period("2024-01", "M"), AS_OF)
    assert none is None and was
    with pytest.raises(ValueError):
        PeriodRange(pd.Period("2025-02", "M"), pd.Period("2025-01", "M"), "bad")


# ---- filters ------------------------------------------------------------------------------
def test_apply_filter_property_and_period(ledger):
    sub = apply_filter(
        ledger.df, LedgerFilter(properties=("Building 17",), period=year_range(2024))
    )
    assert (sub["property_name"] == "Building 17").all()
    assert sub["year_num"].eq(2024).all()
    # unallocated rows must not leak into a property filter
    assert sub["property_name"].isna().sum() == 0


# ---- finance ------------------------------------------------------------------------------
def test_pnl_total_2024(fin):
    res = fin.pnl(LedgerFilter(period=year_range(2024)))
    assert res["revenue"] == pytest.approx(2_295_528.74, abs=0.01)
    assert res["expenses"] == pytest.approx(1_124_007.19, abs=0.01)
    assert res["net"] == pytest.approx(1_171_521.55, abs=0.01)
    assert res["breakdown_by"] == "ledger_group"
    assert any("duplicate" in c for c in res["caveats"])


def test_pnl_this_year_is_partial_and_says_so(fin):
    res = fin.pnl(LedgerFilter(period=year_range(2025)))
    assert res["revenue"] == pytest.approx(592_124.15, abs=0.01)
    assert res["expenses"] == pytest.approx(230_313.83, abs=0.01)
    assert any("partial" in c.lower() for c in res["caveats"])


def test_pnl_out_of_coverage(fin):
    res = fin.pnl(LedgerFilter(period=year_range(2023)))
    assert res["rows"] == 0 and res["net"] == 0
    assert any("No data for 2023" in c for c in res["caveats"])


def test_pnl_property_level_warns_about_unallocated(fin):
    res = fin.pnl(LedgerFilter(properties=("Building 120",)))
    assert res["revenue"] == pytest.approx(880_535.66, abs=0.01)
    assert res["expenses"] == pytest.approx(29_968.24, abs=0.01)
    assert any("NOT included" in c for c in res["caveats"])


def test_compare_quarters(fin):
    res = fin.compare_periods(LedgerFilter(), quarter_range(2025, 1), quarter_range(2024, 1))
    assert res["period_a"]["revenue"] == pytest.approx(592_124.15, abs=0.01)
    assert res["period_b"]["revenue"] == pytest.approx(541_122.00, abs=0.01)
    d = res["delta_a_minus_b"]["revenue"]
    assert d["absolute"] == pytest.approx(51_002.15, abs=0.01)
    assert d["relative"] == pytest.approx(0.0943, abs=0.001)
    assert len(res["largest_category_moves"]) > 0


def test_trend_and_breakdown(fin):
    tr = fin.trend(LedgerFilter(period=year_range(2024)), "quarter")
    assert [r["period"] for r in tr["series"]] == ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]
    assert tr["summary"]["best"]["period"] in {"2024-Q2", "2024-Q3"}
    bd = fin.breakdown(LedgerFilter(ledger_types=("revenue",)), by="property")
    names = [i["name"] for i in bd["items"]]
    assert names[0] == "Building 120"
    assert bd["items"][0]["amount"] == pytest.approx(880_535.66, abs=0.01)
    with pytest.raises(ValueError):
        fin.breakdown(LedgerFilter(), by="colour")


def test_dedupe_changes_numbers(ledger):
    raw = FinanceTools(ledger, dedupe=False).pnl(LedgerFilter(), breakdown_by=None)
    dd = FinanceTools(ledger, dedupe=True).pnl(LedgerFilter(), breakdown_by=None)
    assert raw["rows"] - dd["rows"] == 1747
    assert raw["net"] != dd["net"]
    assert not any("duplicate" in c for c in dd["caveats"])


# ---- portfolio ------------------------------------------------------------------------------
def test_property_details(port):
    d = port.property_details("Building 120")
    assert d["revenue"] == pytest.approx(880_535.66, abs=0.01)
    assert d["tenants"][0]["tenant"] == "Tenant 7"
    assert "valuation / price" in d["not_available"]
    assert port.property_details("123 Main St")["error"].startswith("Unknown property")


def test_portfolio_overview_ranking(port):
    ov = port.portfolio_overview()
    assert [p["property"] for p in ov["properties"]][:2] == ["Building 120", "Building 160"]
    assert ov["unallocated_expenses"] == pytest.approx(1_294_426.37, abs=0.01)


def test_top_tenants(port):
    t = port.top_tenants(n=3)
    assert [x["tenant"] for x in t["top"]] == ["Tenant 7", "Tenant 14", "Tenant 11"]
    assert t["top"][0]["revenue"] == pytest.approx(880_535.66, abs=0.01)
    assert t["top"][0]["properties"] == ["Building 120"]
    assert 0 < t["top_n_share"] < 1
    t2 = port.top_tenants(n=3, period=quarter_range(2025, 1))
    assert t2["top"][0]["revenue"] < t["top"][0]["revenue"]


def test_tenant_details(port):
    d = port.tenant_details("Tenant 7")
    assert d["properties"] == ["Building 120"] and d["active_months"] > 0
    assert "error" in port.tenant_details("Tenant 99")


# ---- audit ----------------------------------------------------------------------------------
def test_audit_finds_known_issues(audit):
    res = audit.run_all()
    checks = {f["check"]: f for f in res["findings"]}
    assert checks["duplicate_rows"]["count"] == 1747
    assert checks["double_mapped_code"]["title"].startswith("Ledger code 4650")
    assert checks["positive_expenses"]["count"] == 179
    assert checks["zero_amount_rows"]["count"] == 1348
    assert "reversal_pairs" in checks
    assert "unallocated_expenses" in checks and "partial_year" in checks
    assert res["summary"]["high"] >= 2
    # first finding is the most severe
    assert res["findings"][0]["severity"] == "high"


def test_audit_scoped_to_property(audit):
    res = audit.run_all(LedgerFilter(properties=("Building 17",)))
    assert res["rows_examined"] < 3924
    assert all(isinstance(f["examples"], list) for f in res["findings"])
