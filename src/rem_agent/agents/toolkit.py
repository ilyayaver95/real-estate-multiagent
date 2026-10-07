"""LangChain tool wrappers around the deterministic analysis functions.

Tools are created per request so they close over the right ``dedupe`` setting, and every call
is recorded (name, args, result) so the verifier can later check that each number in the final
answer really came from a tool.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from langchain_core.tools import BaseTool, tool

from rem_agent.data.catalog import DataCatalog
from rem_agent.data.loader import LedgerData
from rem_agent.schemas import ResolvedTask, ToolCallRecord
from rem_agent.tools.audit import AuditTools
from rem_agent.tools.filters import LedgerFilter
from rem_agent.tools.finance import FinanceTools
from rem_agent.tools.periods import PeriodRange, parse_period_text
from rem_agent.tools.portfolio import PortfolioTools


class ToolRecorder:
    """Collects every tool call and every number returned, for tracing and verification."""

    def __init__(self) -> None:
        self.calls: list[ToolCallRecord] = []
        self.numbers: list[float] = []
        self.caveats: list[str] = []

    def record(self, name: str, args: dict, result: Any) -> str:
        payload = json.dumps(result, default=str, ensure_ascii=False)
        self.calls.append(ToolCallRecord(tool=name, args=args, result_preview=payload[:600]))
        self.numbers.extend(_collect_numbers(result))
        if isinstance(result, dict):
            self.caveats.extend(result.get("caveats", []) or [])
        return payload


_NUM_IN_TEXT = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+")


def _collect_numbers(obj: Any, acc: list[float] | None = None) -> list[float]:
    acc = [] if acc is None else acc
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, int | float):
        acc.append(float(obj))
    elif isinstance(obj, str):
        for m in _NUM_IN_TEXT.findall(obj):  # amounts quoted inside caveat/finding text
            acc.append(float(m.replace(",", "")))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, acc)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _collect_numbers(v, acc)
    return acc


def _period(
    start: str | None, end: str | None, label: str | None, as_of: pd.Period
) -> PeriodRange | None:
    """Accept 'YYYY-MM' bounds or a free-text label ('2024', 'Q1 2025')."""
    if start and end:
        return PeriodRange(pd.Period(start, "M"), pd.Period(end, "M"), label or f"{start} to {end}")
    if label:
        return parse_period_text(label, as_of)
    return None


def build_tools(
    ledger: LedgerData,
    catalog: DataCatalog,
    task: ResolvedTask,
    recorder: ToolRecorder,
    dedupe: bool = False,
) -> dict[str, list[BaseTool]]:
    """Return the tool sets per specialist. Defaults come from the resolved task so the LLM
    only has to pass what it wants to override."""
    fin = FinanceTools(ledger, dedupe=dedupe)
    port = PortfolioTools(ledger, catalog, dedupe=dedupe)
    aud = AuditTools(ledger, dedupe=dedupe)
    as_of = catalog.as_of

    def _filter(
        properties: list[str] | None,
        tenants: list[str] | None,
        start: str | None,
        end: str | None,
        period_label: str | None,
        ledger_types: list[str] | None = None,
        ledger_groups: list[str] | None = None,
        ledger_categories: list[str] | None = None,
    ) -> LedgerFilter:
        props = tuple(properties if properties is not None else task.properties)
        tens = tuple(tenants if tenants is not None else task.tenants)
        if start or end or period_label:
            per = _period(start, end, period_label, as_of)
        else:
            per = _spec_range(task.period)
        return LedgerFilter(
            properties=tuple(p for p in props if p in catalog.properties),
            tenants=tuple(t for t in tens if t in catalog.tenants),
            period=per,
            ledger_types=tuple(ledger_types if ledger_types is not None else task.ledger_types),
            ledger_groups=tuple(ledger_groups if ledger_groups is not None else task.ledger_groups),
            ledger_categories=tuple(
                ledger_categories if ledger_categories is not None else task.ledger_categories
            ),
        )

    # ---- finance -------------------------------------------------------------------------------
    @tool
    def get_pnl(
        properties: list[str] | None = None,
        tenants: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        period_label: str | None = None,
        ledger_types: list[str] | None = None,
        ledger_groups: list[str] | None = None,
        ledger_categories: list[str] | None = None,
        breakdown_by: str | None = "ledger_group",
    ) -> str:
        """Revenue, expenses and net profit for a selection of the ledger.
        All arguments are optional; omitted ones default to the resolved task parameters.
        start/end are 'YYYY-MM' months (inclusive); period_label is a phrase like '2024' or
        'Q1 2025'. breakdown_by: ledger_group | ledger_category | property | tenant | month |
        quarter | None."""
        f = _filter(
            properties,
            tenants,
            start,
            end,
            period_label,
            ledger_types,
            ledger_groups,
            ledger_categories,
        )
        res = fin.pnl(f, breakdown_by=breakdown_by or None)
        return recorder.record("get_pnl", _args(locals()), res)

    @tool
    def compare_periods(
        start_a: str | None = None,
        end_a: str | None = None,
        label_a: str | None = None,
        start_b: str | None = None,
        end_b: str | None = None,
        label_b: str | None = None,
        properties: list[str] | None = None,
        tenants: list[str] | None = None,
        ledger_types: list[str] | None = None,
    ) -> str:
        """Compare P&L between period A and baseline period B (A minus B, absolute and %).
        Defaults to the task's resolved period and comparison period. Bounds are 'YYYY-MM'."""
        pa = _period(start_a, end_a, label_a, as_of) or _spec_range(task.period)
        pb = _period(start_b, end_b, label_b, as_of) or _spec_range(task.comparison_period)
        if pa is None or pb is None:
            return recorder.record(
                "compare_periods",
                _args(locals()),
                {"error": "Both periods are required (e.g. start_a='2025-01', end_a='2025-03')."},
            )
        f = _filter(properties, tenants, None, None, None, ledger_types).with_period(None)
        res = fin.compare_periods(f, pa, pb)
        return recorder.record("compare_periods", _args(locals()), res)

    @tool
    def get_trend(
        granularity: str = "month",
        properties: list[str] | None = None,
        tenants: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        period_label: str | None = None,
        ledger_types: list[str] | None = None,
    ) -> str:
        """Time series of revenue/expenses/net by month, quarter or year, with best/worst period."""
        f = _filter(properties, tenants, start, end, period_label, ledger_types)
        res = fin.trend(f, granularity=granularity)
        return recorder.record("get_trend", _args(locals()), res)

    @tool
    def get_breakdown(
        by: str,
        top: int = 10,
        properties: list[str] | None = None,
        tenants: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        period_label: str | None = None,
        ledger_types: list[str] | None = None,
    ) -> str:
        """Sum of profit grouped by a dimension: property | tenant | ledger_group |
        ledger_category | month | quarter | year. Pass ledger_types=['revenue'] to rank by revenue.
        """
        f = _filter(properties, tenants, start, end, period_label, ledger_types)
        res = fin.breakdown(f, by=by, top=top)
        return recorder.record("get_breakdown", _args(locals()), res)

    # ---- portfolio -----------------------------------------------------------------------------
    @tool
    def get_property_details(
        property_name: str,
        start: str | None = None,
        end: str | None = None,
        period_label: str | None = None,
    ) -> str:
        """Everything the ledger knows about one property: tenants, revenue, direct costs,
        active months, revenue mix, share of portfolio. Also lists what is NOT available."""
        res = port.property_details(
            property_name, _period(start, end, period_label, as_of) or _spec_range(task.period)
        )
        return recorder.record("get_property_details", _args(locals()), res)

    @tool
    def get_portfolio_overview(
        start: str | None = None, end: str | None = None, period_label: str | None = None
    ) -> str:
        """Rank all properties by revenue/direct expenses/net, plus portfolio totals and the
        unallocated (entity-level) expenses."""
        res = port.portfolio_overview(
            _period(start, end, period_label, as_of) or _spec_range(task.period)
        )
        return recorder.record("get_portfolio_overview", _args(locals()), res)

    @tool
    def get_top_tenants(
        n: int = 5,
        properties: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        period_label: str | None = None,
    ) -> str:
        """Top tenants by attributed revenue with share and concentration."""
        props = tuple(properties if properties is not None else task.properties)
        res = port.top_tenants(
            n=n,
            period=_period(start, end, period_label, as_of) or _spec_range(task.period),
            properties=tuple(p for p in props if p in catalog.properties),
        )
        return recorder.record("get_top_tenants", _args(locals()), res)

    @tool
    def get_tenant_details(
        tenant_name: str,
        start: str | None = None,
        end: str | None = None,
        period_label: str | None = None,
    ) -> str:
        """Revenue, properties, active months and monthly series for one tenant."""
        res = port.tenant_details(
            tenant_name, _period(start, end, period_label, as_of) or _spec_range(task.period)
        )
        return recorder.record("get_tenant_details", _args(locals()), res)

    @tool
    def get_data_dictionary() -> str:
        """Columns, chart of accounts and coverage of the dataset (for 'what data do you have')."""
        res = port.data_dictionary()
        return recorder.record("get_data_dictionary", {}, res)

    # ---- audit ---------------------------------------------------------------------------------
    @tool
    def run_anomaly_audit(
        properties: list[str] | None = None,
        tenants: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        period_label: str | None = None,
    ) -> str:
        """Run all data-quality and anomaly checks (duplicates, double-mapped accounts, reversals,
        sign anomalies, zero rows, monthly outliers, tenant gaps, unallocated expenses, partial
        years) on the selected slice. Findings are ordered by severity."""
        f = _filter(properties, tenants, start, end, period_label)
        res = aud.run_all(f)
        return recorder.record("run_anomaly_audit", _args(locals()), res)

    finance = [get_pnl, compare_periods, get_trend, get_breakdown, get_top_tenants]
    portfolio = [
        get_property_details,
        get_portfolio_overview,
        get_top_tenants,
        get_tenant_details,
        get_pnl,
        get_breakdown,
        get_data_dictionary,
    ]
    audit = [run_anomaly_audit, get_pnl, get_trend]
    return {"finance": finance, "portfolio": portfolio, "audit": audit}


def _spec_range(spec) -> PeriodRange | None:
    if spec is None:
        return None
    return PeriodRange(pd.Period(spec.start, "M"), pd.Period(spec.end, "M"), spec.label)


def _args(local_vars: dict) -> dict:
    """Keep only the caller's explicit, JSON-like arguments for the trace."""
    out = {}
    for k, v in local_vars.items():
        if k.startswith("_") or v is None:
            continue
        if isinstance(v, str | int | float | bool) or (
            isinstance(v, list) and all(isinstance(x, str | int | float | bool) for x in v)
        ):
            out[k] = v
    return out
