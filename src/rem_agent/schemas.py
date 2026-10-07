"""Pydantic schemas shared across agents.

Three kinds of objects live here:
* LLM structured-output contracts (RouterOutput, ExtractionOutput) - what the model must fill.
* Resolved, deterministic task descriptions (ResolvedTask) - what the specialists execute.
* Results and trace events - what flows to the synthesizer, verifier and UI.

Everything is JSON-serialisable so LangGraph can checkpoint the state between turns.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Intent(StrEnum):
    PNL = "pnl"
    PERIOD_COMPARISON = "period_comparison"
    TREND = "trend"
    PROPERTY_DETAILS = "property_details"
    PORTFOLIO_OVERVIEW = "portfolio_overview"
    TENANT_ANALYSIS = "tenant_analysis"
    ANOMALY_AUDIT = "anomaly_audit"
    DATA_SCOPE = "data_scope"
    VALUATION_UNSUPPORTED = "valuation_unsupported"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CLARIFICATION = "clarification"
    OUT_OF_SCOPE = "out_of_scope"


# Which specialist handles which intent.
SPECIALIST_FOR_INTENT: dict[Intent, str] = {
    Intent.PNL: "finance",
    Intent.PERIOD_COMPARISON: "finance",
    Intent.TREND: "finance",
    Intent.PROPERTY_DETAILS: "portfolio",
    Intent.PORTFOLIO_OVERVIEW: "portfolio",
    Intent.TENANT_ANALYSIS: "portfolio",
    Intent.ANOMALY_AUDIT: "audit",
    Intent.DATA_SCOPE: "fallback",
    Intent.VALUATION_UNSUPPORTED: "fallback",
    Intent.GENERAL_KNOWLEDGE: "knowledge",
    Intent.OUT_OF_SCOPE: "fallback",
    Intent.CLARIFICATION: "fallback",
}


# ---- Router ------------------------------------------------------------------------------------
class SubQuestion(BaseModel):
    id: str = Field(description="Short id like q1, q2.")
    text: str = Field(description="Self-contained restatement of this part of the request.")
    intent: Intent
    rationale: str = Field(description="One sentence on why this intent was chosen.")


class RouterOutput(BaseModel):
    """Classification + decomposition of one user message."""

    sub_questions: list[SubQuestion] = Field(
        description="One entry per distinct thing the user asks. Compound questions produce "
        "several; a vague or unanswerable message still produces exactly one.",
        min_length=1,
    )
    needs_clarification: bool = Field(
        description="True only when the request cannot be acted on at all without more "
        "information, even with sensible defaults."
    )
    clarification_question: str | None = Field(
        default=None,
        description="The single most useful question to ask the user when needs_clarification.",
    )
    uses_conversation_context: bool = Field(
        default=False,
        description="True if earlier turns were needed to interpret this message "
        "(e.g. 'and for 2025?').",
    )


# ---- Extractor ---------------------------------------------------------------------------------
class TimeSpec(BaseModel):
    """A timeframe mention, as structured as the model can make it."""

    raw: str = Field(description="The phrase as the user wrote it, e.g. 'this year', 'Q1 2025'.")
    kind: Literal["year", "quarter", "month", "half", "range", "relative", "all", "unknown"] = (
        "unknown"
    )
    year: int | None = None
    quarter: int | None = Field(default=None, ge=1, le=4)
    month: int | None = Field(default=None, ge=1, le=12)
    half: int | None = Field(default=None, ge=1, le=2)
    end_year: int | None = Field(default=None, description="For ranges: end year.")
    end_month: int | None = Field(default=None, ge=1, le=12, description="For ranges: end month.")
    relative: (
        Literal[
            "this_year",
            "last_year",
            "this_quarter",
            "last_quarter",
            "this_month",
            "last_month",
            "year_to_date",
            "same_period_last_year",
            "previous_period",
        ]
        | None
    ) = None
    n_months: int | None = Field(default=None, description="For 'last 6 months' style phrases.")


class ExtractedSlots(BaseModel):
    sub_question_id: str
    properties: list[str] = Field(default_factory=list, description="Property mentions verbatim.")
    tenants: list[str] = Field(default_factory=list, description="Tenant mentions verbatim.")
    timeframes: list[TimeSpec] = Field(
        default_factory=list,
        description="Primary timeframe first; for comparisons the baseline/second period next.",
    )
    ledger_types: list[Literal["revenue", "expenses"]] = Field(default_factory=list)
    ledger_terms: list[str] = Field(
        default_factory=list,
        description="Account-like words the user used: rent, parking, interest, insurance, "
        "management fees, discounts, taxes...",
    )
    metric: Literal["net", "revenue", "expenses"] | None = Field(
        default=None, description="What the user asks to measure, if explicit."
    )
    top_n: int | None = Field(default=None, description="For 'top 3 tenants' style asks.")
    breakdown_by: (
        Literal["property", "tenant", "ledger_group", "ledger_category", "month", "quarter"] | None
    ) = None
    granularity: Literal["month", "quarter", "year"] | None = None
    notes: str | None = Field(default=None, description="Anything ambiguous worth flagging.")


class ExtractionOutput(BaseModel):
    slots: list[ExtractedSlots]


# ---- Resolver output ---------------------------------------------------------------------------
class PeriodSpec(BaseModel):
    """Serialisable PeriodRange."""

    start: str  # "2024-01"
    end: str  # "2024-12"
    label: str


class Issue(BaseModel):
    kind: Literal[
        "unknown_property",
        "unknown_tenant",
        "ambiguous_property",
        "ambiguous_tenant",
        "unparseable_timeframe",
        "out_of_coverage",
        "partial_coverage",
        "assumption",
    ]
    message: str
    suggestions: list[str] = Field(default_factory=list)


class ResolvedTask(BaseModel):
    id: str
    text: str
    intent: Intent
    specialist: str
    properties: list[str] = Field(default_factory=list)
    tenants: list[str] = Field(default_factory=list)
    period: PeriodSpec | None = None
    comparison_period: PeriodSpec | None = None
    ledger_types: list[str] = Field(default_factory=list)
    ledger_groups: list[str] = Field(default_factory=list)
    ledger_categories: list[str] = Field(default_factory=list)
    metric: str | None = None
    top_n: int | None = None
    breakdown_by: str | None = None
    granularity: str | None = None
    dedupe: bool | None = Field(default=None, description="User asked to exclude duplicates.")
    issues: list[Issue] = Field(default_factory=list)

    @property
    def blocking_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.kind in {"ambiguous_property", "ambiguous_tenant"}]

    @property
    def out_of_coverage(self) -> bool:
        """True when the primary period has no overlap with the data at all."""
        return self.period is not None and any(
            i.kind == "out_of_coverage" and i.message.startswith("No data for") for i in self.issues
        )

    @property
    def unresolved_entities(self) -> list[Issue]:
        """Unknown property/tenant mentions with no resolved counterpart of the same kind."""
        out = []
        if not self.properties:
            out += [i for i in self.issues if i.kind == "unknown_property"]
        if not self.tenants:
            out += [i for i in self.issues if i.kind == "unknown_tenant"]
        return out


# ---- Specialist results & trace ----------------------------------------------------------------
class ToolCallRecord(BaseModel):
    tool: str
    args: dict
    result_preview: str


class SpecialistResult(BaseModel):
    task_id: str
    specialist: str
    answer: str = Field(description="Specialist's answer to its sub-question, with figures.")
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    numbers_used: list[float] = Field(
        default_factory=list,
        description="Every numeric value returned by tools (for verification).",
    )
    caveats: list[str] = Field(default_factory=list)
    error: str | None = None


class TraceEvent(BaseModel):
    node: str
    summary: str
    detail: dict | list | str | None = None
    duration_ms: int | None = None
