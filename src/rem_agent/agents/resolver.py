"""Resolver: deterministic grounding of extracted mentions onto the real dataset.

This is where "bldg 17", "Building seventeen" and "building 17" all become "Building 17", where
"this year" becomes 2025-01..2025-03 (data-anchored), where "123 Main St" becomes an
``unknown_property`` issue with suggestions, and where silent defaults become explicit
``assumption`` issues so the final answer can state them. No LLM involved: cheap, testable,
and immune to hallucinated entity names.
"""

from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

from rem_agent.data.catalog import DataCatalog
from rem_agent.schemas import (
    SPECIALIST_FOR_INTENT,
    ExtractedSlots,
    Intent,
    Issue,
    PeriodSpec,
    ResolvedTask,
    SubQuestion,
    TimeSpec,
)
from rem_agent.tools import periods as P
from rem_agent.tools.periods import PeriodRange

_WORD_NUMBERS = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_GENERIC_PROPERTY = {
    "all",
    "portfolio",
    "properties",
    "property",
    "buildings",
    "building",
    "assets",
    "asset",
    "everything",
    "all properties",
    "all buildings",
    "my properties",
    "my assets",
    "the portfolio",
    "my property",
    "the property",
    "this property",
    "each property",
    "every property",
    "each building",
    "every building",
    "all my properties",
    "all assets",
    "all my assets",
    "my portfolio",
    "my buildings",
    "all of them",
    "them",
}
_GENERIC_TENANT = {
    "all",
    "tenants",
    "tenant",
    "all tenants",
    "my tenants",
    "top tenants",
    "the tenants",
    "each tenant",
    "every tenant",
    "all my tenants",
    "them",
}

# Account vocabulary -> ledger groups / categories (checked against the catalog at runtime).
_TERM_GROUPS = {
    "rent": {"categories": ["revenue_rent_taxed", "proceeds_rent_untaxed"]},
    "rental": {"groups": ["rental_income"]},
    "rental income": {"groups": ["rental_income"]},
    "parking": {"categories": ["proceeds_parking_taxed", "proceeds_parking_untaxed"]},
    "discount": {"groups": ["sales_discounts"]},
    "discounts": {"groups": ["sales_discounts"]},
    "interest": {"categories": ["interest_mortgage"]},
    "mortgage": {"categories": ["interest_mortgage"]},
    "loan": {"categories": ["interest_mortgage"]},
    "insurance": {"categories": ["insurance_in_general"]},
    "tax": {"categories": ["real_estate_taxes"]},
    "taxes": {"categories": ["real_estate_taxes"]},
    "property tax": {"categories": ["real_estate_taxes"]},
    "management fee": {"groups": ["management_fees"]},
    "management fees": {"groups": ["management_fees"]},
    "fees": {"groups": ["management_fees"]},
    "asset management": {"categories": ["asset_management_fees"]},
    "property management": {"categories": ["property_management_fees"]},
    "success fee": {"categories": ["success_fees"]},
    "vat": {"categories": ["vat_compensation", "non_reclaimable_vat"]},
    "bank": {"categories": ["bank_charges", "financial_expenses"]},
    "legal": {"categories": ["legal_advice"]},
    "maintenance": {"categories": ["maintenance_owner"]},
    "broker": {"categories": ["broker's_fees"]},
    "accountant": {"categories": ["accountant_costs"]},
    "general expenses": {"groups": ["general_expenses"]},
    "overhead": {"groups": ["general_expenses"]},
}


# ---- entity matching -----------------------------------------------------------------------------
def _digits(text: str) -> str | None:
    m = re.search(r"\d+", text)
    if m:
        return m.group(0)
    for w, n in _WORD_NUMBERS.items():
        if re.search(rf"\b{w}\b", text.lower()):
            return str(n)
    return None


def match_name(
    mention: str, candidates: list[str], generic: set[str]
) -> tuple[str | None, list[str], str]:
    """Return (match, suggestions, status) with status in {matched, generic, ambiguous, unknown}."""
    m = mention.strip().lower()
    m = re.sub(r"\b(bldg|bld|blg)\b\.?", "building", m)
    m = re.sub(r"[#.]", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    if m in generic or not m:
        return None, [], "generic"
    num = _digits(m)
    if num is not None:
        exact = [c for c in candidates if _digits(c) == str(int(num))]
        if len(exact) == 1:
            return exact[0], [], "matched"
        if len(exact) > 1:
            return None, exact, "ambiguous"
        sugg = [c for c, _, _ in process.extract(m, candidates, scorer=fuzz.WRatio, limit=3)]
        return None, sugg, "unknown"
    scored = process.extract(m, candidates, scorer=fuzz.WRatio, limit=3)
    if scored and scored[0][1] >= 88:
        return scored[0][0], [], "matched"
    if scored and scored[0][1] >= 60:
        return None, [c for c, _, _ in scored], "ambiguous"
    return None, [c for c, _, _ in scored], "unknown"


# ---- time ----------------------------------------------------------------------------------------
def timespec_to_range(spec: TimeSpec, as_of: pd.Period) -> PeriodRange | None:
    """TimeSpec -> PeriodRange.

    The raw phrase is parsed first (deterministic, anchored to the dataset's as-of month) because
    LLMs tend to turn "this year" into a concrete year from their own sense of "now". The
    structured fields are the fallback for phrasings the parser does not know.
    """
    if spec.kind == "all":
        return None
    parsed = P.parse_period_text(spec.raw, as_of)
    if parsed is not None:
        return parsed
    try:
        if spec.relative:
            rel = spec.relative
            if rel in ("this_year", "year_to_date"):
                return P.ytd_range(as_of)
            if rel == "last_year":
                return P.year_range(as_of.year - 1)
            if rel == "this_quarter":
                return P.current_quarter_range(as_of)
            if rel == "last_quarter":
                return P.current_quarter_range(as_of).previous()
            if rel == "this_month":
                return P.month_range(as_of.year, as_of.month)
            if rel == "last_month":
                prev = as_of - 1
                return P.month_range(prev.year, prev.month)
            if spec.n_months:
                return P.last_n_months(as_of, spec.n_months)
            return None  # same_period_last_year / previous_period are resolved by the caller
        if spec.kind == "year" and spec.year:
            return P.year_range(spec.year)
        if spec.kind == "quarter" and spec.year and spec.quarter:
            return P.quarter_range(spec.year, spec.quarter)
        if spec.kind == "month" and spec.year and spec.month:
            return P.month_range(spec.year, spec.month)
        if spec.n_months:
            return P.last_n_months(as_of, spec.n_months)
    except ValueError:
        pass
    return None


def _spec(r: PeriodRange | None) -> PeriodSpec | None:
    return PeriodSpec(start=str(r.start), end=str(r.end), label=r.label) if r else None


def spec_to_range(s: PeriodSpec | None) -> PeriodRange | None:
    if s is None:
        return None
    return PeriodRange(pd.Period(s.start, "M"), pd.Period(s.end, "M"), s.label)


# ---- main ----------------------------------------------------------------------------------------
_PROP_PATTERN = re.compile(r"\b(?:building|bldg|bld|blg)\.?\s*#?\s*(\d+)\b", re.I)
_TENANT_PATTERN = re.compile(r"\btenants?\s*#?\s*(\d+)\b", re.I)
_RELATIVE_PHRASES = [
    "same period last year",
    "same quarter last year",
    "same month last year",
    "year to date",
    "year-to-date",
    "ytd",
    "this year",
    "current year",
    "last year",
    "previous year",
    "prior year",
    "this quarter",
    "current quarter",
    "latest quarter",
    "most recent quarter",
    "last quarter",
    "previous quarter",
    "prior quarter",
    "this month",
    "current month",
    "latest month",
    "last month",
    "previous month",
]
_LAST_N = re.compile(r"\b(?:last|past|previous|trailing)\s+(\d{1,2})\s+months?\b", re.I)


_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][\w'-]*\s+){1,3}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Ln|Lane|Blvd|Boulevard|Dr|Drive|Way|Pl|Place|Ct|Court|"
    r"Sq|Square|Ter|Terrace|Hwy|Highway|Pkwy|Parkway|Straat|Laan|Weg|Plein)\b\.?"
)


def scan_entities(text: str) -> tuple[list[str], list[str]]:
    """Deterministic mentions ('Building 17', 'tenant 7', '123 Main St') straight from the text."""
    props = [f"Building {m}" for m in _PROP_PATTERN.findall(text or "")]
    props += [m.rstrip(".") for m in _ADDRESS_PATTERN.findall(text or "")]
    tenants = [f"Tenant {m}" for m in _TENANT_PATTERN.findall(text or "")]
    return props, tenants


def relative_phrases(text: str) -> list[str]:
    """Relative time phrases present in the text, in order of appearance."""
    found: list[tuple[int, str]] = []
    low = (text or "").lower()
    for phrase in _RELATIVE_PHRASES:
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", low):
            if not any(a <= m.start() < a + len(ph) for a, ph in found):
                found.append((m.start(), phrase))
    for m in _LAST_N.finditer(low):
        found.append((m.start(), m.group(0)))
    return [ph for _, ph in sorted(found)]


def _words_to_digits(text: str) -> str:
    out = text or ""
    for w, n in _WORD_NUMBERS.items():
        out = re.sub(rf"\b{w}\b", str(n), out, flags=re.I)
    return out


def _mentioned_number(name: str, texts: list[str]) -> bool:
    num = _digits(name)
    return num is None or any(re.search(rf"\b{num}\b", _words_to_digits(t)) for t in texts)


def resolve_task(
    sq: SubQuestion, slots: ExtractedSlots, catalog: DataCatalog, original: str | None = None
) -> ResolvedTask:
    as_of = catalog.as_of
    issues: list[Issue] = []
    task = ResolvedTask(
        id=sq.id, text=sq.text, intent=sq.intent, specialist=SPECIALIST_FOR_INTENT[sq.intent]
    )
    texts = [sq.text, original or ""]

    # Safety nets around the LLM extraction:
    # 1) add entities the regex finds in the text that the extractor missed;
    # 2) drop entities whose number is not in the text at all (hallucinated from the catalog);
    # 3) a relative time phrase in the user's words beats a concrete year the LLM substituted.
    scanned_props, scanned_tenants = scan_entities(f"{sq.text} {original or ''}")
    slots = slots.model_copy(deep=True)
    for name in scanned_props:
        if not any(
            _digits(m) == _digits(name) or m.lower() == name.lower() for m in slots.properties
        ):
            slots.properties.append(name)
    for name in scanned_tenants:
        if not any(_digits(m) == _digits(name) for m in slots.tenants):
            slots.tenants.append(name)
    slots.properties = [m for m in slots.properties if _mentioned_number(m, texts)]
    slots.tenants = [m for m in slots.tenants if _mentioned_number(m, texts)]
    rel = relative_phrases(original or sq.text)
    if rel and (not slots.timeframes or slots.timeframes[0].kind in ("year", "quarter", "month")):
        if not slots.timeframes or not any(
            str(ts.year) in (original or "") for ts in slots.timeframes[:1] if ts.year
        ):
            slots.timeframes = [TimeSpec(raw=r, kind="relative") for r in rel] + [
                ts for ts in slots.timeframes if ts.kind == "all"
            ]
            for ts in slots.timeframes:
                if ts.raw.startswith("same "):
                    ts.relative = "same_period_last_year"

    # Properties / tenants
    for mention in slots.properties:
        name, sugg, status = match_name(mention, catalog.properties, _GENERIC_PROPERTY)
        if status == "matched" and name not in task.properties:
            task.properties.append(name)
        elif status == "ambiguous":
            issues.append(
                Issue(
                    kind="ambiguous_property",
                    suggestions=sugg,
                    message=f"'{mention}' could be any of {', '.join(sugg)}.",
                )
            )
        elif status == "unknown":
            issues.append(
                Issue(
                    kind="unknown_property",
                    suggestions=sugg,
                    message=f"No property called '{mention}' in the dataset. Known properties: "
                    f"{', '.join(catalog.properties)}.",
                )
            )
    for mention in slots.tenants:
        name, sugg, status = match_name(mention, catalog.tenants, _GENERIC_TENANT)
        if status == "matched" and name not in task.tenants:
            task.tenants.append(name)
        elif status == "ambiguous":
            issues.append(
                Issue(
                    kind="ambiguous_tenant",
                    suggestions=sugg,
                    message=f"'{mention}' could be any of {', '.join(sugg)}.",
                )
            )
        elif status == "unknown":
            issues.append(
                Issue(
                    kind="unknown_tenant",
                    suggestions=sugg,
                    message=f"No tenant called '{mention}' in the dataset. Tenants are named "
                    f"{catalog.tenants[0]} to {catalog.tenants[-1]}.",
                )
            )

    if len(task.properties) >= len(catalog.properties):
        task.properties = []  # every property == the whole portfolio, no filter needed
    if (
        sq.intent == Intent.PERIOD_COMPARISON
        and len(task.properties) >= 2
        and len(slots.timeframes) < 2
    ):
        # "Compare Building 17 and Building 120 in 2024" is a property comparison, not a
        # comparison of two timeframes.
        task.intent = Intent.PNL
        task.specialist = SPECIALIST_FOR_INTENT[Intent.PNL]
        task.breakdown_by = task.breakdown_by or "property"
        sq = sq.model_copy(update={"intent": Intent.PNL})

    # Timeframes
    ranges: list[PeriodRange | None] = []
    for ts in slots.timeframes:
        r = timespec_to_range(ts, as_of)
        if (
            r is None
            and ts.kind != "all"
            and ts.relative not in ("same_period_last_year", "previous_period")
        ):
            issues.append(
                Issue(
                    kind="unparseable_timeframe",
                    message=f"Could not interpret the timeframe '{ts.raw}'; "
                    "using all available data.",
                )
            )
        ranges.append(r)
    primary = ranges[0] if ranges else None
    if primary is not None:
        _note_relative(slots.timeframes[0], primary, as_of, issues)
        clipped, was_clipped = primary.clip(catalog.min_period, catalog.max_period)
        if clipped is None:
            issues.append(
                Issue(
                    kind="out_of_coverage",
                    message=f"No data for {primary}; the ledger covers "
                    f"{catalog.min_period} to {catalog.max_period}.",
                )
            )
        elif was_clipped:
            issues.append(
                Issue(
                    kind="partial_coverage",
                    message=f"Only {clipped} of {primary} is covered by the data.",
                )
            )
    elif sq.intent in (Intent.PNL, Intent.TREND, Intent.TENANT_ANALYSIS, Intent.PORTFOLIO_OVERVIEW):
        issues.append(
            Issue(
                kind="assumption",
                message=f"No timeframe given; using all available data "
                f"({catalog.min_period} to {catalog.max_period}).",
            )
        )
    task.period = _spec(primary)

    # Comparison baseline
    if sq.intent == Intent.PERIOD_COMPARISON:
        if primary is None:
            primary = P.current_quarter_range(as_of)
            task.period = _spec(primary)
            issues.append(
                Issue(
                    kind="assumption",
                    message=f"No period given; comparing the latest quarter in the data "
                    f"({primary.label}) with the same quarter a year earlier.",
                )
            )
        second_spec = slots.timeframes[1] if len(slots.timeframes) > 1 else None
        baseline: PeriodRange | None = None
        if second_spec is not None:
            # An explicit, parseable period ("Q3 2024") always beats a relative flag the model
            # may have attached to it by mistake.
            if ranges[1] is not None:
                baseline = ranges[1]
            elif second_spec.relative == "same_period_last_year":
                baseline = primary.shift_years(-1)
            elif second_spec.relative == "previous_period":
                baseline = primary.previous()
        if baseline is None:
            baseline = primary.shift_years(-1)
            if second_spec is None:
                issues.append(
                    Issue(
                        kind="assumption",
                        message=f"Comparing {primary.label} with the same period a year "
                        f"earlier ({baseline.start} to {baseline.end}).",
                    )
                )
        baseline = PeriodRange(
            baseline.start, baseline.end, P.natural_label(baseline.start, baseline.end)
        )
        task.comparison_period = _spec(baseline)
        if baseline.clip(catalog.min_period, catalog.max_period)[0] is None:
            issues.append(
                Issue(
                    kind="out_of_coverage",
                    message=f"The comparison period {baseline} is outside the data "
                    f"({catalog.min_period} to {catalog.max_period}).",
                )
            )

    # Account filters
    if slots.ledger_types:
        task.ledger_types = sorted(set(slots.ledger_types))
    elif slots.metric in ("revenue", "expenses"):
        task.ledger_types = [slots.metric]
    groups, cats = set(), set()
    for term in slots.ledger_terms:
        g, c = match_terms(term, catalog)
        groups |= g
        cats |= c
    if cats:  # categories are more specific than groups; don't double-filter
        task.ledger_categories = sorted(cats)
    elif groups:
        task.ledger_groups = sorted(groups)

    task.metric = slots.metric
    task.top_n = slots.top_n
    task.breakdown_by = slots.breakdown_by or task.breakdown_by
    task.granularity = slots.granularity
    task.issues = issues
    return task


_GENERIC_TERMS = {
    "revenue",
    "revenues",
    "income",
    "expenses",
    "expense",
    "costs",
    "cost",
    "profit",
    "loss",
    "p&l",
    "pnl",
    "net",
    "total",
    "totals",
    "numbers",
    "figures",
    "results",
    "performance",
    "money",
    "cash",
    "earnings",
    "spend",
    "spending",
    "sales",
}


def match_terms(term: str, catalog: DataCatalog) -> tuple[set[str], set[str]]:
    t = term.strip().lower()
    groups: set[str] = set()
    cats: set[str] = set()
    if t in _GENERIC_TERMS:
        return groups, cats
    hit = _TERM_GROUPS.get(t)
    if hit is None:
        for key in sorted(_TERM_GROUPS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(key)}\b", t):
                hit = _TERM_GROUPS[key]
                break
    if hit:
        groups |= {g for g in hit.get("groups", []) if g in catalog.ledger_groups}
        cats |= {c for c in hit.get("categories", []) if c in catalog.ledger_categories}
        return groups, cats
    # Fuzzy fallback against category names ("other consultancy" -> other_consultancy_costs)
    best = process.extractOne(t, catalog.ledger_categories, scorer=fuzz.partial_ratio)
    if best and best[1] >= 90:
        cats.add(best[0])
    return groups, cats


_RELATIVE_WORDS = re.compile(
    r"\b(this|current|latest|last|previous|prior|past|recent|ytd|to date|trailing)\b", re.I
)


def _note_relative(ts: TimeSpec, r: PeriodRange, as_of: pd.Period, issues: list[Issue]) -> None:
    if ts.kind == "relative" or ts.relative or _RELATIVE_WORDS.search(ts.raw):
        issues.append(
            Issue(
                kind="assumption",
                message=f"'{ts.raw}' interpreted as {r.label} ({r}), anchored to the "
                f"latest month in the data ({as_of}).",
            )
        )


_CONCRETE = re.compile(
    r"\b(20\d\d|q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|"
    r"april|june|july|august|september|october|november|december|building|bldg|tenant|rent|"
    r"parking|interest|mortgage|insurance|tax|taxes|fee|fees|discount|vat|revenue|expenses|"
    r"profit|p&l|pnl|net|income|costs|anomal\w*|unusual|duplicate\w*|trend|compare|comparison|"
    r"top|best|worst|portfolio|propert\w+|asset\w*|month\w*|quarter\w*|year\w*|ytd)\b",
    re.I,
)


def has_concrete_mention(text: str) -> bool:
    """Does the user's own wording contain anything we can act on?"""
    return bool(_CONCRETE.search(text or "")) or bool(relative_phrases(text))


def merge_duplicate_tasks(tasks: list[ResolvedTask]) -> list[ResolvedTask]:
    """Collapse sub-questions that resolved to the same specialist and parameters.

    The router sometimes splits "trend for X - which month was worst?" into two sub-questions
    that need exactly the same tool call; running them twice wastes time and tokens.
    """
    merged: list[ResolvedTask] = []
    index: dict[tuple, int] = {}
    for t in tasks:
        key = (
            t.specialist,
            t.intent,
            tuple(t.properties),
            tuple(t.tenants),
            t.period.model_dump_json() if t.period else None,
            t.comparison_period.model_dump_json() if t.comparison_period else None,
            tuple(t.ledger_types),
            tuple(t.ledger_groups),
            tuple(t.ledger_categories),
        )
        if key in index:
            keep = merged[index[key]]
            keep.text = f"{keep.text} Also: {t.text}"
            keep.top_n = keep.top_n or t.top_n
            keep.breakdown_by = keep.breakdown_by or t.breakdown_by
            keep.granularity = keep.granularity or t.granularity
            seen = {i.message for i in keep.issues}
            keep.issues.extend(i for i in t.issues if i.message not in seen)
            continue
        index[key] = len(merged)
        merged.append(t)
    return merged


def resolve_all(
    sub_questions: list[SubQuestion],
    slots: dict[str, ExtractedSlots],
    catalog: DataCatalog,
    original: str | None = None,
) -> list[ResolvedTask]:
    tasks = [
        resolve_task(sq, slots.get(sq.id, ExtractedSlots(sub_question_id=sq.id)), catalog, original)
        for sq in sub_questions
    ]
    return merge_duplicate_tasks(tasks)
