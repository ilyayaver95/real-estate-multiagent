"""Offline tests for the input guard and the deterministic resolver (no LLM needed)."""

import pandas as pd
import pytest

from rem_agent.agents.guard import check_input
from rem_agent.agents.resolver import match_name, match_terms, resolve_task, timespec_to_range
from rem_agent.schemas import ExtractedSlots, Intent, SubQuestion, TimeSpec

AS_OF = pd.Period("2025-03", "M")
PROPS = ["Building 17", "Building 120", "Building 140", "Building 160", "Building 180"]


# ---- guard --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "empty"),
        ("   \n ", "empty"),
        (None, "non_text"),
        ("x" * 2001, "too_long"),
        ('{"a": 1, "b": [1,2,3]}', "structured_blob"),
        ("a,b,c,d\n1,2,3,4\n5,6,7,8", "structured_blob"),
        ("???!!!", "no_letters"),
        ("12345", "no_letters"),
    ],
)
def test_guard_rejects(text, reason):
    res = check_input(text)
    assert not res.ok and res.reason == reason and res.message


def test_guard_accepts_and_cleans():
    res = check_input("  What is   the P&L\nfor 2024? ")
    assert res.ok and res.cleaned == "What is the P&L for 2024?"


# ---- name matching ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mention,expected",
    [
        ("Building 17", "Building 17"),
        ("building 17", "Building 17"),
        ("bldg 17", "Building 17"),
        ("Bldg. #120", "Building 120"),
        ("building seventeen", "Building 17"),
        ("the 160 building", "Building 160"),
        ("17", "Building 17"),
    ],
)
def test_match_property_variants(mention, expected):
    name, _, status = match_name(mention, PROPS, {"all"})
    assert status == "matched" and name == expected


def test_match_property_unknown_and_generic():
    name, sugg, status = match_name("123 Main St", PROPS, {"all"})
    assert status == "unknown" and name is None and len(sugg) == 3
    name, sugg, status = match_name("Building 1", PROPS, {"all"})
    assert status == "unknown"  # no building numbered 1; must not silently pick 17 or 120
    assert match_name("all properties", PROPS, {"all properties"})[2] == "generic"
    name, _, status = match_name("Oak Avenue", PROPS, {"all"})
    assert status in ("unknown", "ambiguous") and name is None


def test_match_tenant_word_numbers():
    tenants = [f"Tenant {i}" for i in range(1, 19)]
    assert match_name("tenant seven", tenants, set())[0] == "Tenant 7"
    assert match_name("Tenant 12", tenants, set())[0] == "Tenant 12"
    assert match_name("Tenant 99", tenants, set())[2] == "unknown"


# ---- time ---------------------------------------------------------------------------------------
def test_timespec_structured_and_fallback():
    assert (
        str(timespec_to_range(TimeSpec(raw="2024", kind="year", year=2024), AS_OF))
        == "2024-01 to 2024-12"
    )
    assert (
        str(timespec_to_range(TimeSpec(raw="Q1 2025", kind="quarter", year=2025, quarter=1), AS_OF))
        == "2025-01 to 2025-03"
    )
    assert (
        str(
            timespec_to_range(
                TimeSpec(raw="this year", kind="relative", relative="this_year"), AS_OF
            )
        )
        == "2025-01 to 2025-03"
    )
    assert (
        str(
            timespec_to_range(
                TimeSpec(raw="last quarter", kind="relative", relative="last_quarter"), AS_OF
            )
        )
        == "2024-10 to 2024-12"
    )
    # model gave kind=unknown but the raw phrase is parseable
    assert str(timespec_to_range(TimeSpec(raw="June 2024"), AS_OF)) == "2024-06"
    assert timespec_to_range(TimeSpec(raw="whenever"), AS_OF) is None
    assert timespec_to_range(TimeSpec(raw="all time", kind="all"), AS_OF) is None
    # invalid structured values must not crash
    assert (
        timespec_to_range(TimeSpec(raw="Q5", kind="quarter", year=2024, quarter=None), AS_OF)
        is None
    )


# ---- account terms ------------------------------------------------------------------------------
def test_match_terms(catalog):
    g, c = match_terms("parking", catalog)
    assert c == {"proceeds_parking_taxed", "proceeds_parking_untaxed"} and not g
    g, c = match_terms("management fees", catalog)
    assert g == {"management_fees"}
    g, c = match_terms("mortgage interest", catalog)
    assert c == {"interest_mortgage"}
    assert match_terms("unicorns", catalog) == (set(), set())


# ---- resolve_task -------------------------------------------------------------------------------
def _sq(intent, text="q", id="q1"):
    return SubQuestion(id=id, text=text, intent=intent, rationale="test")


def test_resolve_pnl_this_year(catalog):
    slots = ExtractedSlots(
        sub_question_id="q1",
        properties=["all my properties"],
        timeframes=[TimeSpec(raw="this year", kind="relative", relative="this_year")],
    )
    task = resolve_task(_sq(Intent.PNL), slots, catalog)
    assert task.specialist == "finance"
    assert task.properties == []  # generic mention -> no filter
    assert task.period.start == "2025-01" and task.period.end == "2025-03"
    kinds = [i.kind for i in task.issues]
    assert "assumption" in kinds and "ambiguous_property" not in kinds


def test_resolve_unknown_property_is_not_blocking(catalog):
    slots = ExtractedSlots(sub_question_id="q1", properties=["123 Main St", "456 Oak Ave"])
    text = "Details for 123 Main St and 456 Oak Ave"
    task = resolve_task(_sq(Intent.PROPERTY_DETAILS, text), slots, catalog, original=text)
    assert task.properties == []
    assert [i.kind for i in task.issues] == ["unknown_property", "unknown_property"]
    assert task.blocking_issues == []
    assert all(i.suggestions for i in task.issues)


def test_resolve_comparison_defaults(catalog):
    slots = ExtractedSlots(
        sub_question_id="q1",
        timeframes=[
            TimeSpec(raw="this quarter", kind="relative", relative="this_quarter"),
            TimeSpec(
                raw="same period last year", kind="relative", relative="same_period_last_year"
            ),
        ],
    )
    task = resolve_task(_sq(Intent.PERIOD_COMPARISON), slots, catalog)
    assert (task.period.start, task.period.end) == ("2025-01", "2025-03")
    assert (task.comparison_period.start, task.comparison_period.end) == ("2024-01", "2024-03")
    assert task.comparison_period.label == "2024-Q1"
    # no timeframe at all -> latest quarter vs year earlier, with an explicit assumption
    task2 = resolve_task(
        _sq(Intent.PERIOD_COMPARISON), ExtractedSlots(sub_question_id="q1"), catalog
    )
    assert task2.period.label == "2025-Q1" and task2.comparison_period.label == "2024-Q1"
    assert any(i.kind == "assumption" for i in task2.issues)


def test_resolve_explicit_years_comparison_and_out_of_coverage(catalog):
    slots = ExtractedSlots(
        sub_question_id="q1",
        timeframes=[
            TimeSpec(raw="2024", kind="year", year=2024),
            TimeSpec(raw="2023", kind="year", year=2023),
        ],
    )
    task = resolve_task(_sq(Intent.PERIOD_COMPARISON), slots, catalog)
    assert task.comparison_period.label == "2023"
    assert any(i.kind == "out_of_coverage" for i in task.issues)


def test_resolve_partial_coverage_and_terms(catalog):
    slots = ExtractedSlots(
        sub_question_id="q1",
        properties=["bldg 17"],
        timeframes=[TimeSpec(raw="2025", kind="year", year=2025)],
        ledger_terms=["parking"],
        metric="revenue",
    )
    text = "Parking revenue for bldg 17 in 2025"
    task = resolve_task(_sq(Intent.PNL, text), slots, catalog, original=text)
    assert task.properties == ["Building 17"]
    assert task.ledger_categories == ["proceeds_parking_taxed", "proceeds_parking_untaxed"]
    assert task.ledger_types == ["revenue"]
    assert any(i.kind == "partial_coverage" for i in task.issues)


def test_resolve_ambiguous_is_blocking(catalog):
    # Force an ambiguous match through a non-numeric fuzzy mention of a tenant list
    slots = ExtractedSlots(sub_question_id="q1", tenants=["Tenan"])
    task = resolve_task(_sq(Intent.TENANT_ANALYSIS), slots, catalog)
    kinds = {i.kind for i in task.issues}
    assert kinds & {"ambiguous_tenant", "unknown_tenant"}


def test_relative_phrase_beats_model_guessed_year():
    """Regression: the LLM turned 'this year' into year=2024; the raw phrase must win."""
    spec = TimeSpec(raw="this year", kind="year", year=2024)
    assert str(timespec_to_range(spec, AS_OF)) == "2025-01 to 2025-03"
    spec = TimeSpec(raw="this quarter", kind="quarter", year=2024, quarter=4)
    assert str(timespec_to_range(spec, AS_OF)) == "2025-01 to 2025-03"
    # unknown raw phrasing falls back to the structured fields
    spec = TimeSpec(raw="the first three months of 2025", kind="quarter", year=2025, quarter=1)
    assert str(timespec_to_range(spec, AS_OF)) == "2025-01 to 2025-03"


# ---- resolver safety nets (regressions from live runs) -----------------------------------------
def test_hallucinated_properties_are_dropped(catalog):
    """Extractor listed all five buildings for a question that named none."""
    text = "How does this quarter compare to the same period last year?"
    slots = ExtractedSlots(
        sub_question_id="q1",
        properties=list(catalog.properties),
        timeframes=[
            TimeSpec(raw="current quarter", kind="relative", relative="this_quarter"),
            TimeSpec(
                raw="same quarter last year", kind="relative", relative="same_period_last_year"
            ),
        ],
    )
    task = resolve_task(_sq(Intent.PERIOD_COMPARISON, text), slots, catalog, original=text)
    assert task.properties == []
    assert task.period.label == "2025-Q1" and task.comparison_period.label == "2024-Q1"


def test_missed_property_is_recovered_from_text(catalog):
    """Extractor returned only Building 120 although the text names both buildings."""
    text = "Compare revenue of bldg 17 and Building 120 in 2024"
    sq = SubQuestion(
        id="q1",
        text="Compare revenue of Building 17 and Building 120 in 2024.",
        intent=Intent.PERIOD_COMPARISON,
        rationale="r",
    )
    slots = ExtractedSlots(
        sub_question_id="q1",
        properties=["Building 120"],
        timeframes=[TimeSpec(raw="2024", kind="year", year=2024)],
        ledger_terms=["revenue"],
        metric="revenue",
    )
    task = resolve_task(sq, slots, catalog, original=text)
    assert sorted(task.properties) == ["Building 120", "Building 17"]
    # property-vs-property in one year is a P&L breakdown, not a period comparison
    assert task.intent == Intent.PNL and task.breakdown_by == "property"
    assert task.comparison_period is None
    assert task.ledger_categories == []  # "revenue" is generic, not an account filter
    assert task.ledger_types == ["revenue"]


def test_relative_phrase_in_original_overrides_router_rewrite(catalog):
    """Router rewrote 'this year' as 'in 2024'; the user's own words must win."""
    original = "What is the total P&L for all my properties this year?"
    sq = SubQuestion(
        id="q1",
        text="What is the total P&L for all properties in 2024?",
        intent=Intent.PNL,
        rationale="r",
    )
    slots = ExtractedSlots(
        sub_question_id="q1", timeframes=[TimeSpec(raw="2024", kind="year", year=2024)]
    )
    task = resolve_task(sq, slots, catalog, original=original)
    assert task.period.label == "2025 YTD"
    assert any(i.kind == "assumption" and "this year" in i.message for i in task.issues)
    # but an explicit year in the user's words is respected
    original2 = "What is the total P&L for 2024?"
    task2 = resolve_task(sq, slots, catalog, original=original2)
    assert task2.period.label == "2024"


def test_word_numbers_survive_the_text_guard(catalog):
    text = "P&L for building seventeen"
    slots = ExtractedSlots(sub_question_id="q1", properties=["building seventeen"])
    task = resolve_task(_sq(Intent.PNL, text), slots, catalog, original=text)
    assert task.properties == ["Building 17"]


def test_address_only_question_is_unresolved(catalog):
    """Regression: 'P&L for 123 Main St' silently computed the whole portfolio."""
    text = "What is the P&L for 123 Main St in 2024?"
    slots = ExtractedSlots(
        sub_question_id="q1", timeframes=[TimeSpec(raw="2024", kind="year", year=2024)]
    )
    task = resolve_task(_sq(Intent.PNL, text), slots, catalog, original=text)
    assert task.properties == []
    assert [i.kind for i in task.unresolved_entities] == ["unknown_property"]
    assert "123 Main St" in task.unresolved_entities[0].message
    # mixed: one known, one unknown -> computable, not "unresolved"
    text2 = "Compare Building 17 with 456 Oak Ave"
    slots2 = ExtractedSlots(sub_question_id="q1", properties=["Building 17", "456 Oak Ave"])
    task2 = resolve_task(_sq(Intent.PNL, text2), slots2, catalog, original=text2)
    assert task2.properties == ["Building 17"] and task2.unresolved_entities == []
    assert any(i.kind == "unknown_property" for i in task2.issues)


@pytest.mark.parametrize(
    "text,concrete",
    [
        ("numbers please", False),
        ("how are we doing?", False),
        ("What is the total P&L?", True),
        ("and for 2025?", True),
        ("same for last quarter", True),
        ("Who are my top tenants?", True),
        ("Anything unusual?", True),
        ("Details for Building 17", True),
    ],
)
def test_has_concrete_mention(text, concrete):
    from rem_agent.agents.resolver import has_concrete_mention

    assert has_concrete_mention(text) is concrete


def test_explicit_second_period_beats_wrong_relative_flag(catalog):
    """Regression: extractor tagged 'Q3 2024' as same_period_last_year."""
    text = "Compare Q4 2024 with Q3 2024"
    slots = ExtractedSlots(
        sub_question_id="q1",
        timeframes=[
            TimeSpec(raw="Q4 2024", kind="quarter", year=2024, quarter=4),
            TimeSpec(
                raw="Q3 2024",
                kind="quarter",
                year=2024,
                quarter=3,
                relative="same_period_last_year",
            ),
        ],
    )
    task = resolve_task(_sq(Intent.PERIOD_COMPARISON, text), slots, catalog, original=text)
    assert task.period.label == "2024-Q4" and task.comparison_period.label == "2024-Q3"
    assert not any(i.kind == "out_of_coverage" for i in task.issues)


def test_out_of_coverage_flag(catalog):
    text = "What was the P&L in 2023?"
    slots = ExtractedSlots(
        sub_question_id="q1", timeframes=[TimeSpec(raw="2023", kind="year", year=2023)]
    )
    task = resolve_task(_sq(Intent.PNL, text), slots, catalog, original=text)
    assert task.out_of_coverage
    # partially covered year is NOT out of coverage
    slots2 = ExtractedSlots(
        sub_question_id="q1", timeframes=[TimeSpec(raw="2025", kind="year", year=2025)]
    )
    assert not resolve_task(
        _sq(Intent.PNL, "P&L 2025"), slots2, catalog, original="P&L 2025"
    ).out_of_coverage
