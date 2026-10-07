"""Specialist agents.

* finance / portfolio / audit: an LLM with a focused tool set runs a short tool-calling loop
  (LLM -> tools -> LLM ...) until it answers. Every number it can use comes from a tool.
* knowledge: one LLM call, no tools, clearly labelled as general knowledge.
* fallback: deterministic, no LLM. Explains unsupported requests (valuations, addresses),
  out-of-scope messages and data-scope questions using the catalog.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rem_agent.agents.prompts import KNOWLEDGE_SYSTEM, SPECIALIST_SYSTEM
from rem_agent.agents.toolkit import ToolRecorder, build_tools
from rem_agent.data.catalog import DataCatalog
from rem_agent.data.loader import LedgerData
from rem_agent.schemas import Intent, ResolvedTask, SpecialistResult

SPECIALIST_EXTRA = {
    "finance": (
        "For P&L questions call get_pnl (and get_breakdown if the user asks what drives it). "
        "For comparisons between periods you MUST call compare_periods, which returns the deltas; "
        "never subtract or compute percentages yourself. If the result has a 'like_for_like' "
        "block, quote those growth rates (same months in both periods) and say the full-period "
        "deltas are distorted by partial coverage. If the result has 'share_of_scope', use it for "
        "any "
        "share/percentage question. To compare properties or tenants call "
        "get_breakdown(by='property'|'tenant'). For trends call get_trend. If the resolved period "
        "is partial (e.g. 2025 has 3 months) say so and, when comparing years, suggest a "
        "like-for-like comparison. If the comparison period is outside the data, still report the "
        "primary period's figures (get_pnl) and offer the earliest comparable period that exists."
    ),
    "portfolio": (
        "For a named property call get_property_details; for rankings or 'which property' call "
        "get_portfolio_overview; for tenants call get_top_tenants or get_tenant_details. If the "
        "task has no resolved property but mentions one that is unknown, do NOT guess: say it is "
        "not in the dataset and list the known properties."
    ),
    "audit": (
        "Call run_anomaly_audit once (scoped to the task's filters). Report the high-severity "
        "findings first with counts and amounts, then warnings briefly. Explain in one line what "
        "each finding would mean for the user's numbers."
    ),
}


def prefetch_call(task: ResolvedTask) -> tuple[str, dict] | None:
    """The one tool call that answers most tasks of this intent, chosen deterministically.

    Running it before the LLM sees the task guarantees the right tool is used (e.g. deltas come
    from compare_periods, not from LLM arithmetic) and saves a full model round-trip.
    """
    i = task.intent
    if i == Intent.PNL:
        return "get_pnl", {"breakdown_by": task.breakdown_by or "ledger_group"}
    if i == Intent.PERIOD_COMPARISON:
        return "compare_periods", {}
    if i == Intent.TREND:
        return "get_trend", {"granularity": task.granularity or "month"}
    if i == Intent.PROPERTY_DETAILS:
        if len(task.properties) == 1:
            return "get_property_details", {"property_name": task.properties[0]}
        return "get_portfolio_overview", {}
    if i == Intent.PORTFOLIO_OVERVIEW:
        return "get_portfolio_overview", {}
    if i == Intent.TENANT_ANALYSIS:
        if len(task.tenants) == 1:
            return "get_tenant_details", {"tenant_name": task.tenants[0]}
        return "get_top_tenants", {"n": task.top_n or 5}
    if i == Intent.ANOMALY_AUDIT:
        return "run_anomaly_audit", {}
    return None


def run_tool_specialist(
    llm: BaseChatModel,
    ledger: LedgerData,
    catalog: DataCatalog,
    task: ResolvedTask,
    dedupe: bool = False,
    max_rounds: int = 6,
) -> SpecialistResult:
    recorder = ToolRecorder()
    tools = build_tools(ledger, catalog, task, recorder, dedupe=dedupe)[task.specialist]
    by_name = {t.name: t for t in tools}
    params = task.model_dump(
        include={
            "properties",
            "tenants",
            "period",
            "comparison_period",
            "ledger_types",
            "ledger_groups",
            "ledger_categories",
            "metric",
            "top_n",
            "breakdown_by",
            "granularity",
        },
        exclude_none=True,
    )
    issues = [i.message for i in task.issues] or ["none"]
    if dedupe:
        issues.append("Exact duplicate ledger rows are EXCLUDED from all figures in this answer.")
    system = SPECIALIST_SYSTEM.format(
        name=task.specialist,
        task_text=task.text,
        params=json.dumps(params, default=str),
        issues="; ".join(issues),
        coverage=f"{catalog.min_period} to {catalog.max_period}",
        as_of=catalog.as_of,
        extra=SPECIALIST_EXTRA.get(task.specialist, ""),
    )
    user_text = task.text
    pre = prefetch_call(task)
    if pre and pre[0] in by_name:
        name, args = pre
        try:
            payload = by_name[name].invoke(args)
        except Exception as exc:  # fall back to letting the model call tools itself
            payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        user_text = (
            f"{task.text}\n\nPre-computed result of {name}({json.dumps(args)}) for this task:\n"
            f"{payload}\n\nUse it. Call other tools only if something needed is missing."
        )
    messages = [SystemMessage(content=system), HumanMessage(content=user_text)]
    model = llm.bind_tools(tools)
    answer: str | None = None
    for _ in range(max_rounds):
        ai: AIMessage = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            answer = _text(ai)
            break
        for call in ai.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                content = json.dumps({"error": f"unknown tool {call['name']}"})
            else:
                try:
                    content = tool.invoke(call["args"])
                except Exception as exc:  # tool errors are fed back, not raised
                    content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
    if answer is None:
        # Ran out of rounds: force a final answer without more tool calls.
        final = llm.bind_tools(tools, tool_choice="none").invoke(
            messages + [HumanMessage(content="Answer now using the tool results above.")]
        )
        answer = _text(final)
    return SpecialistResult(
        task_id=task.id,
        specialist=task.specialist,
        answer=answer.strip(),
        tool_calls=recorder.calls,
        numbers_used=recorder.numbers,
        caveats=sorted(set(recorder.caveats)),
    )


def run_knowledge_specialist(
    llm: BaseChatModel, catalog: DataCatalog, task: ResolvedTask
) -> SpecialistResult:
    prompt = KNOWLEDGE_SYSTEM.format(
        coverage=f"{catalog.min_period} to {catalog.max_period}", task_text=task.text
    )
    ai = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=task.text)])
    return SpecialistResult(
        task_id=task.id,
        specialist="knowledge",
        answer=_text(ai).strip(),
        caveats=["General knowledge answer; not computed from your ledger."],
    )


def run_not_found_specialist(catalog: DataCatalog, task: ResolvedTask) -> SpecialistResult:
    """The user named only properties/tenants that do not exist: say so, never compute."""
    unknown = task.unresolved_entities
    names = ", ".join(f"'{_mention(i.message)}'" for i in unknown)
    kinds = {i.kind for i in unknown}
    lines = [f"I couldn't find {names} in the dataset."]
    if "unknown_property" in kinds:
        lines.append("The ledger covers these properties: " + ", ".join(catalog.properties) + ".")
    if "unknown_tenant" in kinds:
        lines.append(f"Tenants are named {catalog.tenants[0]} to {catalog.tenants[-1]}.")
    sugg = [s for i in unknown for s in i.suggestions][:3]
    if sugg:
        lines.append("Closest names: " + ", ".join(sugg) + ".")
    what = {
        Intent.PNL: "the P&L",
        Intent.PERIOD_COMPARISON: "the period comparison",
        Intent.TREND: "the trend",
        Intent.PROPERTY_DETAILS: "the details",
        Intent.TENANT_ANALYSIS: "the tenant analysis",
    }.get(task.intent, "this")
    period = f" for {task.period.label}" if task.period else ""
    lines.append(f"Tell me which one you mean and I'll compute {what}{period} for it.")
    return SpecialistResult(
        task_id=task.id,
        specialist="fallback",
        answer=" ".join(lines),
        caveats=[i.message for i in unknown],
    )


def run_no_data_specialist(catalog: DataCatalog, task: ResolvedTask) -> SpecialistResult:
    """The requested period lies entirely outside the ledger: say so, suggest what exists."""
    label = task.period.label if task.period else "that period"
    lo, hi = catalog.min_period, catalog.max_period
    if task.period and task.period.end < str(lo):
        closest = f"{lo.year} (the first full year available)"
    else:
        closest = f"{hi.year} year-to-date ({hi.year}-01 to {hi})"
    scope = ""
    if task.properties:
        scope = " for " + ", ".join(task.properties)
    if task.tenants:
        scope += " for " + ", ".join(task.tenants)
    answer = (
        f"There is no data for {label}{scope}: the ledger covers {lo} to {hi} only, so nothing "
        f"can be computed for that period. The closest available period is {closest}; ask me "
        "for that and I'll run the numbers."
    )
    return SpecialistResult(
        task_id=task.id,
        specialist="fallback",
        answer=answer,
        caveats=[i.message for i in task.issues if i.kind == "out_of_coverage"],
    )


def run_fallback_specialist(catalog: DataCatalog, task: ResolvedTask) -> SpecialistResult:
    """Deterministic answers for things the data cannot support."""
    props = ", ".join(catalog.properties)
    unknown = [i for i in task.issues if i.kind in ("unknown_property", "unknown_tenant")]
    lines: list[str] = []
    if task.intent == Intent.VALUATION_UNSUPPORTED:
        lines.append(
            "I can't answer that from the data I have. The dataset is a general ledger "
            "(monthly revenue and expense postings per property and tenant for "
            f"{catalog.min_period} to {catalog.max_period}); it contains no prices, valuations, "
            "appraisals, addresses, floor areas or occupancy figures."
        )
        if unknown:
            names = ", ".join(f"'{_mention(i.message)}'" for i in unknown)
            lines.append(f"Also, {names} do not appear in the dataset at all.")
        lines.append(
            f"Known properties: {props}. What I can do instead: compare their revenue, direct "
            "costs and net result for any period, e.g. 'Compare Building 17 and Building 120 in "
            "2024'."
        )
    elif task.intent == Intent.DATA_SCOPE:
        lines.append(
            f"The dataset is a monthly general ledger for {', '.join(catalog.entities)} covering "
            f"{catalog.min_period} to {catalog.max_period} ({len(catalog.ledger.df):,} rows)."
        )
        lines.append(f"Properties ({len(catalog.properties)}): {props}.")
        lines.append(
            f"Tenants: {len(catalog.tenants)} ({catalog.tenants[0]} to {catalog.tenants[-1]})."
        )
        lines.append(
            "Accounts: revenue (rent, parking, VAT compensation, discounts) and expenses "
            f"({', '.join(_expense_groups(catalog))})."
        )
        lines.append(
            "Not available: prices/valuations, addresses, floor areas, occupancy, lease terms, "
            "debt balances. Ask me about P&L, period comparisons, property or tenant details, "
            "trends, or anomalies."
        )
    elif task.intent == Intent.OUT_OF_SCOPE:
        lines.append(
            "That's outside what I can help with. I'm a real-estate asset-management assistant "
            "working from your ledger: ask me about P&L, revenue or expenses for a property, "
            "tenant or period, comparisons between periods, top tenants, or anomalies in the "
            "numbers."
        )
    else:  # CLARIFICATION or anything unexpected
        lines.append(
            "I'm not sure what you'd like me to compute. You can ask, for example: 'What is the "
            "total P&L for 2024?', 'Compare Q1 2025 with Q1 2024', 'Details for Building 17', "
            "'Who are my top tenants?' or 'Is anything unusual in the numbers?'."
        )
        if unknown:
            lines.append(
                "Note: " + " ".join(i.message for i in unknown) + f" Known properties: {props}."
            )
    return SpecialistResult(
        task_id=task.id,
        specialist="fallback",
        answer=" ".join(lines),
        caveats=[i.message for i in task.issues],
    )


def _expense_groups(catalog: DataCatalog) -> list[str]:
    return [g for g in catalog.ledger_groups if g not in ("rental_income", "sales_discounts")]


def _text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    # content blocks (list of dicts) -> concatenate text parts
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def _mention(message: str) -> str:
    # "No property called 'X' in the dataset." -> X
    if "'" in message:
        return message.split("'")[1]
    return message
