"""Extractor agent: pulls properties, tenants, timeframes and account terms out of each
sub-question (one LLM call for all sub-questions)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rem_agent.agents.prompts import EXTRACTOR_SYSTEM
from rem_agent.data.catalog import DataCatalog
from rem_agent.llm import structured
from rem_agent.schemas import ExtractedSlots, ExtractionOutput, SubQuestion


def extract(
    llm: BaseChatModel,
    catalog: DataCatalog,
    sub_questions: list[SubQuestion],
    original_message: str | None = None,
) -> dict[str, ExtractedSlots]:
    listing = "\n".join(f"- [{sq.id}] ({sq.intent.value}) {sq.text}" for sq in sub_questions)
    original = (
        f"Original user message (authoritative wording): {original_message}\n\n"
        if original_message
        else ""
    )
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM.format(catalog=catalog.describe_for_prompt())),
        HumanMessage(
            content=f"{original}Sub-questions:\n{listing}\n\nReturn one slots entry per id."
        ),
    ]
    result = structured(llm, ExtractionOutput, messages)
    by_id = {s.sub_question_id: s for s in result.slots}
    # Guarantee one entry per sub-question even if the model skipped one.
    return {sq.id: by_id.get(sq.id, ExtractedSlots(sub_question_id=sq.id)) for sq in sub_questions}
