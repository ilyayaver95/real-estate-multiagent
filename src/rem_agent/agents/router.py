"""Router agent: intent detection + decomposition of compound requests (one LLM call)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rem_agent.agents.prompts import ROUTER_SYSTEM
from rem_agent.data.catalog import DataCatalog
from rem_agent.llm import structured
from rem_agent.schemas import Intent, RouterOutput, SubQuestion

MAX_HISTORY_TURNS = 6


def history_messages(history: list[dict]) -> list:
    """Convert [{'role': 'user'|'assistant', 'content': str}] into LangChain messages."""
    out = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        if turn.get("role") == "user":
            out.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            out.append(AIMessage(content=turn["content"]))
    return out


def route(
    llm: BaseChatModel, catalog: DataCatalog, question: str, history: list[dict] | None = None
) -> RouterOutput:
    messages = [SystemMessage(content=ROUTER_SYSTEM.format(catalog=catalog.describe_for_prompt()))]
    messages += history_messages(history or [])
    messages.append(HumanMessage(content=question))
    result = structured(llm, RouterOutput, messages)
    return _sanitise(result, question)


def _sanitise(result: RouterOutput, question: str) -> RouterOutput:
    """Defensive clean-up so downstream nodes can trust the shape."""
    if not result.sub_questions:
        result.sub_questions = [
            SubQuestion(id="q1", text=question, intent=Intent.CLARIFICATION, rationale="empty")
        ]
    seen: set[str] = set()
    for i, sq in enumerate(result.sub_questions, start=1):
        if not sq.id or sq.id in seen:
            sq.id = f"q{i}"
        seen.add(sq.id)
        sq.text = sq.text.strip() or question
    if result.needs_clarification and not result.clarification_question:
        result.clarification_question = (
            "Could you tell me which property, tenant or period you mean?"
        )
    return result
