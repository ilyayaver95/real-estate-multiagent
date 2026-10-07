"""Synthesizer: composes the final reply from specialist results.

Efficiency rule: a single specialist result is returned as-is (no extra LLM call) unless the
verifier asked for a rewrite. Multiple results, or a rewrite, go through one LLM call that is
instructed to use only numbers present in the specialist answers.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rem_agent.agents.prompts import SYNTHESIZER_SYSTEM
from rem_agent.schemas import ResolvedTask, SpecialistResult


def needs_llm(results: list[SpecialistResult]) -> bool:
    return len(results) != 1 or bool(results[0].error)


def synthesize(
    llm: BaseChatModel | None,
    question: str,
    tasks: list[ResolvedTask],
    results: list[SpecialistResult],
    feedback: str | None = None,
) -> str:
    ordered = sorted(results, key=lambda r: _order(tasks, r.task_id))
    if not ordered:
        return "I couldn't produce an answer for that request."
    if not needs_llm(ordered) and feedback is None:
        return ordered[0].answer
    if llm is None:
        return "\n\n".join(_fallback_section(tasks, r) for r in ordered)

    parts = []
    for r in ordered:
        task = next((t for t in tasks if t.id == r.task_id), None)
        head = f"[{r.task_id}] {task.text if task else ''} (specialist: {r.specialist})"
        body = r.answer if not r.error else f"ERROR: {r.error}"
        caveats = ("\nCaveats: " + " | ".join(r.caveats)) if r.caveats else ""
        parts.append(f"{head}\n{body}{caveats}")
    user = f"User message: {question}\n\nSpecialist answers:\n\n" + "\n\n".join(parts)
    if feedback:
        user += f"\n\nREVISION REQUIRED: {feedback}"
    ai = llm.invoke([SystemMessage(content=SYNTHESIZER_SYSTEM), HumanMessage(content=user)])
    text = ai.content if isinstance(ai.content, str) else str(ai.content)
    return text.strip()


def _order(tasks: list[ResolvedTask], task_id: str) -> int:
    for i, t in enumerate(tasks):
        if t.id == task_id:
            return i
    return len(tasks)


def _fallback_section(tasks: list[ResolvedTask], r: SpecialistResult) -> str:
    task = next((t for t in tasks if t.id == r.task_id), None)
    title = task.text if task else r.task_id
    return f"**{title}**\n\n{r.answer}"
