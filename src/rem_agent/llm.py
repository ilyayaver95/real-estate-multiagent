"""Chat-model factory.

One place decides which model is used and how. Structured outputs use LangChain's
``with_structured_output`` (OpenAI function calling under the hood), which gives us validated
Pydantic objects instead of free text to parse.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from rem_agent.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


def get_chat_model(settings: Settings | None = None, **overrides) -> BaseChatModel:
    settings = settings or get_settings()
    if not settings.llm_available:
        raise LLMUnavailable(
            "OPENAI_API_KEY is not set. Add it to .env (local) or Streamlit secrets (cloud)."
        )
    from langchain_openai import ChatOpenAI

    params = {
        "model": settings.model,
        "temperature": settings.temperature,
        "timeout": settings.request_timeout,
        "max_retries": 2,
        "api_key": settings.openai_api_key,
    }
    params.update(overrides)
    return ChatOpenAI(**params)


def structured(llm: BaseChatModel, schema: type[T], messages: list) -> T:
    """Invoke ``llm`` and return a validated ``schema`` instance (never None)."""
    runnable = llm.with_structured_output(schema)
    result = runnable.invoke(messages)
    if result is None:
        raise ValueError(f"Model returned no structured output for {schema.__name__}")
    if isinstance(result, dict):  # some providers return dicts
        result = schema.model_validate(result)
    return result
