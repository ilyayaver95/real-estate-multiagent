"""Runtime configuration, read once from environment variables (and a local .env file).

Everything that differs between a laptop, CI and Streamlit Cloud lives here so the rest of the
code never touches ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    model: str
    data_path: Path
    temperature: float
    max_tool_rounds: int
    request_timeout: float

    @property
    def llm_available(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    """Build settings from the environment. Streamlit secrets are mirrored into env by the app."""
    data_path = Path(os.getenv("REM_DATA_PATH", "data/ledger.parquet"))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        model=os.getenv("REM_MODEL", "gpt-4o-mini"),
        data_path=data_path,
        temperature=float(os.getenv("REM_TEMPERATURE", "0")),
        max_tool_rounds=int(os.getenv("REM_MAX_TOOL_ROUNDS", "6")),
        request_timeout=float(os.getenv("REM_REQUEST_TIMEOUT", "60")),
    )
