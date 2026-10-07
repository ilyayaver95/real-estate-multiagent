from __future__ import annotations

from pathlib import Path

import pytest

from rem_agent.data import DataCatalog, LedgerData, load_ledger

DATA = Path(__file__).resolve().parents[1] / "data" / "ledger.parquet"


@pytest.fixture(scope="session")
def ledger() -> LedgerData:
    return load_ledger(DATA)


@pytest.fixture(scope="session")
def catalog(ledger: LedgerData) -> DataCatalog:
    return DataCatalog(ledger)
