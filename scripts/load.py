"""Load cleaned tickets into the configured database."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import Integer, Text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_connection import engine  # noqa: E402
from database.models import create_tables  # noqa: E402

TARGET_TABLE = "tickets"
CHUNKSIZE = 1000
CLEAN_PATH = PROJECT_ROOT / "raw_data" / "clean" / "tickets_clean.csv"


def _sql_dtypes() -> dict[str, Text]:
    """Return SQL types for the source-compatible table columns."""
    return {"ticket_id": Integer(), "Document": Text(), "Topic_group": Text()}


def load_clean_data() -> None:
    """Load cleaned rows, preserving the original source column names."""
    create_tables(engine)
    for chunk in pd.read_csv(CLEAN_PATH, chunksize=CHUNKSIZE):
        source_columns = chunk.rename(columns={"text": "Document", "label": "Topic_group"})
        source_columns[["ticket_id", "Document", "Topic_group"]].to_sql(
            TARGET_TABLE,
            con=engine,
            if_exists="append",
            index=False,
            dtype=_sql_dtypes(),
        )


if __name__ == "__main__":
    load_clean_data()

