"""Deprecated compatibility alias for scripts.load."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.load import load_clean_data  # noqa: E402

warnings.warn(
    "scripts/ingest.py is deprecated; use scripts/load.py.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    load_clean_data()

