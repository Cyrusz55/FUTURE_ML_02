"""Check database connectivity without exposing credentials."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Connect once and return a shell-friendly status code."""
    try:
        from database.db_connection import engine

        with engine.connect():
            print("Database connection succeeded.")
        return 0
    except Exception as exc:
        try:
            from database.db_connection import _database_url

            url = _database_url()
            scheme, _, remainder = url.partition("://")
            host = remainder.rsplit("@", 1)[-1].split("/", 1)[0]
            masked_url = f"{scheme}://***@{host}"
        except (ValueError, ImportError):
            masked_url = "<missing or invalid DATABASE_URL>"
        print(f"Database connection failed for {masked_url}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
