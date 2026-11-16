"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH, encoding="utf-8-sig")


def _database_url() -> str:
    import os

    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is missing from .env.")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def get_engine() -> Engine:
    """Create the configured SQLAlchemy engine."""
    url = _database_url()
    connect_args = {"connect_timeout": 5} if url.startswith("postgresql+") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


engine = get_engine()


@contextmanager
def get_session() -> Iterator[Session]:
    """Provide a transaction-scoped session."""
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise

