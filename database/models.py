"""Database table definitions for the source ticket fields."""

from __future__ import annotations

from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative metadata base."""


class Ticket(Base):
    """Ticket fields from the processed source CSV."""

    __tablename__ = "tickets"

    ticket_id = Column(Integer, primary_key=True, autoincrement=False)
    document = Column("Document", Text, nullable=False)
    topic_group = Column("Topic_group", Text, nullable=False)


def create_tables(engine) -> None:
    """Create the ticket table when it does not already exist."""
    Base.metadata.create_all(engine)

