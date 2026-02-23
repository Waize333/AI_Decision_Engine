"""
backend/app/models/base.py
===========================
PURPOSE:
  Defines the SQLAlchemy declarative base that ALL models inherit from.
  Also defines a TimestampMixin — shared columns (created_at, updated_at)
  added to every table automatically.

WHY A BASE CLASS?
  Every database table needs:
    - A primary key (id)
    - created_at / updated_at timestamps

  Rather than copy-pasting these into every model, we define them
  once in a mixin and inherit.

USAGE:
  from app.models.base import Base, TimestampMixin

  class User(Base, TimestampMixin):
      __tablename__ = "users"
      # only define columns specific to User here
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base.

    All ORM models inherit from this. SQLAlchemy uses it to:
      - Track all model classes
      - Generate CREATE TABLE statements (via Alembic)
      - Provide the metadata object for migrations
    """
    pass


class UUIDMixin:
    """
    Adds a UUID primary key column named 'id' to any model.

    WHY UUID INSTEAD OF INTEGER?
      - UUIDs are globally unique — safe to generate in the app layer
        without checking the DB first (no race conditions)
      - Hard to enumerate/guess (security benefit)
      - Works across distributed systems (can merge records safely)
    """
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )


class TimestampMixin:
    """
    Adds created_at and updated_at columns to any model.

    server_default=func.now() — the DB sets the value, not Python.
    onupdate=func.now()      — auto-updated on every UPDATE statement.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
