"""
backend/alembic/env.py
========================
PURPOSE:
  Alembic's environment configuration file.
  This tells Alembic how to connect to your database and which
  models to track for auto-generating migrations.

HOW MIGRATIONS WORK:
  1. You change a SQLAlchemy model (add a column, rename a table)
  2. Run: alembic revision --autogenerate -m "add column X to users"
  3. Alembic diffs current models vs the DB schema
  4. Generates a migration file in alembic/versions/
  5. Run: alembic upgrade head → applies the migration to the DB

WHY NOT JUST USE CREATE_ALL?
  create_all() creates tables fresh — it can't ALTER existing tables.
  Alembic generates safe ALTER TABLE statements that work on
  production databases with live data.
"""

import asyncio
from logging.config import fileConfig
from typing import Optional

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ── Import all models so Alembic can detect them ──────────────────
# This is why models/__init__.py matters — one import gets everything
from app.models import Base  # noqa: F401  (import registers all models)
from app.core.config import settings

# Alembic config object
config = context.config

# Use Python logging config from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the metadata Alembic uses to auto-detect schema changes
target_metadata = Base.metadata

# Override the DB URL from our Settings (not alembic.ini)
# This means one source of truth for the connection string
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Used to generate SQL scripts without connecting to the DB.
    Useful for reviewing migrations before applying them.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations using the async SQLAlchemy engine.
    Required because we use asyncpg (an async PostgreSQL driver).
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,   # don't keep connections open after migration
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connected to DB)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
