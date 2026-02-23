"""
backend/app/db/postgresql/session.py
=====================================
PURPOSE:
  Creates and manages the async PostgreSQL connection pool using
  SQLAlchemy's async engine.

KEY CONCEPTS:
  engine       — the low-level connection pool. Think of it as the
                 "phone line" to the database. Only ONE engine per app.

  AsyncSession  — a single conversation with the database. Short-lived.
                  Created per-request, closed when the request ends.

  get_db()     — a FastAPI dependency that yields one session per
                 request and guarantees cleanup (rollback on error,
                 close always).

USAGE IN ENDPOINTS:
  from app.db.postgresql.session import get_db
  from sqlalchemy.ext.asyncio import AsyncSession

  @router.post("/users")
  async def create_user(db: AsyncSession = Depends(get_db)):
      # use db here
      ...
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── ENGINE ──────────────────────────────────────────────────────────────────
# The engine is the connection pool — expensive to create, so we create
# it ONCE at module import time and reuse it for the lifetime of the process.
#
# echo=False in production (SQL is verbose and expensive to log).
# pool_pre_ping=True — test connections before using them. Prevents errors
#   when the DB restarts or the connection times out.

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.is_development,       # log SQL in dev, not in prod
    pool_pre_ping=True,
    pool_size=10,                       # keep 10 connections open
    max_overflow=20,                    # allow 20 more during traffic spikes
    pool_recycle=3600,                  # recycle connections every 1 hour
)

# ─── SESSION FACTORY ─────────────────────────────────────────────────────────
# async_sessionmaker creates AsyncSession instances.
# expire_on_commit=False — after a commit, ORM objects remain accessible
#   in the same request without needing to reload from DB.

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ─── DEPENDENCY ──────────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides one database session per request.

    Pattern:
        1. Create a new session
        2. Yield it to the endpoint handler
        3. If a DB error occurs → rollback (undo partial changes)
        4. Always close the session when the request finishes

    This ensures no half-written data and no connection leaks.

    Usage:
        @router.get("/users/{id}")
        async def get_user(
            user_id: str,
            db: AsyncSession = Depends(get_db)
        ):
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("db_session_error", error=str(e), exc_info=True)
            raise
        finally:
            await session.close()
