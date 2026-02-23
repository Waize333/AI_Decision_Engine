"""
backend/tests/conftest.py
==========================
PURPOSE:
  pytest configuration and shared fixtures.

  A "fixture" is a piece of setup code that pytest injects into
  test functions. Instead of copy-pasting DB setup in every test,
  define it once here as a fixture.

KEY FIXTURES:
  client         — a test HTTP client (no real server needed)
  db_session     — an isolated DB transaction, rolled back after each test
  test_user      — a pre-created CLIENT user (for auth tests)
  admin_user     — a pre-created ADMIN user
  auth_headers   — JWT token headers ready to paste into requests

ISOLATION STRATEGY:
  Each test runs inside a DB transaction that is ROLLED BACK when
  the test finishes. This means:
    - Tests don't interfere with each other
    - No cleanup code needed
    - Tests are fast (no actual row deletions)
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.security import create_access_token, hash_password
from app.db.postgresql.session import get_db
from app.main import app
from app.models.base import Base
from app.models.model_version import ModelStatus, ModelVersion
from app.models.user import User, UserRole

# ── Use SQLite for tests (no Docker needed) ────────────────────────────────────
# SQLite is an in-memory database — perfect for tests.
# asyncpg (our production driver) doesn't work with SQLite,
# so we use aiosqlite instead for tests only.
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """
    Use one event loop for the entire test session.
    Required for session-scoped async fixtures.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """
    Create the test database engine once for the whole test session.
    In-memory SQLite — tables created fresh, torn down after all tests.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},  # SQLite threading setting
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provides an isolated DB session per test.

    Uses a SAVEPOINT (nested transaction) so each test runs in its own
    sub-transaction that gets rolled back regardless of test outcome.
    This means no data bleeds between tests.
    """
    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)

    async with session_factory() as session:
        async with session.begin():
            yield session
            await session.rollback()  # ← ALWAYS roll back after each test


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP test client that uses the test DB session.

    Overrides the app's get_db dependency so all endpoints
    use the isolated test session (not the real DB).
    """
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ─── USER FIXTURES ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """A standard CLIENT user for testing auth-required endpoints."""
    user = User(
        email="testclient@example.com",
        hashed_password=hash_password("TestPass1"),
        full_name="Test Client",
        role=UserRole.CLIENT,
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """An ADMIN user for testing privileged endpoints."""
    user = User(
        email="admin@example.com",
        hashed_password=hash_password("AdminPass1"),
        full_name="Test Admin",
        role=UserRole.ADMIN,
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def analyst_user(db_session: AsyncSession) -> User:
    """An ANALYST user for testing metrics endpoints."""
    user = User(
        email="analyst@example.com",
        hashed_password=hash_password("AnalystPass1"),
        full_name="Test Analyst",
        role=UserRole.ANALYST,
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.flush()
    return user


# ─── AUTH HEADER FIXTURES ─────────────────────────────────────────────────────

@pytest.fixture
def client_auth_headers(test_user: User) -> dict:
    """JWT Authorization headers for the test CLIENT user."""
    token = create_access_token(
        subject=test_user.id,
        extra_claims={"role": test_user.role.value},
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(admin_user: User) -> dict:
    """JWT Authorization headers for the test ADMIN user."""
    token = create_access_token(
        subject=admin_user.id,
        extra_claims={"role": admin_user.role.value},
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def analyst_auth_headers(analyst_user: User) -> dict:
    """JWT Authorization headers for the test ANALYST user."""
    token = create_access_token(
        subject=analyst_user.id,
        extra_claims={"role": analyst_user.role.value},
    )
    return {"Authorization": f"Bearer {token}"}


# ─── MODEL FIXTURES ───────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def active_model_version(db_session: AsyncSession) -> ModelVersion:
    """An ACTIVE model version — required for inference to work."""
    model = ModelVersion(
        name="test_fraud_detector",
        version_tag="v1",
        description="Test model version",
        artifact_path="./models/nonexistent.joblib",  # uses stub predictor
        status=ModelStatus.ACTIVE,
        traffic_percentage=100,
        feature_schema={"amount": "float", "hour": "int"},
    )
    db_session.add(model)
    await db_session.flush()
    return model
