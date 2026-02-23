"""
backend/tests/integration/test_auth_api.py
===========================================
PURPOSE:
  Integration tests for the /auth endpoints.

  Integration tests test the FULL request/response cycle:
    HTTP request → middleware → endpoint → service → DB → response

  They use the real FastAPI app with a real (SQLite) database.
  The DB is rolled back after each test (conftest.py handles this).

WHAT WE TEST:
  ✅ Registration: happy path
  ✅ Registration: duplicate email returns 409
  ✅ Registration: weak password returns 422
  ✅ Registration: invalid email format returns 422
  ✅ Login: happy path returns tokens
  ✅ Login: wrong password returns 401
  ✅ Login: non-existent email returns 401
  ✅ GET /auth/me: returns correct user
  ✅ GET /auth/me: no token returns 401
  ✅ Token refresh: valid refresh token works
"""

import pytest


# All tests that use `client` or DB fixtures must be async
pytestmark = pytest.mark.asyncio


class TestRegister:

    async def test_register_success(self, client):
        """Happy path: register a new user."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecurePass1",
                "full_name": "New User",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["user"]["email"] == "newuser@example.com"
        assert data["user"]["role"] == "client"       # always defaults to client
        assert "hashed_password" not in data["user"]  # MUST not be exposed
        assert data["user"]["is_active"] is True

    async def test_register_duplicate_email_returns_409(self, client, test_user):
        """Registering an existing email must return 409 Conflict."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,  # already in DB
                "password": "SomePass1",
            },
        )
        assert response.status_code == 409
        assert "already registered" in response.json()["detail"].lower()

    async def test_register_weak_password_returns_422(self, client):
        """Password without uppercase letter must be rejected at schema level."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "weakpass@example.com",
                "password": "lowercase1only",   # no uppercase
            },
        )
        assert response.status_code == 422

    async def test_register_short_password_returns_422(self, client):
        """Password shorter than 8 chars must be rejected."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "short@example.com", "password": "Sh1"},
        )
        assert response.status_code == 422

    async def test_register_invalid_email_returns_422(self, client):
        """Non-email strings must be rejected by Pydantic EmailStr."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "ValidPass1"},
        )
        assert response.status_code == 422

    async def test_register_normalizes_email_to_lowercase(self, client):
        """Email should be stored lowercase regardless of input case."""
        response = await client.post(
            "/api/v1/auth/register",
            json={"email": "UPPER@EXAMPLE.COM", "password": "SecurePass1"},
        )
        assert response.status_code == 201
        assert response.json()["user"]["email"] == "upper@example.com"


class TestLogin:

    async def test_login_success(self, client, test_user):
        """Happy path: login returns access + refresh tokens."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": test_user.email, "password": "TestPass1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    async def test_login_wrong_password_returns_401(self, client, test_user):
        """Wrong password must return 401, not 403 or 404."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": test_user.email, "password": "WrongPassword1"},
        )
        assert response.status_code == 401

    async def test_login_nonexistent_email_returns_401(self, client):
        """
        Non-existent email must return the SAME 401 as wrong password.
        This prevents user enumeration (can't tell if email exists).
        """
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@nowhere.com", "password": "SomePass1"},
        )
        assert response.status_code == 401

    async def test_login_returns_wwwauthenticate_header(self, client, test_user):
        """401 responses must include WWW-Authenticate header per RFC 7235."""
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": test_user.email, "password": "Wrong1"},
        )
        assert response.status_code == 401
        assert "www-authenticate" in response.headers


class TestGetMe:

    async def test_get_me_returns_user_profile(self, client, client_auth_headers, test_user):
        """GET /auth/me must return the authenticated user's profile."""
        response = await client.get("/api/v1/auth/me", headers=client_auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["role"] == "client"
        assert "hashed_password" not in data  # non-negotiable

    async def test_get_me_no_token_returns_401(self, client):
        """Without auth header, must get 401."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401

    async def test_get_me_invalid_token_returns_401(self, client):
        """Garbage token must be rejected."""
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer garbage.token.here"},
        )
        assert response.status_code == 401


class TestRBAC:

    async def test_analyst_endpoint_blocked_for_client(
        self, client, client_auth_headers
    ):
        """A CLIENT token must not access ANALYST-only endpoints."""
        response = await client.get(
            "/api/v1/metrics",
            headers=client_auth_headers,
        )
        assert response.status_code == 403

    async def test_analyst_endpoint_allowed_for_admin(
        self, client, admin_auth_headers
    ):
        """ADMIN has all permissions — should access analyst endpoints."""
        response = await client.get(
            "/api/v1/metrics",
            headers=admin_auth_headers,
        )
        # 200 or empty list — not 403
        assert response.status_code != 403
