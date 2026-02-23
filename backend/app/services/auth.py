"""
backend/app/services/auth.py
==============================
PURPOSE:
  Business logic for authentication:
    - Register new users (hash password, save to DB)
    - Login (verify password, issue JWT tokens)
    - Refresh tokens
    - Get current user profile

WHY A SERVICE LAYER?
  Endpoints should be thin — just parse HTTP, call service, return response.
  Services hold the real logic. This separation means:
    - You can call service functions from CLI scripts, background jobs, tests
      without spinning up an HTTP server
    - Logic is testable without mocking HTTP

LAYER FLOW:
  HTTP Request
    → endpoint (validates schema, calls service)
    → service (business logic, DB calls)
    → repository / ORM (raw DB operations)
    → response
"""

from datetime import timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User, UserRole
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse

logger = get_logger(__name__)


class AuthService:
    """
    Handles all authentication business logic.
    Methods are async to support non-blocking DB operations.
    """

    @staticmethod
    async def register(
        data: RegisterRequest,
        db: AsyncSession,
    ) -> User:
        """
        Register a new user account.

        Steps:
          1. Check if email already exists (unique constraint)
          2. Hash the password (bcrypt — never store plain text)
          3. Create User ORM instance
          4. Add to DB session + flush (gets assigned DB-side defaults)

        Raises:
          ValueError — if email is already registered

        Returns:
          The newly created User object (before commit — caller commits)
        """
        # ── 1. Check email uniqueness ──────────────────────────────
        existing = await db.execute(
            select(User).where(User.email == data.email)
        )
        if existing.scalar_one_or_none():
            raise ValueError(f"Email '{data.email}' is already registered.")

        # ── 2. Hash password ───────────────────────────────────────
        hashed = hash_password(data.password)

        # ── 3. Create user ─────────────────────────────────────────
        user = User(
            email=data.email,
            hashed_password=hashed,
            full_name=data.full_name,
            role=UserRole.CLIENT,     # New users always start as CLIENT
            is_active=True,
            is_verified=False,        # Email verification can be added later
        )

        # ── 4. Persist ─────────────────────────────────────────────
        db.add(user)
        await db.flush()   # assigns user.id without committing the transaction
                           # caller (get_db dependency) will commit

        logger.info("user_registered", user_id=user.id, email=user.email)
        return user

    @staticmethod
    async def login(
        data: LoginRequest,
        db: AsyncSession,
    ) -> TokenResponse:
        """
        Authenticate a user and issue JWT tokens.

        Steps:
          1. Look up user by email
          2. Verify password (bcrypt comparison)
          3. Check account is active
          4. Generate access + refresh tokens with role embedded
          5. Return token pair

        Raises:
          ValueError — if credentials invalid or account inactive

        Security note:
          We return the same error message for "email not found" and
          "wrong password" — this prevents user enumeration attacks
          (attacker can't tell which accounts exist).
        """
        INVALID_CREDENTIALS_MSG = (
            "Invalid email or password. Please check your credentials."
        )

        # ── 1. Find user ───────────────────────────────────────────
        result = await db.execute(
            select(User).where(User.email == data.email)
        )
        user = result.scalar_one_or_none()

        # ── 2. Verify password ─────────────────────────────────────
        if not user or not verify_password(data.password, user.hashed_password):
            logger.warning("login_failed", email=data.email)
            raise ValueError(INVALID_CREDENTIALS_MSG)

        # ── 3. Check account status ────────────────────────────────
        if not user.is_active:
            raise ValueError("Your account has been deactivated. Contact support.")

        # ── 4. Issue tokens ────────────────────────────────────────
        # Embed role in the token so middleware doesn't need a DB call
        # to check permissions on every request
        extra_claims = {"role": user.role.value, "email": user.email}

        access_token = create_access_token(
            subject=user.id,
            extra_claims=extra_claims,
        )
        refresh_token = create_refresh_token(subject=user.id)

        logger.info("user_logged_in", user_id=user.id, role=user.role.value)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    @staticmethod
    async def refresh_access_token(
        refresh_token: str,
        db: AsyncSession,
    ) -> TokenResponse:
        """
        Exchange a valid refresh token for a new access token.

        The refresh token is long-lived (7 days). The new access token
        is short-lived (30 min). This keeps users logged in without
        re-entering credentials, while limiting the blast radius of a
        stolen access token.

        Raises:
          ValueError — if refresh token is invalid or expired
        """
        from jose import JWTError

        try:
            payload = decode_token(refresh_token)
        except JWTError:
            raise ValueError("Invalid or expired refresh token. Please login again.")

        if payload.get("type") != "refresh":
            raise ValueError("Token type mismatch. Provide a refresh token.")

        user_id = payload.get("sub")
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise ValueError("User not found or account deactivated.")

        extra_claims = {"role": user.role.value, "email": user.email}
        new_access_token = create_access_token(subject=user.id, extra_claims=extra_claims)
        new_refresh_token = create_refresh_token(subject=user.id)

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
