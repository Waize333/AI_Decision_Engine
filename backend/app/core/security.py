"""
backend/app/core/security.py
============================
PURPOSE:
  All cryptographic/security utilities live here:
    - Password hashing (bcrypt)
    - Password verification
    - JWT access token creation
    - JWT token decoding and validation

WHY SEPARATE FILE?
  Security logic is reused by:
    - /auth/register (hash password before saving)
    - /auth/login (verify password, create token)
    - Auth middleware (decode token on every request)
  Keeping it here avoids circular imports.

CONCEPTS:
  bcrypt  — a one-way hashing algorithm. You can never recover the
            original password from the hash. At login, you hash the
            user's input and compare hashes.

  JWT     — JSON Web Token. A signed, compact, URL-safe string that
            encodes claims (e.g. user_id, role, expiry). The server
            signs it with SECRET_KEY — so it cannot be forged.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# ─── PASSWORD HASHING ────────────────────────────────────────────────────────

# CryptContext manages the hashing algorithm.
# schemes=["bcrypt"] means we use bcrypt.
# deprecated="auto" means if we ever switch algorithms, old passwords
# are automatically re-hashed on next login.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    Called during user registration before storing in the database.
    The output looks like: $2b$12$... (60 characters)

    Args:
        plain_password: The raw password from the registration form

    Returns:
        A bcrypt hash string (safe to store in DB)
    """
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a stored bcrypt hash.

    Called during login. We NEVER store or compare plain passwords.

    Args:
        plain_password:  What the user typed at login
        hashed_password: What we stored in the DB at registration

    Returns:
        True if the password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


# ─── JWT TOKENS ──────────────────────────────────────────────────────────────

def create_access_token(
    subject: str | Any,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[dict] = None,
) -> str:
    """
    Create a signed JWT access token.

    The token encodes:
      - sub  (subject)  — typically the user's UUID or email
      - exp  (expiry)   — when this token stops being valid
      - iat  (issued at) — when this token was created
      - any extra_claims you pass (e.g. {"role": "admin"})

    Args:
        subject:      String identifier for the user (usually user_id)
        expires_delta: How long until expiry. Defaults to 30 minutes.
        extra_claims: Additional data to embed in the token payload.

    Returns:
        A JWT string like "eyJhbGci..."
    """
    now = datetime.now(timezone.utc)

    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": now + expires_delta,
        "type": "access",
    }

    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def create_refresh_token(subject: str | Any) -> str:
    """
    Create a long-lived refresh token.

    Refresh tokens are used to obtain new access tokens without
    requiring the user to log in again. They live longer (days)
    but should be stored securely (HttpOnly cookies or encrypted storage).
    """
    expires_delta = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(subject),
        "iat": now,
        "exp": now + expires_delta,
        "type": "refresh",
    }

    return jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Raises:
        jose.JWTError — if token is expired, malformed, or signature invalid.
            The caller (auth middleware) should catch this and return 401.

    Returns:
        The decoded payload dict, e.g.:
        {"sub": "user-uuid", "role": "admin", "exp": 1234567890, ...}
    """
    return jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )


def extract_user_id(token: str) -> Optional[str]:
    """
    Convenience helper: decode token and return just the user ID.

    Returns None if the token is invalid (instead of raising).
    Use decode_token() directly when you want the full payload.
    """
    try:
        payload = decode_token(token)
        return payload.get("sub")
    except JWTError:
        return None
