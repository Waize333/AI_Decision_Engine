"""
backend/app/schemas/auth.py
============================
PURPOSE:
  Pydantic v2 schemas for authentication endpoints.

  These define the exact JSON shape that:
    - Clients must send  (Request schemas → validate input)
    - Our API returns    (Response schemas → shape output)

WHY PYDANTIC SCHEMAS SEPARATE FROM SQLALCHEMY MODELS?
  SQLAlchemy models know about DB internals (columns, foreign keys).
  Pydantic schemas know about the API contract (what clients see).

  Example: The 'hashed_password' column exists in the DB model,
  but it must NEVER appear in any API response schema.
  Keeping them separate makes this impossible to accidentally expose.

VALIDATION BUILT-IN:
  Pydantic runs validators automatically when data is received:
    - Wrong type → 422 Unprocessable Entity
    - Missing required field → 422
    - Password too short → 422 with custom message
"""

import re
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.models.user import UserRole


# ─── REQUEST SCHEMAS (what clients send) ─────────────────────────────────────

class RegisterRequest(BaseModel):
    """
    Body for POST /auth/register

    email     — must be a valid email format (Pydantic validates this)
    password  — minimum 8 chars, must contain uppercase + digit
    full_name — optional display name
    """
    email: EmailStr = Field(
        ...,
        description="Valid email address. Used as login identifier.",
        examples=["alice@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password: min 8 chars, requires uppercase and digit.",
        examples=["SecurePass1"],
    )
    full_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional display name.",
        examples=["Alice Smith"],
    )

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Enforce a minimum password policy:
          - At least one uppercase letter
          - At least one digit

        WHY HERE (not in the endpoint)?
          Pydantic validators run before the endpoint code ever executes.
          This means the endpoint body never even runs if the password
          is weak — clean separation of validation from business logic.
        """
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        return v

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Lowercase email so 'Alice@Example.com' and 'alice@example.com' match."""
        return v.lower().strip()


class LoginRequest(BaseModel):
    """
    Body for POST /auth/login

    Simple: just email + password.
    FastAPI will auto-validate types and required fields.
    """
    email: EmailStr = Field(..., examples=["alice@example.com"])
    password: str = Field(..., min_length=1, examples=["SecurePass1"])

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.lower().strip()


# ─── RESPONSE SCHEMAS (what the API returns) ─────────────────────────────────

class TokenResponse(BaseModel):
    """
    Response for POST /auth/login (successful)

    access_token  — short-lived JWT (30 min) — include in Authorization header
    refresh_token — long-lived JWT (7 days) — use to get new access token
    token_type    — always "bearer" (OAuth2 standard)
    expires_in    — seconds until access_token expires
    """
    access_token: str = Field(..., examples=["eyJhbGci..."])
    refresh_token: str = Field(..., examples=["eyJhbGci..."])
    token_type: str = Field(default="bearer")
    expires_in: int = Field(..., description="Access token TTL in seconds", examples=[1800])


class UserResponse(BaseModel):
    """
    Safe user representation — NEVER includes hashed_password.

    Used in:
      - Response to /auth/register
      - Response to /auth/me (current user info)
      - Embedded in other responses where user info is needed
    """
    id: str
    email: str
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    is_verified: bool

    # Pydantic v2 config — allows reading from SQLAlchemy ORM objects
    model_config = {"from_attributes": True}


class RegisterResponse(BaseModel):
    """Response for successful registration."""
    message: str = "Account created successfully."
    user: UserResponse


class RefreshRequest(BaseModel):
    """Body for POST /auth/refresh — exchange refresh token for new access token."""
    refresh_token: str = Field(..., description="The refresh token from login.")


class MessageResponse(BaseModel):
    """Generic response for simple success/failure messages."""
    message: str
    success: bool = True
