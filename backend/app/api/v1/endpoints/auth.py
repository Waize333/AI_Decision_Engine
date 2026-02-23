"""
backend/app/api/v1/endpoints/auth.py
======================================
PURPOSE:
  FastAPI route handlers for authentication.
  Each function handles one HTTP endpoint.

RESPONSIBILITIES (thin layer):
  1. Parse + validate incoming JSON (Pydantic does this automatically)
  2. Call the appropriate service method
  3. Return a formatted response
  4. Handle errors (convert ValueError → HTTP 400)

WHAT ENDPOINTS DON'T DO:
  - Business logic (that's in services/)
  - DB queries (that's in services/ too)
  - Password hashing (that's in core/security.py)
  - If you find yourself writing `if user.role == ...` here, stop.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgresql.session import get_db
from app.middleware.auth import get_current_user
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
    UserResponse,
)
from app.services.auth import AuthService

# APIRouter groups related endpoints under a common prefix.
# The prefix ("/auth") and tags (["Authentication"]) are added
# when this router is registered in the main router file.
router = APIRouter()


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new account",
    description=(
        "Create a new user account. New accounts are assigned the **CLIENT** role. "
        "An admin must manually elevate roles after registration."
    ),
)
async def register(
    body: RegisterRequest,           # ← Pydantic validates + parses the JSON body
    db: AsyncSession = Depends(get_db),  # ← DB session injected by FastAPI
):
    """
    POST /auth/register

    Request body: { email, password, full_name? }
    Response:     { message, user: { id, email, role, ... } }

    Errors:
      409 Conflict — email already registered
      422 Unprocessable — validation failed (weak password, bad email)
    """
    try:
        user = await AuthService.register(data=body, db=db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    return RegisterResponse(
        message="Account created successfully.",
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get JWT tokens",
    description=(
        "Authenticate with email + password. "
        "Returns a short-lived **access token** (30 min) and "
        "a long-lived **refresh token** (7 days). "
        "Include the access token as: `Authorization: Bearer <token>`"
    ),
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    POST /auth/login

    Request body: { email, password }
    Response:     { access_token, refresh_token, token_type, expires_in }

    Errors:
      401 Unauthorized — invalid credentials (intentionally vague)
      403 Forbidden    — account deactivated
    """
    try:
        tokens = await AuthService.login(data=body, db=db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return tokens


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description=(
        "Exchange a valid refresh token for a new access + refresh token pair. "
        "Use this to keep users logged in without re-entering their password."
    ),
)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    POST /auth/refresh

    Request body: { refresh_token }
    Response:     { access_token, refresh_token, token_type, expires_in }
    """
    try:
        tokens = await AuthService.refresh_access_token(
            refresh_token=body.refresh_token, db=db
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    return tokens


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Returns the profile of the currently authenticated user.",
)
async def get_me(
    current_user: User = Depends(get_current_user),   # ← enforces authentication
):
    """
    GET /auth/me

    Headers required: Authorization: Bearer <access_token>
    Response:         { id, email, full_name, role, is_active, is_verified }

    Errors:
      401 Unauthorized — missing/invalid token
    """
    return UserResponse.model_validate(current_user)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout (client-side)",
    description=(
        "Stateless logout. Since JWTs are stateless, the server can't invalidate them. "
        "The client must delete the tokens. "
        "For production, implement a token blocklist in Redis."
    ),
)
async def logout(
    current_user: User = Depends(get_current_user),
):
    """
    POST /auth/logout

    NOTE: This is a stateless logout. The token remains cryptographically
    valid until it expires. True revocation requires a Redis blocklist
    (Phase 8 enhancement).
    """
    return MessageResponse(
        message="Logged out successfully. Please delete your tokens client-side.",
        success=True,
    )
