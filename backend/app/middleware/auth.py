"""
backend/app/middleware/auth.py
================================
PURPOSE:
  FastAPI dependency functions for authentication and authorization.

  These are NOT traditional WSGI middleware (they don't wrap the app).
  Instead, they are FastAPI "dependencies" — injected into endpoint
  function signatures so each route declares its own access requirements.

HOW IT WORKS:
  1. Client sends: Authorization: Bearer <jwt_token>
  2. get_current_user() extracts and validates the token
  3. require_role(UserRole.ANALYST) also checks role hierarchy

PATTERN (Dependency Injection):
  @router.get("/metrics")
  async def get_metrics(
      current_user: User = Depends(require_role(UserRole.ANALYST))
  ):
      ...  # Only Analysts, Engineers, and Admins can reach this code

ADVANTAGE:
  Role requirements are declared where the route is defined,
  not hidden in separate middleware files. Self-documenting.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.security import decode_token
from app.db.postgresql.session import get_db
from app.models.user import User, UserRole

logger = get_logger(__name__)

# HTTPBearer parses the "Authorization: Bearer <token>" header.
# auto_error=False means we get None instead of 401 if header is missing
# (so we can give a more informative error message ourselves).
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Core authentication dependency.

    Steps:
      1. Extract token from Authorization header
      2. Decode and validate JWT signature + expiry
      3. Extract user_id from token payload ("sub" claim)
      4. Load user from PostgreSQL by user_id
      5. Verify user is still active (not soft-deleted)

    Raises:
      401 UNAUTHORIZED — if token missing, invalid, or expired
      401 UNAUTHORIZED — if user no longer exists in DB
      403 FORBIDDEN    — if user account is deactivated

    Returns:
      The authenticated User ORM object (safe to use in endpoints)
    """
    # ── Step 1: Check header presence ──────────────────────────────
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Include 'Authorization: Bearer <token>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # ── Step 2: Decode and validate JWT ────────────────────────────
    try:
        payload = decode_token(token)
    except JWTError as e:
        logger.warning("jwt_decode_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Step 3: Extract user ID ─────────────────────────────────────
    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing 'sub' claim.",
        )

    # Token type check — refresh tokens cannot be used as access tokens
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Use your access token, not your refresh token.",
        )

    # ── Step 4: Load user from database ────────────────────────────
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        logger.warning("auth_user_not_found", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token no longer exists.",
        )

    # ── Step 5: Check account is active ────────────────────────────
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Contact support.",
        )

    logger.debug("auth_success", user_id=user.id, role=user.role.value)
    return user


def require_role(required_role: UserRole):
    """
    Role-based access control (RBAC) dependency factory.

    Returns a FastAPI dependency that:
      1. Authenticates the user (via get_current_user)
      2. Checks if their role meets or exceeds the required level

    ROLE HIERARCHY (lowest → highest):
      client (0) < analyst (1) < engineer (2) < admin (3)

    So require_role(UserRole.ANALYST) allows:
      ✅ analyst, engineer, admin
      ❌ client

    USAGE:
      # Only engineers and admins can deploy models
      @router.post("/models/{id}/deploy")
      async def deploy_model(
          model_id: str,
          current_user: User = Depends(require_role(UserRole.ENGINEER))
      ):
          ...
    """
    async def _check_role(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if not current_user.has_role(required_role):
            logger.warning(
                "rbac_denied",
                user_id=current_user.id,
                user_role=current_user.role.value,
                required_role=required_role.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Insufficient permissions. "
                    f"Required: '{required_role.value}' or higher. "
                    f"Your role: '{current_user.role.value}'."
                ),
            )
        return current_user

    return _check_role


# ─── CONVENIENCE ALIASES ─────────────────────────────────────────────────────
# Import these pre-built dependencies in endpoints for cleaner code.
# Instead of: Depends(require_role(UserRole.ADMIN))
# Write:      Depends(require_admin)

require_admin = require_role(UserRole.ADMIN)
require_engineer = require_role(UserRole.ENGINEER)
require_analyst = require_role(UserRole.ANALYST)
require_client = require_role(UserRole.CLIENT)   # any authenticated user
