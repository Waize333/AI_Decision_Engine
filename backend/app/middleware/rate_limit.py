"""
backend/app/middleware/rate_limit.py
======================================
PURPOSE:
  FastAPI middleware that enforces rate limits on every request.

  Uses the Redis sliding window rate limiter from db/redis/client.py.
  Adds X-RateLimit-* headers to responses so clients know their quota.

HOW IT WORKS:
  - Every incoming request:
      1. Identify the client (by user_id if authenticated, else IP)
      2. Check rate limit in Redis
      3. If over limit → 429 Too Many Requests
      4. If under limit → let request proceed, add quota headers

WHY MIDDLEWARE vs. DEPENDENCY?
  Dependencies run per-route (you explicitly add Depends(...)).
  Middleware runs on EVERY request automatically — you can't forget it.
  Rate limiting should be universal, so middleware is the right choice.
"""

import time
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import get_logger
from app.db.redis.client import RateLimiter, get_redis

logger = get_logger(__name__)

# Paths that bypass rate limiting entirely
EXEMPT_PATHS = {
    "/health",           # health check (called by load balancers every few seconds)
    "/metrics",          # Prometheus scrape endpoint
    "/docs",             # Swagger UI
    "/redoc",            # ReDoc UI
    "/openapi.json",     # OpenAPI schema
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limit middleware.

    Adds these headers to every non-exempt response:
      X-RateLimit-Limit     — max requests allowed in the window
      X-RateLimit-Remaining — how many requests client has left
      X-RateLimit-Reset     — Unix timestamp when the window resets
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:

        # ── Skip exempt paths ──────────────────────────────────────
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # ── Identify client ────────────────────────────────────────
        # Prefer user_id (injected by auth on authenticated routes),
        # fall back to IP address for unauthenticated endpoints.
        client_id = self._get_client_id(request)

        # ── Check rate limit ───────────────────────────────────────
        try:
            redis = await get_redis()
            is_limited = await RateLimiter.is_rate_limited(redis, client_id)
            remaining = await RateLimiter.get_remaining(redis, client_id)
        except Exception as e:
            # Redis is down — log the error but DON'T block traffic.
            # Failing open is safer than taking the whole API down.
            logger.error("rate_limit_redis_error", error=str(e))
            return await call_next(request)

        # ── Build rate limit headers ───────────────────────────────
        reset_time = int(time.time()) + settings.RATE_LIMIT_WINDOW_SECONDS
        rate_limit_headers = {
            "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(reset_time),
        }

        # ── Reject if over limit ───────────────────────────────────
        if is_limited:
            logger.warning("rate_limit_exceeded", client_id=client_id)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"You have exceeded {settings.RATE_LIMIT_REQUESTS} requests "
                        f"per {settings.RATE_LIMIT_WINDOW_SECONDS} seconds. "
                        f"Try again after {reset_time}."
                    ),
                    "retry_after": settings.RATE_LIMIT_WINDOW_SECONDS,
                },
                headers=rate_limit_headers,
            )

        # ── Allow request through ──────────────────────────────────
        response = await call_next(request)

        # Attach rate limit headers to the actual response
        for key, value in rate_limit_headers.items():
            response.headers[key] = value

        return response

    @staticmethod
    def _get_client_id(request: Request) -> str:
        """
        Determine a unique identifier for the client making this request.

        Priority:
          1. X-User-ID header — set by auth middleware after token validation
          2. X-Forwarded-For  — real IP when behind a reverse proxy (Nginx, AWS ALB)
          3. request.client.host — direct connection IP (local dev)
        """
        # Check if a previously-run auth step injected the user ID
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"

        # Fall back to IP-based limiting
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # "X-Forwarded-For: client, proxy1, proxy2" — take first (real client)
            ip = forwarded_for.split(",")[0].strip()
            return f"ip:{ip}"

        # Direct connection
        if request.client:
            return f"ip:{request.client.host}"

        return "ip:unknown"
