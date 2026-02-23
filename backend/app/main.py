"""
backend/app/main.py
====================
PURPOSE:
  The application entry point — creates and configures the FastAPI app.

  This file is responsible for:
    1. Creating the FastAPI application instance
    2. Registering all middleware (CORS, rate limiting, request tracing)
    3. Connecting to all databases at startup
    4. Disconnecting from databases on shutdown
    5. Attaching all API routes
    6. Configuring Prometheus metrics instrumentation

STARTUP SEQUENCE:
  uvicorn app.main:app
    → FastAPI app created
    → lifespan context starts
    → databases connected
    → indexes created
    → app starts serving requests

SHUTDOWN SEQUENCE (Ctrl+C / SIGTERM):
  → lifespan context exits
  → database connections closed
  → process exits cleanly (no dangling connections)

WHY LIFESPAN (not @app.on_event)?
  The newer `lifespan` context manager pattern:
    - Cleaner: startup and shutdown in one function
    - Async-safe: resources created in async context
    - FastAPI recommends it over deprecated on_event handlers
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.mongodb.client import close_mongo_connection, create_indexes
from app.db.redis.client import close_redis_connection
from app.middleware.rate_limit import RateLimitMiddleware

# Configure logging FIRST — before any other imports that log
configure_logging()
logger = get_logger(__name__)


# ─── LIFESPAN CONTEXT ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifecycle: startup and shutdown.

    Code BEFORE yield → runs at startup
    Code AFTER yield  → runs at shutdown

    This is where we:
      - Establish DB connections (once, not per request)
      - Create MongoDB indexes (idempotent — safe to re-run)
      - Log readiness
    """
    # ── STARTUP ────────────────────────────────────────────────────
    logger.info("app_starting", name=settings.APP_NAME, env=settings.APP_ENV)

    # Create MongoDB indexes for fast queries
    # (safe to call on every start — MongoDB ignores existing indexes)
    try:
        await create_indexes()
        logger.info("mongodb_indexes_ready")
    except Exception as e:
        logger.error("mongodb_index_creation_failed", error=str(e))

    logger.info(
        "app_ready",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        docs_url=f"http://{settings.APP_HOST}:{settings.APP_PORT}/docs",
    )

    yield  # ← app is alive and serving requests between startup and shutdown

    # ── SHUTDOWN ───────────────────────────────────────────────────
    logger.info("app_shutting_down")
    await close_mongo_connection()
    await close_redis_connection()
    logger.info("app_shutdown_complete")


# ─── FASTAPI APP ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "**AI Decision Engine Platform** — production-grade ML inference API.\n\n"
        "## Features\n"
        "- 🧠 Live model inference with confidence scores\n"
        "- 🔐 JWT authentication with role-based access control\n"
        "- 📊 Drift detection, A/B testing, and model versioning\n"
        "- 💬 Feedback loop for continuous model improvement\n"
        "- 📈 Prometheus metrics + structured JSON logging\n\n"
        "## Authentication\n"
        "All protected endpoints require: `Authorization: Bearer <jwt_token>`\n\n"
        "Get your token at `POST /api/v1/auth/login`"
    ),
    version="1.0.0",
    docs_url="/docs",           # Swagger UI at /docs
    redoc_url="/redoc",         # ReDoc UI at /redoc
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "AI Decision Engine Team",
        "url": "https://github.com/your-org/ai-decision-engine",
    },
    license_info={
        "name": "MIT",
    },
)


# ─── MIDDLEWARE ────────────────────────────────────────────────────────────────
# Middleware is executed in REVERSE order of registration.
# Last added = first to run on incoming requests.

# 1. CORS — must be outermost middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-Request-ID",
    ],
)

# 2. Rate limiting — checks Redis before routing to any endpoint
app.add_middleware(RateLimitMiddleware)


# ─── REQUEST ID MIDDLEWARE ────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Inject a unique X-Request-ID into every request/response.

    WHY?
      When debugging production issues, you can grep logs for the
      request ID to trace the entire lifecycle of a single request
      across all log lines, even across multiple services.

    The client can supply their own request ID via the header,
    or we generate one automatically.
    """
    request_id = request.headers.get("X-Request-ID") or f"rid-{uuid.uuid4().hex[:16]}"

    # Make ID available to route handlers via request.state
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ─── GLOBAL ERROR HANDLER ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exception and return a clean JSON error response.
    Without this, FastAPI returns HTML error pages, which break API clients.

    In production, this should also alert your on-call team.
    """
    logger.error(
        "unhandled_exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Our team has been notified.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ─── ROUTES ───────────────────────────────────────────────────────────────────

# Mount all v1 API routes under /api/v1
app.include_router(api_router, prefix="/api/v1")


# ─── ROOT ENDPOINT ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect hint for the root path."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ─── RUN DIRECTLY ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.is_development,   # auto-restart on code changes in dev
        log_level=settings.LOG_LEVEL.lower(),
    )
