"""
backend/app/api/v1/router.py
==============================
PURPOSE:
  Assembles all endpoint routers into one v1 API router.
  This is a "switchboard" — it registers every sub-router with its
  URL prefix and Swagger tags.

WHY VERSION THE API (/v1/...)?
  Versioning lets you make breaking changes (v2) without breaking
  existing clients who still use v1.
  Best practice: always version from day one.

USAGE:
  In main.py:
    from app.api.v1.router import api_router
    app.include_router(api_router, prefix="/api/v1")

  This makes routes available at:
    POST /api/v1/auth/login
    POST /api/v1/inference
    POST /api/v1/feedback
    GET  /api/v1/metrics
    GET  /api/v1/health
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, feedback, inference, metrics, models, experiments

# The top-level v1 router — all sub-routers plug into this
api_router = APIRouter()

# ─── Register sub-routers ─────────────────────────────────────────────────────
# Each include_router call:
#   router  — the APIRouter from the endpoint module
#   prefix  — URL segment added to all routes in that router
#   tags    — Swagger UI grouping label

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    inference.router,
    prefix="/inference",
    tags=["Inference"],
)

api_router.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["Feedback"],
)

api_router.include_router(
    metrics.router,
    prefix="/metrics",
    tags=["Metrics & Monitoring"],
)

api_router.include_router(
    models.router,
    prefix="/models",
    tags=["Models & Targeting"],
)

api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["A/B Testing"],
)

# The /health endpoint lives at /health (not /metrics/health)
# because load balancers hit it directly
api_router.include_router(
    metrics.router,
    prefix="",
    tags=["Health"],
    include_in_schema=False,  # don't duplicate in Swagger
)
