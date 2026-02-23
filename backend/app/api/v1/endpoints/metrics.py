"""
backend/app/api/v1/endpoints/metrics.py
=========================================
PURPOSE:
  Route handlers for metrics, health checks, and drift alerts.
  These power your monitoring dashboard.

ACCESS:
  /health     → public (load balancers need this)
  /metrics    → ANALYST+ only
  /metrics/drift → ANALYST+ only
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, status

from app.core.logging import get_logger
from app.db.mongodb.client import get_collection
from app.middleware.auth import require_analyst
from app.models.user import User

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Public endpoint for load balancers and uptime monitors.",
    tags=["Health"],
)
async def health_check():
    """
    GET /health

    Returns 200 OK if the application is running.
    Production note: extend this to check DB connectivity.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "",
    summary="Get model performance metrics",
    description=(
        "Returns aggregated accuracy, latency, and feedback statistics "
        "for all active model versions over the last N days. "
        "Requires **Analyst** role."
    ),
)
async def get_metrics(
    days: int = Query(default=7, ge=1, le=90, description="Lookback period in days"),
    current_user: User = Depends(require_analyst),
) -> Dict[str, Any]:
    """
    GET /metrics?days=7

    Queries MongoDB 'requests' and 'feedback' collections to compute:
      - Total inference count
      - Average latency
      - Cache hit rate
      - Accuracy (from feedback)
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    requests_col = get_collection("requests")
    feedback_col = get_collection("feedback")

    # ── Inference stats ────────────────────────────────────────────
    pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {"$group": {
            "_id": "$model_version_tag",
            "total_requests": {"$sum": 1},
            "avg_latency_ms": {"$avg": "$latency_ms"},
            "avg_confidence": {"$avg": "$confidence"},
            "cache_hits": {"$sum": {"$cond": ["$cached", 1, 0]}},
        }},
    ]
    inference_cursor = requests_col.aggregate(pipeline)
    inference_stats = await inference_cursor.to_list(length=100)

    # ── Feedback stats ─────────────────────────────────────────────
    fb_pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {"$group": {
            "_id": "$model_version_tag",
            "total_feedback": {"$sum": 1},
            "correct": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
        }},
    ]
    fb_cursor = feedback_col.aggregate(fb_pipeline)
    feedback_stats_raw = await fb_cursor.to_list(length=100)
    feedback_map = {f["_id"]: f for f in feedback_stats_raw}

    # ── Combine ────────────────────────────────────────────────────
    combined = []
    for stat in inference_stats:
        version = stat["_id"]
        fb = feedback_map.get(version, {})
        total_fb = fb.get("total_feedback", 0)
        correct = fb.get("correct", 0)

        combined.append({
            "model_version": version,
            "period_days": days,
            "total_requests": stat["total_requests"],
            "avg_latency_ms": round(stat["avg_latency_ms"] or 0, 2),
            "avg_confidence": round(stat["avg_confidence"] or 0, 4),
            "cache_hit_rate": round(
                stat["cache_hits"] / stat["total_requests"], 4
            ) if stat["total_requests"] > 0 else 0,
            "total_feedback": total_fb,
            "accuracy_from_feedback": round(correct / total_fb, 4) if total_fb > 0 else None,
        })

    return {
        "period_days": days,
        "since": since.isoformat(),
        "models": combined,
    }


@router.get(
    "/drift",
    summary="Get drift detection alerts",
    description=(
        "Returns all drift events stored by the background drift detection job. "
        "Drift indicates that model inputs or outputs have shifted from the baseline. "
        "Requires **Analyst** role."
    ),
)
async def get_drift_alerts(
    model_version: str = Query(default=None, description="Filter by model version tag"),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(require_analyst),
) -> Dict[str, Any]:
    """
    GET /metrics/drift?model_version=v1&limit=20

    The drift detection background job writes to MongoDB 'drift_stats'.
    This endpoint reads and returns those records.
    """
    drift_col = get_collection("drift_stats")

    query_filter: Dict = {}
    if model_version:
        query_filter["model_version"] = model_version

    cursor = drift_col.find(query_filter).sort("detected_at", -1).limit(limit)
    alerts = await cursor.to_list(length=limit)

    # Remove MongoDB internal _id field
    for alert in alerts:
        alert.pop("_id", None)

    return {
        "total": len(alerts),
        "alerts": alerts,
    }
