"""
scripts/run_drift_detection.py
================================
PURPOSE:
  Runs drift detection across recent inference requests and writes stats to the `drift_stats` MongoDB collection.
  This script can be executed via a cron job or scheduled task to continuously monitor model health.
"""
import asyncio
import uuid
import sys
import os
from datetime import datetime, timedelta, timezone

# Add backend directory to sys.path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from app.db.mongodb.client import get_collection, close_mongo_connection
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

async def detect_drift(lookback_hours: int = 24):
    """
    Calculate simple drift statistics for each active model version over the last N hours.
    In a real ML system, this would use KL-divergence, KS-tests, etc., on feature distributions.
    Here we compute basic distribution metrics and simulated scores to fulfill the Phase 5 requirement.
    """
    logger.info("drift_detection_started", lookback_hours=lookback_hours)
    
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    requests_col = get_collection("requests")
    drift_col = get_collection("drift_stats")
    
    # 1. Aggregate features per model version
    pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {"$group": {
            "_id": "$model_version_tag",
            "request_count": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence"},
        }}
    ]
    
    try:
        cursor = requests_col.aggregate(pipeline)
        stats = await cursor.to_list(length=100)
    except Exception as e:
        logger.error("drift_detection_query_failed", error=str(e))
        return

    for stat in stats:
        model_version = stat.get("_id")
        if not model_version:
            continue
            
        request_count = stat.get("request_count", 0)
        avg_confidence = stat.get("avg_confidence", 0.0)
        
        # Simulated drift score (in production, compare current distribution to baseline distribution)
        # We will mock a drift score based on confidence degradation
        # e.g., if confidence < 0.8, we increase the simulated drift score.
        base_drift = 0.05
        confidence_penalty = max(0, 0.8 - avg_confidence) * 2.0
        simulated_drift_score = min(1.0, base_drift + confidence_penalty)
        
        has_drifted = simulated_drift_score > 0.3
        
        drift_record = {
            "drift_id": f"drift-{uuid.uuid4().hex[:12]}",
            "model_version": model_version,
            "detected_at": datetime.now(timezone.utc),
            "period_hours": lookback_hours,
            "request_count": request_count,
            "drift_score": round(simulated_drift_score, 4),
            "has_drifted": has_drifted,
            "metrics": {
                "avg_confidence": round(avg_confidence, 4)
            }
        }
        
        await drift_col.insert_one(drift_record)
        logger.info("drift_stats_saved", model_version=model_version, drift_score=drift_record["drift_score"])

    logger.info("drift_detection_completed", models_analyzed=len(stats))

async def main():
    try:
        await detect_drift()
    finally:
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(main())
