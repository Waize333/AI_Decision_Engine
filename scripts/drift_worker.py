import asyncio
import logging
from datetime import datetime, timedelta, timezone

# Assuming you run this from the project root, we import the mongoclient
from backend.app.db.mongodb.client import get_collection, connect_to_mongo, close_mongo_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift_worker")

DRIFT_THRESHOLD_CONFIDENCE = 0.75  # If avg confidence drops below 75%, trigger alert

async def detect_drift():
    """
    Simulates a background data drift detection job.
    In a real FAANG production system, this would use KL-divergence 
    or Kolmogorov-Smirnov tests on feature distributions.
    Here, we use a simpler proxy: Average Model Confidence 
    and Feedback Accuracy over the last 24 hours.
    """
    logger.info("Starting Drift Detection Analysis...")
    await connect_to_mongo()

    requests_col = get_collection("requests")
    drift_col = get_collection("drift_stats")

    # Look at data from the last 24 hours
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    
    # 1. Aggregate average confidence per model version
    pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {"$group": {
            "_id": "$model_version_tag",
            "avg_confidence": {"$avg": "$confidence"},
            "total_requests": {"$sum": 1}
        }}
    ]
    
    cursor = requests_col.aggregate(pipeline)
    stats = await cursor.to_list(length=100)

    for stat in stats:
        version = stat["_id"]
        avg_conf = stat["avg_confidence"]
        total = stat["total_requests"]

        logger.info(f"Model {version} - Avg Confidence: {avg_conf:.4f} across {total} requests.")

        if avg_conf < DRIFT_THRESHOLD_CONFIDENCE and total > 50:
            logger.warning(f"🚨 DRIFT DETECTED in {version}!")
            
            # 2. Write alert to drift_stats so metrics.py can serve it to Analysts
            await drift_col.insert_one({
                "model_version": version,
                "detected_at": datetime.now(timezone.utc),
                "metric_name": "average_confidence",
                "metric_value": avg_conf,
                "threshold": DRIFT_THRESHOLD_CONFIDENCE,
                "severity": "high",
                "message": f"Average confidence dropped to {avg_conf:.2f}, indicating potential data drift."
            })
            logger.info("Alert written to MongoDB successfully.")
        else:
            logger.info(f"Model {version} is healthy.")

    await close_mongo_connection()
    logger.info("Drift Analysis Complete.")

if __name__ == "__main__":
    asyncio.run(detect_drift())
