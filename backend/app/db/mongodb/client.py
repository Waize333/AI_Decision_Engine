"""
backend/app/db/mongodb/client.py
=================================
PURPOSE:
  Creates the async MongoDB client using Motor (async driver).

WHY MONGODB HERE?
  MongoDB is a document database — perfect for variable-structure data:
    - Inference requests/responses (fields vary by model)
    - Feedback documents
    - Drift event records

  PostgreSQL would require schema migrations for every new field.
  MongoDB lets us store what we have without a rigid schema.

USAGE:
  from app.db.mongodb.client import get_mongo_db, get_collection

  async def save_inference_log(data: dict):
      db = get_mongo_db()
      collection = get_collection("requests")
      await collection.insert_one(data)
"""

from typing import Optional

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── CLIENT ──────────────────────────────────────────────────────────────────
# Motor is the async wrapper around pymongo.
# One client is created at module level and shared across all requests.
# Motor manages its own connection pool internally.

_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None


def get_mongo_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    """
    Get or create the MongoDB client singleton.

    Called on first use — not at import time — so tests can mock it.
    """
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGO_URL,
            serverSelectionTimeoutMS=5000,   # fail fast if DB unreachable
            maxPoolSize=50,                  # max concurrent connections
            minPoolSize=5,                   # keep 5 connections warm
        )
        logger.info("mongodb_client_created", url=settings.MONGO_URL)
    return _client


def get_mongo_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """
    Get the application's MongoDB database object.

    Returns the database named in settings.MONGO_DB.
    """
    client = get_mongo_client()
    return client[settings.MONGO_DB]


def get_collection(
    collection_name: str,
) -> motor.motor_asyncio.AsyncIOMotorCollection:
    """
    Get a specific collection from the application database.

    Collections in this project:
      - "requests"    → inference request logs
      - "responses"   → model output logs
      - "feedback"    → user feedback documents
      - "errors"      → error event logs
      - "drift_stats" → drift detection results

    Args:
        collection_name: Name of the MongoDB collection

    Example:
        collection = get_collection("feedback")
        await collection.find_one({"request_id": "abc-123"})
    """
    db = get_mongo_db()
    return db[collection_name]


async def close_mongo_connection() -> None:
    """
    Close the MongoDB client.
    Called during application shutdown (in main.py lifespan).
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("mongodb_connection_closed")


async def create_indexes() -> None:
    """
    Create MongoDB indexes for fast queries.

    WHY INDEXES?
      Without indexes, MongoDB scans every document in a collection.
      With indexes, it goes directly to matching documents.
      These should run once at startup.

    Index strategy:
      - requests: filter by request_id, user_id, model_version, timestamp
      - feedback: filter by request_id
      - drift_stats: filter by model_version and detected_at
    """
    db = get_mongo_db()

    # Requests collection
    await db["requests"].create_index(
        [("request_id", ASCENDING)], unique=True, name="idx_requests_request_id"
    )
    await db["requests"].create_index(
        [("user_id", ASCENDING)], name="idx_requests_user_id"
    )
    await db["requests"].create_index(
        [("model_version", ASCENDING)], name="idx_requests_model_version"
    )
    await db["requests"].create_index(
        [("created_at", DESCENDING)], name="idx_requests_created_at"
    )

    # Feedback collection
    await db["feedback"].create_index(
        [("request_id", ASCENDING)], name="idx_feedback_request_id"
    )
    await db["feedback"].create_index(
        [("user_id", ASCENDING)], name="idx_feedback_user_id"
    )

    # Drift stats collection
    await db["drift_stats"].create_index(
        [("model_version", ASCENDING), ("detected_at", DESCENDING)],
        name="idx_drift_version_time",
    )

    logger.info("mongodb_indexes_created")
