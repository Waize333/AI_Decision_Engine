"""
backend/app/api/v1/endpoints/feedback.py
==========================================
PURPOSE:
  Route handlers for the feedback API.
  Feedback is what closes the ML training loop.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.core.logging import get_logger
from app.db.mongodb.client import get_collection
from app.middleware.auth import get_current_user
from app.models.user import User
from app.schemas.feedback import FeedbackDocument, FeedbackRequest, FeedbackResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit prediction feedback",
    description=(
        "Tell the system whether a prediction was correct or wrong. "
        "Optionally provide the true ground-truth label to auto-generate training data.\n\n"
        "You must submit the `request_id` from the inference you're rating."
    ),
)
async def submit_feedback(
    body: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    POST /feedback

    Request body:
    {
      "request_id": "req-abc123",
      "rating": 0,                   // 0=wrong, 1=correct
      "true_label": 1,               // optional
      "comment": "Was fraud."        // optional
    }
    """
    # ── Fetch the original inference from MongoDB ──────────────────
    requests_col = get_collection("requests")
    inference_doc = await requests_col.find_one({"request_id": body.request_id})

    if not inference_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No inference found for request_id='{body.request_id}'. "
                   "Submit feedback only for requests you have made.",
        )

    # ── Build full feedback document ───────────────────────────────
    feedback_id = f"fb-{uuid.uuid4().hex[:12]}"
    feedback_doc = FeedbackDocument(
        feedback_id=feedback_id,
        request_id=body.request_id,
        user_id=current_user.id,
        model_version_id=inference_doc.get("model_version_id", ""),
        model_version_tag=inference_doc.get("model_version_tag", "unknown"),
        rating=body.rating,
        true_label=body.true_label,
        comment=body.comment,
        feedback_type=body.feedback_type,
        # Copy from inference for easy training data extraction
        input_features=inference_doc.get("input_features"),
        model_prediction=inference_doc.get("prediction"),
        model_confidence=inference_doc.get("confidence"),
        is_correct=body.rating == 1,
        created_at=datetime.now(timezone.utc),
    )

    # ── Save to MongoDB (background — don't block HTTP response) ───
    background_tasks.add_task(_save_feedback, feedback_doc)

    logger.info(
        "feedback_submitted",
        feedback_id=feedback_id,
        request_id=body.request_id,
        rating=body.rating,
        user_id=current_user.id,
    )

    return FeedbackResponse(
        feedback_id=feedback_id,
        request_id=body.request_id,
        message="Feedback recorded. Thank you!",
    )


async def _save_feedback(doc: FeedbackDocument) -> None:
    """Background task: persist feedback document to MongoDB."""
    try:
        collection = get_collection("feedback")
        await collection.insert_one(doc.model_dump(mode="json"))
        logger.debug("feedback_saved", feedback_id=doc.feedback_id)
    except Exception as e:
        logger.error("feedback_save_failed", feedback_id=doc.feedback_id, error=str(e))
