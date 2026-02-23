"""
backend/app/api/v1/endpoints/inference.py
==========================================
PURPOSE:
  FastAPI route handlers for the inference API.
  This is the most performance-critical part of the system.

KEY DESIGN:
  - BackgroundTasks for async MongoDB logging (don't block the HTTP response)
  - Redis cache checked before hitting the model
  - Any authenticated user (CLIENT+) can submit inference requests
  - Only ANALYST+ can retrieve historical inference records
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgresql.session import get_db
from app.middleware.auth import get_current_user, require_analyst
from app.models.user import User
from app.schemas.inference import InferenceDetailResponse, InferenceRequest, InferenceResponse
from app.services.inference import InferenceService

router = APIRouter()


@router.post(
    "",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Run model inference",
    description=(
        "Submit input features and receive a model prediction. "
        "Save the `request_id` from the response — you'll need it to submit feedback.\n\n"
        "**Caching:** Identical inputs are cached for 5 minutes. "
        "Cached responses have `cached: true` and `latency_ms: 0`."
    ),
)
async def run_inference(
    body: InferenceRequest,
    background_tasks: BackgroundTasks,          # for async MongoDB logging
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),  # any authenticated user
):
    """
    POST /inference

    Request body:
    {
      "features": { "transaction_amount": 1250.00, ... },
      "model_version": null,    // optional — pin to specific version
      "request_id": null        // optional — client-supplied idempotency key
    }

    Response:
    {
      "request_id": "req-abc123",
      "prediction": 1,
      "confidence": 0.94,
      "model_version": "v1",
      "latency_ms": 12.4,
      "cached": false,
      "created_at": "2026-02-23T12:00:00Z"
    }
    """
    response = await InferenceService.run(
        request=body,
        user_id=current_user.id,
        db=db,
        background_tasks=background_tasks,
    )
    return response


@router.get(
    "/{request_id}",
    response_model=InferenceDetailResponse,
    summary="Get inference record by ID",
    description=(
        "Retrieve a full inference record by its `request_id`. "
        "Includes the original input, prediction, confidence, and metadata. "
        "Requires **Analyst** role or higher."
    ),
)
async def get_inference(
    request_id: str = Path(
        ...,
        description="The request_id returned by POST /inference",
        examples=["req-abc123"],
    ),
    current_user: User = Depends(require_analyst),  # analyst+ only
):
    """
    GET /inference/{request_id}

    Analyst and above can look up any inference record.
    Clients can only call POST /inference (not this endpoint).

    Errors:
      404 Not Found — request_id doesn't exist in MongoDB
    """
    doc = await InferenceService.get_by_request_id(request_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No inference record found for request_id='{request_id}'.",
        )

    # Map MongoDB document to response schema
    return InferenceDetailResponse(
        request_id=doc["request_id"],
        prediction=doc["prediction"],
        confidence=doc["confidence"],
        model_version=doc["model_version_tag"],
        latency_ms=doc["latency_ms"],
        cached=doc.get("cached", False),
        created_at=doc["created_at"],
        input_features=doc["input_features"],
        user_id=doc["user_id"],
    )
