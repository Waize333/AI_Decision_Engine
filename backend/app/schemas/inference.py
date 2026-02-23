"""
backend/app/schemas/inference.py
==================================
PURPOSE:
  Pydantic schemas for the /inference endpoint.
  This is the most critical API in the platform — every prediction
  goes through these schemas.

FLOW:
  Client sends InferenceRequest →
  Model produces a prediction →
  We wrap it in InferenceResponse →
  We log an InferenceLog to MongoDB (different schema — more fields)

DESIGN DECISION — flexible features dict:
  We use Dict[str, Any] for features because:
    - Different models expect different input shapes
    - A fraud model might need transaction_amount, merchant_category
    - A churn model might need usage_days, plan_type
  The actual feature validation happens inside the model service,
  not in the Pydantic schema.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── REQUEST ─────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    """
    Body for POST /inference

    features        — dict of input features for the model
                      Keys and value types depend on the active model
    model_version   — optional: request a specific version (e.g. "v2")
                      If None, the router picks based on active config + A/B split
    request_id      — optional client-supplied idempotency key
                      If omitted, we generate a UUID server-side

    Example body:
    {
      "features": {
        "transaction_amount": 1250.00,
        "merchant_category": "electronics",
        "hour_of_day": 23,
        "is_international": true
      },
      "model_version": null
    }
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Input features for the model. Keys depend on the active model.",
        examples=[{
            "transaction_amount": 1250.00,
            "merchant_category": "electronics",
            "hour_of_day": 23,
            "is_international": True,
        }],
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Pin to a specific model version tag (e.g. 'v2'). "
                    "Leave null to use the active routing config.",
        examples=["v2"],
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Client-supplied idempotency key. Auto-generated if not provided.",
        examples=["client-req-abc123"],
    )


# ─── RESPONSE ────────────────────────────────────────────────────────────────

class InferenceResponse(BaseModel):
    """
    Response for POST /inference (successful prediction)

    request_id      — unique ID for this inference (use for feedback/lookup)
    prediction      — the model's output (class label, score, etc.)
    confidence      — probability of the prediction being correct (0.0–1.0)
    model_version   — which model version processed this request
    latency_ms      — how long model inference took
    cached          — True if this result was served from cache
    """
    request_id: str = Field(..., description="Unique ID — use this for feedback.")
    prediction: Any = Field(..., description="Model output (class label, score, etc.)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Prediction confidence score (0.0 = no confidence, 1.0 = certain).",
    )
    model_version: str = Field(..., description="Version tag of the model that ran.")
    latency_ms: float = Field(..., description="Inference time in milliseconds.")
    cached: bool = Field(default=False, description="True if served from cache.")
    created_at: datetime = Field(..., description="When this inference ran.")


class InferenceDetailResponse(InferenceResponse):
    """
    Extended response for GET /inference/{request_id}
    Includes the original input features (so you can review what was sent).
    """
    input_features: Dict[str, Any] = Field(
        ..., description="The original input features sent to the model."
    )
    user_id: str = Field(..., description="ID of the user who made this request.")


class InferenceListResponse(BaseModel):
    """Paginated list of inference records for GET /inference (admin/analyst)."""
    items: List[InferenceDetailResponse]
    total: int
    page: int
    page_size: int


# ─── MONGODB LOG SCHEMA ───────────────────────────────────────────────────────

class InferenceLog(BaseModel):
    """
    The full inference record stored in MongoDB.
    Richer than the API response — includes everything we need for:
      - Drift detection (input feature distributions)
      - Feedback linking
      - Debugging
      - Retraining datasets

    NOTE: This is NOT a FastAPI response schema.
    It's used internally by the inference service to build the MongoDB doc.
    """
    request_id: str
    user_id: str
    model_version_id: str
    model_version_tag: str

    # Input and output
    input_features: Dict[str, Any]
    prediction: Any
    confidence: float
    raw_probabilities: Optional[List[float]] = None  # full class probability vector

    # Performance
    latency_ms: float
    cached: bool = False

    # Context
    experiment_id: Optional[str] = None   # set if this was part of an A/B test
    ab_group: Optional[str] = None        # "control" | "treatment"
    ip_address: Optional[str] = None      # for rate limit forensics

    # Timestamps
    created_at: datetime

    model_config = {"from_attributes": True}
