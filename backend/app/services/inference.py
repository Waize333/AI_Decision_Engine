"""
backend/app/services/inference.py
====================================
PURPOSE:
  The core inference pipeline — this is where AI actually runs.

  Flow for every POST /inference request:
    1. Generate unique request_id
    2. Check Redis cache (skip model if identical input seen before)
    3. Load the active model version config
    4. Run the ML model → get prediction + confidence
    5. Log the full inference to MongoDB (async background task)
    6. Return the response to the client

  Steps 1-4-5-6 are all async and non-blocking.

MODEL ABSTRACTION:
  The actual model (RandomForestClassifier, logistic regression, etc.)
  is wrapped behind the ModelRunner class. To swap from a fraud model
  to an LLM, only ModelRunner changes — the pipeline stays the same.
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.mongodb.client import get_collection
from app.db.redis.client import CacheManager, get_redis
from app.models.model_version import ModelStatus, ModelVersion
from app.schemas.inference import InferenceLog, InferenceRequest, InferenceResponse

logger = get_logger(__name__)


# ─── MODEL RUNNER ─────────────────────────────────────────────────────────────

class ModelRunner:
    """
    Wraps the actual ML model and provides a uniform predict() interface.

    WHY A WRAPPER CLASS?
      The model is loaded from disk once and reused across requests.
      The wrapper handles:
        - Loading from disk (joblib.load)
        - Input array construction (dict → numpy array in correct order)
        - Output parsing (raw numpy output → Python types)
        - Timing measurement

    In Phase 2 we'll train a real RandomForestClassifier.
    Right now it's a stub so the API works end-to-end.
    """

    def __init__(self, version_tag: str, artifact_path: str):
        self.version_tag = version_tag
        self.artifact_path = artifact_path
        self._model = None
        self._feature_names: list[str] = []

    def _load(self):
        """
        Load model from disk using joblib.
        Called lazily — model is only loaded on first prediction.
        """
        try:
            import joblib
            self._model = joblib.load(self.artifact_path)
            logger.info("model_loaded", version=self.version_tag, path=self.artifact_path)
        except FileNotFoundError:
            logger.warning(
                "model_file_not_found",
                path=self.artifact_path,
                note="Using stub predictions until a real model is trained."
            )
            self._model = None

    def predict(
        self, features: Dict[str, Any]
    ) -> tuple[Any, float, list[float]]:
        """
        Run model inference on the input features.

        Returns:
            prediction       — the model's output class label (int or str)
            confidence       — probability of the predicted class (0.0–1.0)
            raw_probabilities — full probability vector for all classes

        STUB BEHAVIOUR (before real model is trained):
          Returns deterministic fake predictions based on input hash.
          This lets you test the API end-to-end before Phase 2.
        """
        if self._model is None:
            self._load()

        # ── Real model prediction ──────────────────────────────────
        if self._model is not None:
            try:
                import numpy as np
                # Build feature array in the order the model expects
                feature_vector = [features.get(f, 0) for f in self._feature_names]
                X = np.array([feature_vector])
                proba = self._model.predict_proba(X)[0].tolist()
                prediction = int(np.argmax(proba))
                confidence = float(max(proba))
                return prediction, confidence, proba
            except Exception as e:
                logger.error("model_predict_error", error=str(e))
                # Fall through to stub

        # ── Stub prediction (Phase 1 placeholder) ─────────────────
        # Deterministic based on input so cache tests are consistent
        feature_hash = int(hashlib.md5(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest(), 16)
        prediction = feature_hash % 2        # binary: 0 or 1
        confidence = 0.5 + (feature_hash % 100) / 200.0  # 0.50–1.00
        proba = [1 - confidence, confidence] if prediction == 1 else [confidence, 1 - confidence]

        return prediction, round(confidence, 4), [round(p, 4) for p in proba]


# ─── MODEL REGISTRY (in-process cache) ────────────────────────────────────────

_loaded_models: Dict[str, ModelRunner] = {}


def _get_runner(version_tag: str, artifact_path: str) -> ModelRunner:
    """
    Get a cached ModelRunner for the given version.
    Avoids reloading the model from disk on every request.
    """
    if version_tag not in _loaded_models:
        _loaded_models[version_tag] = ModelRunner(version_tag, artifact_path)
    return _loaded_models[version_tag]


# ─── INFERENCE SERVICE ────────────────────────────────────────────────────────

class InferenceService:

    @staticmethod
    async def run(
        request: InferenceRequest,
        user_id: str,
        db: AsyncSession,
        background_tasks: BackgroundTasks,
    ) -> InferenceResponse:
        """
        Full inference pipeline.

        Args:
            request:          Validated InferenceRequest from the endpoint
            user_id:          ID of the authenticated user
            db:               PostgreSQL session (to look up model version)
            background_tasks: FastAPI background task queue (for async logging)

        Returns:
            InferenceResponse with prediction, confidence, latency, etc.
        """
        # ── 1. Assign request ID ───────────────────────────────────
        request_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # ── 2. Check cache ─────────────────────────────────────────
        cache_key = InferenceService._build_cache_key(request.features)
        try:
            redis = await get_redis()
            cached_raw = await CacheManager.get(redis, cache_key)
            if cached_raw:
                cached = json.loads(cached_raw)
                logger.info("inference_cache_hit", request_id=request_id)
                return InferenceResponse(
                    request_id=request_id,
                    prediction=cached["prediction"],
                    confidence=cached["confidence"],
                    model_version=cached["model_version"],
                    latency_ms=0.0,
                    cached=True,
                    created_at=now,
                )
        except Exception as e:
            logger.warning("cache_read_error", error=str(e))

        # ── 3. Load active model version ───────────────────────────
        model_version = await InferenceService._get_active_model(
            request.model_version, db
        )

        # ── 4. Run model ───────────────────────────────────────────
        t_start = time.perf_counter()
        runner = _get_runner(model_version.version_tag, model_version.artifact_path)
        prediction, confidence, raw_proba = runner.predict(request.features)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        logger.info(
            "inference_complete",
            request_id=request_id,
            model_version=model_version.version_tag,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
        )

        # ── 5. Store in cache ──────────────────────────────────────
        try:
            await CacheManager.set(
                redis,
                cache_key,
                json.dumps({"prediction": prediction, "confidence": confidence,
                            "model_version": model_version.version_tag}),
            )
        except Exception as e:
            logger.warning("cache_write_error", error=str(e))

        # ── 6. Log to MongoDB (non-blocking background task) ───────
        inference_log = InferenceLog(
            request_id=request_id,
            user_id=user_id,
            model_version_id=model_version.id,
            model_version_tag=model_version.version_tag,
            input_features=request.features,
            prediction=prediction,
            confidence=confidence,
            raw_probabilities=raw_proba,
            latency_ms=latency_ms,
            cached=False,
            created_at=now,
        )
        background_tasks.add_task(
            InferenceService._log_to_mongo, inference_log
        )

        return InferenceResponse(
            request_id=request_id,
            prediction=prediction,
            confidence=confidence,
            model_version=model_version.version_tag,
            latency_ms=latency_ms,
            cached=False,
            created_at=now,
        )

    @staticmethod
    async def _get_active_model(
        requested_version: Optional[str],
        db: AsyncSession,
    ) -> ModelVersion:
        """
        Find the model version to use for this request.

        If a specific version is requested → use that.
        Otherwise → use the currently ACTIVE version in the registry.
        """
        query = select(ModelVersion)

        if requested_version:
            query = query.where(ModelVersion.version_tag == requested_version)
        else:
            query = query.where(ModelVersion.status == ModelStatus.ACTIVE)

        result = await db.execute(query)
        model_version = result.scalar_one_or_none()

        if not model_version:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No active model version available. Please contact an engineer.",
            )

        return model_version

    @staticmethod
    async def _log_to_mongo(log: InferenceLog) -> None:
        """
        Persist the inference record to MongoDB.
        Runs as a background task — does NOT block the HTTP response.

        The stored document enables:
          - GET /inference/{request_id} lookups
          - Drift detection (feature distribution analysis)
          - Feedback linking (feedback.request_id → log.request_id)
          - Retraining dataset generation
        """
        try:
            collection = get_collection("requests")
            await collection.insert_one(
                log.model_dump(mode="json")
            )
            logger.debug("inference_logged", request_id=log.request_id)
        except Exception as e:
            logger.error("inference_log_failed", request_id=log.request_id, error=str(e))

    @staticmethod
    def _build_cache_key(features: Dict[str, Any]) -> str:
        """
        Build a deterministic cache key from input features.

        We sort the keys so {"a": 1, "b": 2} and {"b": 2, "a": 1}
        produce the same cache key (same request, different key order).
        """
        canonical = json.dumps(features, sort_keys=True)
        return f"inference:cache:{hashlib.sha256(canonical.encode()).hexdigest()}"

    @staticmethod
    async def get_by_request_id(request_id: str) -> Optional[dict]:
        """
        Fetch a past inference log from MongoDB by request_id.
        Used by GET /inference/{request_id}.
        """
        collection = get_collection("requests")
        doc = await collection.find_one({"request_id": request_id})
        if doc:
            doc.pop("_id", None)   # remove MongoDB's internal ObjectId
        return doc
