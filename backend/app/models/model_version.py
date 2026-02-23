"""
backend/app/models/model_version.py
=====================================
PURPOSE:
  SQLAlchemy model for the 'model_versions' table.
  This IS the model registry — every trained model gets a row here.

WHY TRACK VERSIONS?
  Without versioning you can't:
    - Know which model made a specific prediction
    - Roll back to a previous version when accuracy drops
    - Run A/B tests between versions
    - Audit "who deployed what, when"

ROW LIFECYCLE:
  1. Engineer trains a new model → creates a row (status=staged)
  2. Engineer deploys it        → status=active
  3. Better version arrives     → old one becomes status=deprecated
  4. Something breaks           → rollback sets previous to active again
"""

import enum
from typing import Optional

from sqlalchemy import JSON, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ModelStatus(str, enum.Enum):
    """
    Lifecycle state of a model version.

    staged     → trained, not yet serving traffic
    active     → currently serving production traffic
    shadow     → running in parallel (logging, not serving)
    deprecated → replaced by a newer version, kept for rollback
    archived   → too old to rollback to, safe to delete artifacts
    """
    STAGED = "staged"
    ACTIVE = "active"
    SHADOW = "shadow"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelVersion(Base, UUIDMixin, TimestampMixin):
    """
    One row = one trained model artifact.

    Columns:
      name             — human-readable name ("fraud_detector_v2")
      version_tag      — semantic version ("v2.1.3")
      description      — what changed vs previous version
      artifact_path    — where the .joblib / .pkl file lives
      training_data_hash — SHA256 of training dataset (reproducibility)
      status           — lifecycle state (enum above)
      traffic_percentage — % of requests routed here (for A/B)
      metrics          — JSON blob: {"accuracy": 0.94, "f1": 0.91, ...}
      deployed_by      — FK to users.id (who deployed it)
      feature_schema   — JSON blob: expected input field names + types
    """
    __tablename__ = "model_versions"

    # Human-readable identifier
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    # Semantic version string for display
    version_tag: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Path to the saved model file (local path or S3 URI)
    artifact_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    # SHA256 hash of the training dataset — proves reproducibility
    training_data_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    # Lifecycle state
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus, name="model_status_enum"),
        default=ModelStatus.STAGED,
        nullable=False,
        index=True,
    )

    # Traffic split for A/B testing (0–100)
    traffic_percentage: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Evaluation metrics as a flexible JSON blob
    # Example: {"accuracy": 0.94, "precision": 0.91, "recall": 0.88,
    #           "f1": 0.895, "auc_roc": 0.97, "latency_p99_ms": 12}
    metrics: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )

    # Input schema — used for validation before inference
    # Example: {"feature1": "float", "feature2": "int", ...}
    feature_schema: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )

    # Audit: who deployed this version
    deployed_by: Mapped[Optional[str]] = mapped_column(
        String(36),     # FK to users.id (string UUID)
        nullable=True,
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<ModelVersion name={self.name!r} "
            f"tag={self.version_tag!r} status={self.status!r}>"
        )

    @property
    def is_serving(self) -> bool:
        """True if this version is currently receiving traffic."""
        return self.status == ModelStatus.ACTIVE and self.traffic_percentage > 0
