"""
backend/app/models/experiment.py
==================================
PURPOSE:
  SQLAlchemy model for the 'experiments' table.
  An experiment is a controlled A/B test between two model versions.

HOW AN EXPERIMENT WORKS:
  1. Engineer creates an experiment:
       - name: "v1 vs v2 fraud detection"
       - control_version_id: model v1 (baseline)
       - treatment_version_id: model v2 (challenger)
       - traffic_split: 70 (70% → v1, 30% → v2)

  2. Inference router reads the active experiment and routes
     requests based on the traffic split.

  3. The system collects accuracy, latency, and feedback for both.

  4. After evaluation_period_hours, auto-promote if:
       - Treatment accuracy > control accuracy
       - Treatment latency is acceptable
       - Treatment drift score is low

DECISION OUTCOMES:
  pending    → experiment running, no winner yet
  promoted   → treatment won, now active
  rolled_back → control won, treatment deprecated
  inconclusive → no statistically significant difference
"""

import enum
from typing import Optional

from sqlalchemy import Boolean, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ExperimentStatus(str, enum.Enum):
    DRAFT = "draft"           # created but not started
    RUNNING = "running"       # actively splitting traffic
    PAUSED = "paused"         # temporarily stopped
    COMPLETED = "completed"   # evaluation done
    CANCELLED = "cancelled"   # aborted


class ExperimentOutcome(str, enum.Enum):
    PENDING = "pending"
    PROMOTED = "promoted"            # treatment version won
    ROLLED_BACK = "rolled_back"      # control version won
    INCONCLUSIVE = "inconclusive"    # no significant difference


class Experiment(Base, UUIDMixin, TimestampMixin):
    """
    Represents one A/B test between two model versions.

    Columns:
      name                    — human-readable experiment name
      description             — hypothesis being tested
      control_version_id      — baseline model (FK → model_versions.id)
      treatment_version_id    — challenger model (FK → model_versions.id)
      traffic_split           — % of traffic sent to CONTROL (remainder → treatment)
      status                  — lifecycle of the experiment
      outcome                 — final decision after evaluation
      min_sample_size         — don't decide until this many requests
      evaluation_period_hours — how long to run before evaluating
      auto_promote            — promote treatment automatically if it wins?
      promotion_threshold     — accuracy improvement required to promote (e.g. 0.02 = 2%)
      result_summary          — JSON blob with final stats after evaluation
    """
    __tablename__ = "experiments"

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # FK to model_versions table (stored as string UUID)
    control_version_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    treatment_version_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    # 70 means 70% to control, 30% to treatment
    traffic_split: Mapped[int] = mapped_column(
        Integer,
        default=50,
        nullable=False,
    )

    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus, name="experiment_status_enum"),
        default=ExperimentStatus.DRAFT,
        nullable=False,
        index=True,
    )

    outcome: Mapped[ExperimentOutcome] = mapped_column(
        Enum(ExperimentOutcome, name="experiment_outcome_enum"),
        default=ExperimentOutcome.PENDING,
        nullable=False,
    )

    # Statistical power settings
    min_sample_size: Mapped[int] = mapped_column(Integer, default=1000)
    evaluation_period_hours: Mapped[int] = mapped_column(Integer, default=24)

    # Promotion settings
    auto_promote: Mapped[bool] = mapped_column(Boolean, default=False)
    promotion_threshold: Mapped[float] = mapped_column(Float, default=0.02)

    # JSON blob with final results after the experiment ends
    # Example: {
    #   "control_accuracy": 0.91, "treatment_accuracy": 0.94,
    #   "control_latency_p99": 10, "treatment_latency_p99": 12,
    #   "p_value": 0.03, "winner": "treatment"
    # }
    result_summary: Mapped[Optional[dict]] = mapped_column(
        "result_summary",
        type_=Text,  # stored as JSON string for compatibility
        nullable=True,
    )

    # Audit
    created_by: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Experiment name={self.name!r} "
            f"status={self.status!r} outcome={self.outcome!r}>"
        )
