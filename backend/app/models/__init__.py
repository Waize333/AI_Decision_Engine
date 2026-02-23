"""
backend/app/models/__init__.py
===============================
PURPOSE:
  Import ALL SQLAlchemy models in one place.

WHY THIS IS CRITICAL:
  SQLAlchemy's `Base.metadata` only knows about models that have been
  imported at least once. If a model isn't imported here:
    - Alembic won't generate a migration for its table
    - `Base.metadata.create_all()` won't create its table

  By importing everything here, any code that imports from
  `app.models` automatically registers all tables.

USAGE IN ALEMBIC (alembic/env.py):
  from app.models import *   # registers all models
  target_metadata = Base.metadata
"""

from app.models.base import Base, TimestampMixin, UUIDMixin
from app.models.user import User, UserRole
from app.models.model_version import ModelVersion, ModelStatus
from app.models.experiment import Experiment, ExperimentStatus, ExperimentOutcome

__all__ = [
    # Base
    "Base",
    "TimestampMixin",
    "UUIDMixin",
    # User
    "User",
    "UserRole",
    # Model versioning
    "ModelVersion",
    "ModelStatus",
    # A/B Testing
    "Experiment",
    "ExperimentStatus",
    "ExperimentOutcome",
]
