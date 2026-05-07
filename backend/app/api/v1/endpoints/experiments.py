"""
backend/app/api/v1/endpoints/experiments.py
===========================================
PURPOSE:
  Route handlers for A/B testing experiments.
  Engineers can create experiments, monitor them, and update statuses.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.postgresql.session import get_db
from app.middleware.auth import require_engineer
from app.models.user import User
from app.models.experiment import Experiment, ExperimentStatus
from app.schemas.experiment import ExperimentCreate, ExperimentResponse, ExperimentUpdateStatus

router = APIRouter()

@router.post(
    "",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new A/B experiment",
    description="Engineers can create experiments to test a challenger model against a baseline."
)
async def create_experiment(
    experiment_in: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_engineer),
):
    # Only one running experiment allowed at a time for simplicity
    query = select(Experiment).where(Experiment.status == ExperimentStatus.RUNNING)
    result = await db.execute(query)
    if result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An experiment is already running. Complete it or pause it first."
        )

    experiment = Experiment(
        name=experiment_in.name,
        description=experiment_in.description,
        control_version_id=experiment_in.control_version_id,
        treatment_version_id=experiment_in.treatment_version_id,
        traffic_split=experiment_in.traffic_split,
        min_sample_size=experiment_in.min_sample_size,
        evaluation_period_hours=experiment_in.evaluation_period_hours,
        auto_promote=experiment_in.auto_promote,
        promotion_threshold=experiment_in.promotion_threshold,
        status=ExperimentStatus.DRAFT,
        created_by=current_user.id
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment

@router.get(
    "",
    response_model=List[ExperimentResponse],
    summary="List all experiments"
)
async def list_experiments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_engineer),
):
    query = select(Experiment).order_by(Experiment.created_at.desc())
    result = await db.execute(query)
    return result.scalars().all()

@router.patch(
    "/{experiment_id}/status",
    response_model=ExperimentResponse,
    summary="Update experiment status"
)
async def update_experiment_status(
    experiment_id: str,
    update: ExperimentUpdateStatus,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_engineer),
):
    query = select(Experiment).where(Experiment.id == experiment_id)
    result = await db.execute(query)
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found."
        )

    # Ensure no other experiment is running if we are starting this one
    if update.status == ExperimentStatus.RUNNING:
        running_query = select(Experiment).where(Experiment.status == ExperimentStatus.RUNNING, Experiment.id != experiment_id)
        running_result = await db.execute(running_query)
        if running_result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Another experiment is already running."
            )

    experiment.status = update.status
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment
