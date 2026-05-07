"""
backend/app/api/v1/endpoints/models.py
======================================
PURPOSE:
  Route handlers for model versioning and instantaneous rollbacks.
  These routes are restricted exclusively to Engineers and Admins.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.postgresql.session import get_db
from app.middleware.auth import require_engineer
from app.models.user import User
from app.models.model_version import ModelVersion, ModelStatus
from app.schemas.model_version import ModelVersionResponse, ModelStatusUpdate

router = APIRouter()

@router.get(
    "",
    response_model=List[ModelVersionResponse],
    summary="List all model versions",
    description="Engineers can view all deployed models and their statuses."
)
async def list_model_versions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_engineer),
):
    query = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    result = await db.execute(query)
    models = result.scalars().all()
    return models

@router.post(
    "/{version_id}/rollback",
    response_model=ModelVersionResponse,
    summary="Rollback / Change Model Status",
    description="Instantly mark a model version as ACTIVE. Any previously ACTIVE models will be auto-demoted."
)
async def rollback_model_version(
    version_id: str,
    update: ModelStatusUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_engineer),
):
    """
    POST /models/{uuid}/rollback
    
    If an Engineer passes status="ACTIVE" for an older model (e.g., v1), 
    this safely deactivates the broken v2 model and promotes v1 as the new main model.
    """
    # 1. Ensure the requested model exists
    query = select(ModelVersion).where(ModelVersion.id == version_id)
    result = await db.execute(query)
    target_model = result.scalar_one_or_none()

    if not target_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ModelVersion with ID {version_id} not found."
        )

    # 2. If trying to make this ACTIVE, demote all other currently active models 
    if update.status == ModelStatus.ACTIVE.value:
        active_query = select(ModelVersion).where(ModelVersion.status == ModelStatus.ACTIVE)
        active_result = await db.execute(active_query)
        currently_active_models = active_result.scalars().all()
        
        for active in currently_active_models:
            # Demote them out of ACTIVE
            active.status = ModelStatus.DEPRECATED
            db.add(active)

    # 3. Update the target model status
    target_model.status = ModelStatus(update.status)
    db.add(target_model)
    
    await db.commit()
    await db.refresh(target_model)
    
    return target_model
