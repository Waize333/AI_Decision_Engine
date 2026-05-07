from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict
from app.models.experiment import ExperimentStatus, ExperimentOutcome

class ExperimentBase(BaseModel):
    name: str
    description: Optional[str] = None
    control_version_id: str
    treatment_version_id: str
    traffic_split: int
    min_sample_size: int = 1000
    evaluation_period_hours: int = 24
    auto_promote: bool = False
    promotion_threshold: float = 0.02

class ExperimentCreate(ExperimentBase):
    pass

class ExperimentUpdateStatus(BaseModel):
    status: ExperimentStatus

class ExperimentResponse(ExperimentBase):
    id: str
    status: ExperimentStatus
    outcome: ExperimentOutcome
    result_summary: Optional[dict] = None
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
