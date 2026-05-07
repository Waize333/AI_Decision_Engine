from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class ModelVersionResponse(BaseModel):
    id: str
    version_tag: str
    status: str
    artifact_path: str
    description: Optional[str] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ModelStatusUpdate(BaseModel):
    status: str  # e.g., "ACTIVE", "DEPRECATED", "STAGING"
