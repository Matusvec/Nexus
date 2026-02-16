from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

JobStatus = Literal["pending", "running", "completed", "failed"]


class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    job_type: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result_count: int | None = None
