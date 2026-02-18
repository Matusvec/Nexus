"""Pydantic schemas for task tree endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TaskBase(BaseModel):
    title: str
    description: str | None = None
    category: str  # backend|frontend|data|qa
    acceptance_criteria: list[str] = []
    estimated_effort: str | None = None  # XS|S|M|L|XL


class TaskCreate(TaskBase):
    proposal_id: UUID
    parent_task_id: UUID | None = None
    dependencies: list[UUID] = []
    sort_order: int = 0


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None  # backend|frontend|data|qa
    acceptance_criteria: list[str] | None = None
    estimated_effort: str | None = None  # XS|S|M|L|XL
    sort_order: int | None = None


class TaskResponse(TaskBase):
    id: UUID
    proposal_id: UUID
    parent_task_id: UUID | None = None
    dependencies: list[UUID] = []
    sort_order: int = 0
    prompt_version: str | None = None
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class TaskTreeResponse(BaseModel):
    """Task tree grouped by category."""
    proposal_id: UUID
    total_tasks: int
    backend: list[TaskResponse] = []
    frontend: list[TaskResponse] = []
    data: list[TaskResponse] = []
    qa: list[TaskResponse] = []


class GenerateTasksRequest(BaseModel):
    proposal_id: UUID
