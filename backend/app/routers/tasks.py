"""Task tree endpoints — strategy Section F."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.tasks import Task
from app.schemas.tasks import (
    TaskResponse,
    TaskTreeResponse,
    TaskUpdate,
)
from app.services.task_tree_service import get_task_tree

router = APIRouter()


@router.get("/proposals/{proposal_id}/tasks", response_model=TaskTreeResponse)
async def get_tasks_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> TaskTreeResponse:
    """Return the task tree for a proposal, grouped by category."""
    try:
        tasks = await get_task_tree(session, proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    grouped: dict[str, list[TaskResponse]] = {
        "backend": [], "frontend": [], "data": [], "qa": [],
    }
    for task in tasks:
        category = task.category if task.category in grouped else "backend"
        grouped[category].append(TaskResponse.model_validate(task))

    return TaskTreeResponse(
        proposal_id=proposal_id,
        total_tasks=len(tasks),
        **grouped,
    )


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task_endpoint(
    task_id: UUID,
    payload: TaskUpdate,
    session: AsyncSession = Depends(get_session),
) -> TaskResponse:
    """Update a task (partial update)."""
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(task, field, value)
    await session.commit()
    return TaskResponse.model_validate(task)
