"""Task tree endpoints — strategy Section F."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas.tasks import (
    TaskResponse,
    TaskTreeResponse,
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
