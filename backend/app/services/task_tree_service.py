"""Task tree generation service — strategy Section F.

Transforms feature proposals into implementation-ready task trees
with categories (backend, frontend, data, qa) and acceptance criteria.
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.llm.client import get_client
from app.models.clusters import FeatureProposal
from app.models.tasks import Task
from app.utils.retry import call_with_retry

logger = logging.getLogger(__name__)
PROMPT_VERSION = "generate_tasks_v1"


def _build_task_prompt(proposal: FeatureProposal) -> str:
    """Build the LLM prompt for task tree generation."""
    metadata = proposal.metadata_ or {}
    return (
        "You are a senior tech lead. Convert this feature proposal into an\n"
        "implementation task tree.\n\n"
        f"PROPOSAL:\n"
        f"- Title: {proposal.title}\n"
        f"- Description: {proposal.description}\n"
        f"- Impact: {proposal.impact or 'N/A'}\n"
        f"- Effort: {proposal.effort or 'N/A'}\n"
        f"- User Story: {metadata.get('user_story', 'N/A')}\n"
        f"- One-liner: {metadata.get('one_liner', 'N/A')}\n\n"
        "Generate a hierarchical task tree. Return valid JSON only:\n"
        "{\n"
        '  "tasks": [\n'
        "    {\n"
        '      "title": "string",\n'
        '      "description": "string",\n'
        '      "category": "backend|frontend|data|qa",\n'
        '      "acceptance_criteria": ["Given X, when Y, then Z"],\n'
        '      "estimated_effort": "XS|S|M|L|XL",\n'
        '      "subtasks": [\n'
        "        {\n"
        '          "title": "string",\n'
        '          "description": "string",\n'
        '          "category": "backend|frontend|data|qa",\n'
        '          "acceptance_criteria": ["..."],\n'
        '          "estimated_effort": "XS|S|M|L|XL"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Tasks should be small enough for one developer\n"
        "- Every task must have at least one acceptance criterion\n"
        "- Data migrations come before backend tasks\n"
        "- QA tasks reference the features they validate\n"
    )


async def generate_tasks_for_proposal(
    session: AsyncSession,
    proposal_id: UUID,
) -> list[Task]:
    """Generate a task tree for a proposal using LLM.

    1. Load the proposal
    2. Call LLM with the task generation prompt
    3. Parse the response and create Task records
    4. Return the created tasks
    """
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise ValueError(f"Proposal {proposal_id} not found")

    # Generate tasks via LLM
    client = get_client()
    prompt = _build_task_prompt(proposal)
    raw = await call_with_retry(
        client.generate_json, prompt, PROMPT_VERSION,
        label="Task tree generation",
    )

    raw_tasks = raw.get("tasks", [])
    if not raw_tasks:
        logger.warning("LLM returned no tasks for proposal %s", proposal_id)
        return []

    # Delete existing tasks for this proposal (re-generation replaces old tasks)
    from sqlalchemy import delete
    await session.execute(delete(Task).where(Task.proposal_id == proposal_id))
    await session.flush()

    created_tasks: list[Task] = []
    sort_order = 0

    for raw_task in raw_tasks:
        task = Task(
            proposal_id=proposal_id,
            title=raw_task.get("title", "Untitled"),
            description=raw_task.get("description"),
            category=raw_task.get("category", "backend"),
            acceptance_criteria=raw_task.get("acceptance_criteria", []),
            estimated_effort=raw_task.get("estimated_effort"),
            sort_order=sort_order,
            prompt_version=PROMPT_VERSION,
        )
        session.add(task)
        await session.flush()
        created_tasks.append(task)
        sort_order += 1

        # Handle subtasks
        for sub in raw_task.get("subtasks", []):
            subtask = Task(
                proposal_id=proposal_id,
                parent_task_id=task.id,
                title=sub.get("title", "Untitled"),
                description=sub.get("description"),
                category=sub.get("category", task.category),
                acceptance_criteria=sub.get("acceptance_criteria", []),
                estimated_effort=sub.get("estimated_effort"),
                sort_order=sort_order,
                prompt_version=PROMPT_VERSION,
            )
            session.add(subtask)
            created_tasks.append(subtask)
            sort_order += 1

    await session.commit()
    logger.info(
        "Generated %d tasks for proposal %s", len(created_tasks), proposal_id
    )
    return created_tasks


async def get_task_tree(
    session: AsyncSession,
    proposal_id: UUID,
) -> list[Task]:
    """Return all tasks for a proposal, ordered by sort_order."""
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise ValueError(f"Proposal {proposal_id} not found")

    query = (
        select(Task)
        .where(Task.proposal_id == proposal_id)
        .order_by(Task.sort_order.asc())
    )
    return list((await session.execute(query)).scalars().all())
