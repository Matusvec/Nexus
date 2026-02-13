from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas.problems import ExtractProblemsRequest, ExtractProblemsResponse, ProblemMentionResponse
from app.services.extraction_service import extract_problems_for_evidence

router = APIRouter()


@router.post("/jobs/extract_problems", response_model=ExtractProblemsResponse)
async def extract_problems_endpoint(
    payload: ExtractProblemsRequest,
    session: AsyncSession = Depends(get_session),
) -> ExtractProblemsResponse:
    try:
        mentions = await extract_problems_for_evidence(
            session, payload.evidence_id, payload.max_chunks
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ExtractProblemsResponse(
        evidence_id=payload.evidence_id,
        extracted_count=len(mentions),
        problems=[ProblemMentionResponse.model_validate(m) for m in mentions],
    )
