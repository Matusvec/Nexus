"""Phase 2 routes: clusters, proposals, roadmap."""

import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas.clusters import (
    CitationResponse,
    ClusterDetailResponse,
    ClusterResponse,
    ProposalCreate,
    ProposalDetailResponse,
    ProposalResponse,
    RoadmapItem,
    RoadmapResponse,
)
from app.services.cluster_service import (
    add_citation,
    create_proposal,
    get_cluster_detail,
    get_proposal_detail,
    get_roadmap,
    list_clusters,
    run_threshold_clustering,
)

router = APIRouter()


@router.post("/clusters/run")
async def run_clustering_endpoint(
    threshold: float = Query(0.75, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Trigger threshold clustering on all problem embeddings.

    Note: runs synchronously — returns 200 with results when done.
    """
    clusters = await run_threshold_clustering(session, threshold=threshold)
    return {"clusters_created": len(clusters)}


@router.get("/clusters")
async def list_clusters_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> dict:
    items, total = await list_clusters(session, page=page, per_page=per_page)
    return {
        "items": [ClusterResponse.model_validate(c) for c in items],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, math.ceil(total / per_page)),
    }


@router.get("/clusters/{cluster_id}")
async def get_cluster_endpoint(
    cluster_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ClusterDetailResponse:
    cluster = await get_cluster_detail(session, cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return ClusterDetailResponse.model_validate(cluster)


@router.post("/proposals", response_model=ProposalResponse, status_code=201)
async def create_proposal_endpoint(
    payload: ProposalCreate,
    session: AsyncSession = Depends(get_session),
) -> ProposalResponse:
    try:
        proposal = await create_proposal(
            session,
            cluster_id=payload.cluster_id,
            title=payload.title,
            description=payload.description,
            priority_score=payload.priority_score,
            impact=payload.impact,
            effort=payload.effort,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ProposalResponse.model_validate(proposal)


@router.post("/proposals/{proposal_id}/citations", status_code=201)
async def add_citation_endpoint(
    proposal_id: UUID,
    problem_id: UUID = Query(...),
    relevance_note: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        citation = await add_citation(session, proposal_id, problem_id, relevance_note)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"id": citation.id, "proposal_id": proposal_id, "problem_id": problem_id}


@router.get("/proposals/{proposal_id}", response_model=ProposalDetailResponse)
async def get_proposal_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ProposalDetailResponse:
    """Return proposal detail with citations."""
    proposal = await get_proposal_detail(session, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return ProposalDetailResponse.model_validate(proposal)


@router.get("/roadmap", response_model=RoadmapResponse)
async def roadmap_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> RoadmapResponse:
    items, total = await get_roadmap(session, page=page, per_page=per_page)
    return RoadmapResponse(
        items=[
            RoadmapItem(
                proposal=ProposalResponse.model_validate(item["proposal"]),
                cluster_label=item["cluster_label"],
                mention_count=item["mention_count"],
                priority_score=item["priority_score"],
            )
            for item in items
        ],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=max(1, math.ceil(total / per_page)),
    )
