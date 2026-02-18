"""Phase 2 routes: clusters, proposals, roadmap."""

import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.clusters import FeatureProposal, ProposalVersion
from app.schemas.clusters import (
    CitationResponse,
    ClusterDetailResponse,
    ClusterResponse,
    ProposalCreate,
    ProposalDetailResponse,
    ProposalResponse,
    ProposalUpdate,
    ProposalVersionResponse,
    RoadmapItem,
    RoadmapResponse,
)
from app.schemas.priority_scores import PriorityScoreResponse, StrategicWeightUpdate
from app.services.cluster_service import (
    add_citation,
    create_proposal,
    get_cluster_detail,
    get_proposal_detail,
    get_roadmap,
    list_clusters,
    list_proposals,
    run_hdbscan_clustering,
    run_threshold_clustering,
    summarize_cluster,
)
from app.services.prioritization_service import (
    score_all_proposals,
    update_strategic_weight,
)
from app.services.proposal_service import generate_proposal_for_cluster
from app.services.task_tree_service import generate_tasks_for_proposal

router = APIRouter()


@router.post("/clusters/run")
async def run_clustering_endpoint(
    threshold: float = Query(0.75, ge=0.0, le=1.0),
    algorithm: str = Query("auto"),  # "threshold", "hdbscan", "auto"
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Trigger clustering on all problem embeddings.

    algorithm: 'threshold' (greedy), 'hdbscan' (density-based), or 'auto' (picks based on count).
    """
    if algorithm == "auto":
        from sqlalchemy import func as sqlfunc
        from app.models.embeddings import ProblemEmbedding
        count = (await session.execute(
            select(sqlfunc.count(ProblemEmbedding.id))
        )).scalar() or 0
        algorithm = "hdbscan" if count > 500 else "threshold"

    if algorithm == "hdbscan":
        clusters = await run_hdbscan_clustering(session)
    else:
        clusters = await run_threshold_clustering(session, threshold=threshold)

    return {"clusters_created": len(clusters), "algorithm": algorithm}


@router.post("/clusters/run_hdbscan")
async def run_hdbscan_endpoint(
    min_cluster_size: int = Query(3, ge=2, le=50),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Trigger HDBSCAN clustering on all problem embeddings."""
    clusters = await run_hdbscan_clustering(session, min_cluster_size=min_cluster_size)
    return {"clusters_created": len(clusters), "algorithm": "hdbscan"}


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


# ── LLM-powered generation endpoints ────────────────────────

@router.post("/clusters/{cluster_id}/summarize")
async def summarize_cluster_endpoint(
    cluster_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Generate an LLM-based summary for a cluster (strategy Section D)."""
    try:
        cluster = await summarize_cluster(session, cluster_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "cluster_id": cluster.id,
        "label": cluster.label,
        "summary": cluster.summary,
    }


@router.post("/clusters/{cluster_id}/generate_proposal", response_model=ProposalResponse)
async def generate_proposal_endpoint(
    cluster_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ProposalResponse:
    """Generate a feature proposal from a cluster using LLM (strategy Section E)."""
    try:
        proposal = await generate_proposal_for_cluster(session, cluster_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ProposalResponse.model_validate(proposal)


@router.post("/proposals/{proposal_id}/generate_tasks")
async def generate_tasks_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Generate a task tree for a proposal using LLM (strategy Section F)."""
    try:
        tasks = await generate_tasks_for_proposal(session, proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"proposal_id": str(proposal_id), "tasks_created": len(tasks)}


@router.post("/roadmap/score")
async def score_all_endpoint(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Recalculate priority scores for all proposals (strategy Section G)."""
    scores = await score_all_proposals(session)
    return {"scored_count": len(scores)}


@router.patch("/roadmap/{proposal_id}/weight", response_model=PriorityScoreResponse)
async def update_weight_endpoint(
    proposal_id: UUID,
    payload: StrategicWeightUpdate,
    session: AsyncSession = Depends(get_session),
) -> PriorityScoreResponse:
    """Adjust the strategic weight for a proposal and recalculate its score."""
    score = await update_strategic_weight(session, proposal_id, payload.strategic_weight)
    if not score:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return PriorityScoreResponse.model_validate(score)


# ── Proposal lifecycle endpoints ────────────────────────────────

@router.get("/proposals")
async def list_proposals_endpoint(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """List all proposals with optional status filter and pagination."""
    items, total = await list_proposals(session, page=page, per_page=per_page, status=status)
    return {
        "items": [ProposalResponse.model_validate(p) for p in items],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, math.ceil(total / per_page)),
    }


@router.put("/proposals/{proposal_id}", response_model=ProposalResponse)
async def update_proposal_endpoint(
    proposal_id: UUID,
    payload: ProposalUpdate,
    session: AsyncSession = Depends(get_session),
) -> ProposalResponse:
    """Update a proposal (partial update). Creates a version snapshot before applying changes."""
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    # Save version snapshot of current state BEFORE applying changes
    version = ProposalVersion(
        proposal_id=proposal.id,
        version_number=proposal.version,
        snapshot={
            "title": proposal.title,
            "description": proposal.description,
            "impact": proposal.impact,
            "effort": proposal.effort,
            "status": proposal.status,
            "metadata": proposal.metadata_,
        },
        change_reason="Manual edit",
    )
    session.add(version)

    # Apply updates
    for field, value in payload.model_dump(exclude_none=True).items():
        if field == "metadata":
            setattr(proposal, "metadata_", value)
        else:
            setattr(proposal, field, value)
    proposal.version += 1

    await session.commit()
    return ProposalResponse.model_validate(proposal)


@router.delete("/proposals/{proposal_id}", status_code=204)
async def delete_proposal_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a proposal and all related data (citations, versions, tasks, scores)."""
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    await session.delete(proposal)
    await session.commit()


@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Approve a proposal — sets status to 'approved' and creates a version snapshot."""
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    version = ProposalVersion(
        proposal_id=proposal.id,
        version_number=proposal.version + 1,
        snapshot={
            "title": proposal.title,
            "description": proposal.description,
            "impact": proposal.impact,
            "effort": proposal.effort,
            "status": "approved",
            "metadata": proposal.metadata_,
        },
        change_reason="Approved by PM",
    )
    proposal.status = "approved"
    proposal.version += 1
    session.add(version)
    await session.commit()
    return {"proposal_id": str(proposal_id), "status": "approved"}


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Reject a proposal — sets status to 'rejected' and creates a version snapshot."""
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    version = ProposalVersion(
        proposal_id=proposal.id,
        version_number=proposal.version + 1,
        snapshot={
            "title": proposal.title,
            "description": proposal.description,
            "impact": proposal.impact,
            "effort": proposal.effort,
            "status": "rejected",
            "metadata": proposal.metadata_,
        },
        change_reason="Rejected by PM",
    )
    proposal.status = "rejected"
    proposal.version += 1
    session.add(version)
    await session.commit()
    return {"proposal_id": str(proposal_id), "status": "rejected"}


@router.post("/proposals/{proposal_id}/regenerate", response_model=ProposalResponse)
async def regenerate_proposal_endpoint(
    proposal_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ProposalResponse:
    """Regenerate a proposal using LLM, updating it in-place with version tracking."""
    try:
        proposal = await generate_proposal_for_cluster(
            session, proposal_id=proposal_id, existing_proposal_id=proposal_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ProposalResponse.model_validate(proposal)
