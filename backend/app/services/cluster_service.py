"""Clustering and proposal services — Phase 2 scaffold.

Implements threshold-based clustering of problem embeddings and
CRUD for feature proposals with citation provenance.
"""

import logging
from uuid import UUID

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.clusters import (
    ClusterMembership,
    FeatureProposal,
    ProblemCluster,
    ProposalCitation,
)
from app.models.embeddings import ProblemEmbedding
from app.models.problems import ProblemMention

logger = logging.getLogger(__name__)


# ── Clustering ──────────────────────────────────────────────────

async def run_threshold_clustering(
    session: AsyncSession,
    threshold: float = 0.75,
) -> list[ProblemCluster]:
    """Simple greedy threshold clustering.

    1. Load all problem embeddings.
    2. For each embedding, find the nearest existing cluster centroid.
    3. If similarity >= threshold, add to that cluster; otherwise create a new one.

    This is a Phase 2 placeholder — upgrade to HDBSCAN or similar later.
    """
    rows = (
        await session.execute(
            select(ProblemEmbedding).options(
                selectinload(ProblemEmbedding.problem)
            )
        )
    ).scalars().all()

    if not rows:
        return []

    clusters: list[ProblemCluster] = []

    for pe in rows:
        vec = np.array(pe.embedding, dtype=np.float32)
        best_cluster = None
        best_sim = -1.0

        for cluster in clusters:
            if cluster.centroid is not None:
                centroid = np.array(cluster.centroid, dtype=np.float32)
                sim = float(np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-9))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster

        if best_cluster and best_sim >= threshold:
            # Add to existing cluster
            membership = ClusterMembership(
                cluster_id=best_cluster.id,
                problem_id=pe.problem_id,
                similarity=best_sim,
            )
            session.add(membership)
            best_cluster.mention_count += 1
            # Update centroid as running mean
            old_c = np.array(best_cluster.centroid, dtype=np.float32)
            n = best_cluster.mention_count
            new_c = old_c + (vec - old_c) / n
            best_cluster.centroid = new_c.tolist()
        else:
            # Create new cluster
            cluster = ProblemCluster(
                label=pe.problem.problem_statement[:120] if pe.problem else "Unnamed",
                summary=None,
                centroid=vec.tolist(),
                threshold=threshold,
                mention_count=1,
            )
            session.add(cluster)
            await session.flush()
            membership = ClusterMembership(
                cluster_id=cluster.id,
                problem_id=pe.problem_id,
                similarity=1.0,
            )
            session.add(membership)
            clusters.append(cluster)

    await session.commit()
    logger.info("Created %d clusters from %d embeddings", len(clusters), len(rows))
    return clusters


async def list_clusters(
    session: AsyncSession,
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[ProblemCluster], int]:
    total = (await session.execute(select(func.count(ProblemCluster.id)))).scalar() or 0

    query = (
        select(ProblemCluster)
        .order_by(ProblemCluster.mention_count.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    items = (await session.execute(query)).scalars().all()
    return list(items), total


async def get_cluster_detail(
    session: AsyncSession, cluster_id: UUID
) -> ProblemCluster | None:
    query = (
        select(ProblemCluster)
        .options(
            selectinload(ProblemCluster.members),
            selectinload(ProblemCluster.proposals),
        )
        .where(ProblemCluster.id == cluster_id)
    )
    return (await session.execute(query)).scalar_one_or_none()


# ── Proposals ───────────────────────────────────────────────────

async def create_proposal(
    session: AsyncSession,
    cluster_id: UUID,
    title: str,
    description: str,
    priority_score: float | None = None,
    impact: str | None = None,
    effort: str | None = None,
) -> FeatureProposal:
    proposal = FeatureProposal(
        cluster_id=cluster_id,
        title=title,
        description=description,
        priority_score=priority_score,
        impact=impact,
        effort=effort,
    )
    session.add(proposal)
    await session.commit()
    return proposal


async def add_citation(
    session: AsyncSession,
    proposal_id: UUID,
    problem_id: UUID,
    relevance_note: str | None = None,
) -> ProposalCitation:
    citation = ProposalCitation(
        proposal_id=proposal_id,
        problem_id=problem_id,
        relevance_note=relevance_note,
    )
    session.add(citation)
    await session.commit()
    return citation


async def get_roadmap(
    session: AsyncSession,
) -> list[dict]:
    """Return all proposals ordered by priority, joined with cluster info."""
    query = (
        select(FeatureProposal, ProblemCluster.label, ProblemCluster.mention_count)
        .join(ProblemCluster, FeatureProposal.cluster_id == ProblemCluster.id)
        .order_by(FeatureProposal.priority_score.desc().nullslast())
    )
    rows = (await session.execute(query)).all()
    return [
        {
            "proposal": proposal,
            "cluster_label": label,
            "mention_count": count,
            "priority_score": proposal.priority_score,
        }
        for proposal, label, count in rows
    ]
