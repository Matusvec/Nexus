"""Clustering and proposal services — Phase 2 scaffold.

Implements threshold-based clustering of problem embeddings and
CRUD for feature proposals with citation provenance.
Includes LLM-based cluster summarization (strategy Section D).
"""

import logging
from uuid import UUID

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.llm.client import get_client
from app.models.clusters import (
    ClusterMembership,
    FeatureProposal,
    ProblemCluster,
    ProposalCitation,
)
from app.models.embeddings import ProblemEmbedding
from app.models.problems import ProblemMention
from app.utils.retry import call_with_retry

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
    # C1 fix: clear old clusters before re-run to prevent duplicates.
    # FK CASCADE on ClusterMembership, FeatureProposal, and ProposalCitation
    # will clean up related rows automatically.
    old_count = (await session.execute(select(func.count(ProblemCluster.id)))).scalar() or 0
    if old_count:
        logger.info("Clearing %d existing clusters before re-clustering", old_count)
        await session.execute(delete(ProblemCluster))
        await session.flush()

    # A4 fix: deterministic ordering for reproducible clustering
    rows = (
        await session.execute(
            select(ProblemEmbedding).options(
                selectinload(ProblemEmbedding.problem)
            ).order_by(ProblemEmbedding.created_at.asc())
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

async def get_proposal_detail(
    session: AsyncSession, proposal_id: UUID
) -> FeatureProposal | None:
    """Return a single proposal with its citations eagerly loaded."""
    query = (
        select(FeatureProposal)
        .options(selectinload(FeatureProposal.citations))
        .where(FeatureProposal.id == proposal_id)
    )
    return (await session.execute(query)).scalar_one_or_none()


async def create_proposal(
    session: AsyncSession,
    cluster_id: UUID,
    title: str,
    description: str,
    priority_score: float | None = None,
    impact: str | None = None,
    effort: str | None = None,
) -> FeatureProposal:
    # M2 fix: validate cluster_id exists before insert
    cluster = await session.get(ProblemCluster, cluster_id)
    if not cluster:
        raise ValueError(f"Cluster {cluster_id} not found")
    # A8 fix: warn if cluster has no members
    if cluster.mention_count == 0:
        logger.warning("Creating proposal for empty cluster %s (mention_count=0)", cluster_id)
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
    # M3 fix: validate FK targets exist before insert
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        raise ValueError(f"Proposal {proposal_id} not found")
    problem = await session.get(ProblemMention, problem_id)
    if not problem:
        raise ValueError(f"Problem mention {problem_id} not found")
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
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[dict], int]:
    """Return paginated proposals ordered by priority, joined with cluster info."""
    # A6 fix: add pagination to avoid unbounded result sets
    total = (await session.execute(select(func.count(FeatureProposal.id)))).scalar() or 0

    query = (
        select(FeatureProposal, ProblemCluster.label, ProblemCluster.mention_count)
        .join(ProblemCluster, FeatureProposal.cluster_id == ProblemCluster.id)
        .order_by(FeatureProposal.priority_score.desc().nullslast())
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    rows = (await session.execute(query)).all()
    items = [
        {
            "proposal": proposal,
            "cluster_label": label,
            "mention_count": count,
            "priority_score": proposal.priority_score,
        }
        for proposal, label, count in rows
    ]
    return items, total


# ── Cluster summarization ───────────────────────────────────────

async def summarize_cluster(
    session: AsyncSession,
    cluster_id: UUID,
) -> ProblemCluster:
    """Generate an LLM-based summary and label for a cluster (strategy Section D).

    Loads member problem mentions, asks the LLM for a short label,
    2-3 sentence summary, and top quotes, then updates the cluster row.
    """
    cluster = (
        await session.execute(
            select(ProblemCluster)
            .options(selectinload(ProblemCluster.members))
            .where(ProblemCluster.id == cluster_id)
        )
    ).scalar_one_or_none()

    if not cluster:
        raise ValueError(f"Cluster {cluster_id} not found")
    if not cluster.members:
        logger.warning("Cluster %s has no members — skipping summarization", cluster_id)
        return cluster

    # Load member problem mentions
    member_ids = [m.problem_id for m in cluster.members]
    members_q = select(ProblemMention).where(ProblemMention.id.in_(member_ids))
    members = (await session.execute(members_q)).scalars().all()

    formatted_mentions = "\n".join(
        f"- [{m.severity}] {m.problem_statement}\n  Quote: \"{m.quote_text}\""
        for m in members
    )

    prompt = (
        "Given these customer problem mentions:\n"
        f"{formatted_mentions}\n\n"
        "Generate:\n"
        "1. label: A short (3-8 word) actionable label for this pain cluster\n"
        "2. summary: A 2-3 sentence summary of the core issue\n"
        "3. top_quotes: The 3 most compelling direct quotes\n\n"
        "Return valid JSON only:\n"
        '{"label": "string", "summary": "string", "top_quotes": ["string"]}\n'
    )

    client = get_client()
    raw = await call_with_retry(
        client.generate_json, prompt, "summarize_cluster_v1",
        label="Cluster summarization",
    )

    cluster.label = raw.get("label", cluster.label)
    cluster.summary = raw.get("summary")

    # Merge top quotes and tags into cluster tags for discoverability
    top_quotes = raw.get("top_quotes", [])
    if top_quotes:
        cluster.metadata_ = {**(cluster.metadata_ or {}), "top_quotes": top_quotes}

    await session.commit()
    logger.info("Summarized cluster %s: '%s'", cluster_id, cluster.label)
    return cluster


async def summarize_all_clusters(session: AsyncSession) -> int:
    """Summarize all clusters that lack a summary."""
    query = select(ProblemCluster).where(ProblemCluster.summary.is_(None))
    clusters = (await session.execute(query)).scalars().all()
    count = 0
    for cluster in clusters:
        try:
            await summarize_cluster(session, cluster.id)
            count += 1
        except Exception:  # noqa: BLE001
            logger.warning("Failed to summarize cluster %s", cluster.id, exc_info=True)
    return count
