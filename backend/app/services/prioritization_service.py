"""Prioritization engine — strategy Section G.

Ranks feature proposals with transparent, explainable scoring:
    final_score = (frequency_score × severity_score × strategic_weight) / effort_estimate
"""

import logging
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.clusters import FeatureProposal, ProblemCluster
from app.models.priority_scores import PriorityScore
from app.models.problems import ProblemMention

logger = logging.getLogger(__name__)

# Severity weights for computing weighted average
_SEVERITY_WEIGHTS = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# Effort mapping from scope_estimate (or effort field) to numeric value
_EFFORT_MAP = {"S": 1, "M": 3, "L": 8, "XL": 20}


async def calculate_priority(
    session: AsyncSession,
    proposal: FeatureProposal,
    cluster: ProblemCluster,
) -> PriorityScore:
    """Calculate and store a priority score for a proposal.

    The scoring formula follows the strategy document (Section G):
        final_score = (frequency × severity × weight) / effort
    """
    # 1. Frequency: cluster mention_count / total mentions (normalized 0-100)
    total_mentions = (
        await session.execute(select(func.count(ProblemMention.id)))
    ).scalar() or 1  # avoid division by zero
    frequency_score = (cluster.mention_count / total_mentions) * 100

    # 2. Severity: weighted average of cluster member severities
    from app.models.clusters import ClusterMembership
    sev_q = (
        select(ProblemMention.severity, func.count(ProblemMention.id))
        .join(ClusterMembership, ClusterMembership.problem_id == ProblemMention.id)
        .where(ClusterMembership.cluster_id == cluster.id)
        .group_by(ProblemMention.severity)
    )
    sev_rows = (await session.execute(sev_q)).all()
    severity_distribution = {sev: cnt for sev, cnt in sev_rows}
    total_weighted = sum(
        _SEVERITY_WEIGHTS.get(sev, 1) * cnt for sev, cnt in sev_rows
    )
    total_count = sum(cnt for _, cnt in sev_rows) or 1
    severity_score = total_weighted / total_count

    # 3. Strategic weight (default 1.0, PM can override via PATCH)
    strategic_weight = 1.0

    # 4. Effort: derived from proposal effort field
    effort_str = (proposal.effort or "M").upper()
    effort_estimate = _EFFORT_MAP.get(effort_str, 3)

    # 5. Final score
    final_score = (frequency_score * severity_score * strategic_weight) / effort_estimate

    score_breakdown = {
        "formula": "(frequency × severity × weight) / effort",
        "frequency": {
            "value": round(frequency_score, 2),
            "explanation": f"{cluster.mention_count} mentions out of {total_mentions}",
        },
        "severity": {
            "value": round(severity_score, 2),
            "distribution": severity_distribution,
        },
        "weight": {"value": strategic_weight, "reason": "default"},
        "effort": {"value": effort_estimate, "scope": effort_str},
        "final": round(final_score, 4),
    }

    # Upsert priority score
    existing = (
        await session.execute(
            select(PriorityScore).where(PriorityScore.proposal_id == proposal.id)
        )
    ).scalar_one_or_none()

    if existing:
        existing.frequency_score = frequency_score
        existing.severity_score = severity_score
        existing.strategic_weight = strategic_weight
        existing.effort_estimate = effort_estimate
        existing.final_score = final_score
        existing.score_breakdown = score_breakdown
        priority = existing
    else:
        priority = PriorityScore(
            proposal_id=proposal.id,
            frequency_score=frequency_score,
            severity_score=severity_score,
            strategic_weight=strategic_weight,
            effort_estimate=effort_estimate,
            final_score=final_score,
            score_breakdown=score_breakdown,
        )
        session.add(priority)

    # Also update the proposal's priority_score for quick sorting
    await session.execute(
        update(FeatureProposal)
        .where(FeatureProposal.id == proposal.id)
        .values(priority_score=final_score)
    )

    await session.flush()
    return priority


async def score_all_proposals(session: AsyncSession) -> list[PriorityScore]:
    """Recalculate priority scores for all proposals."""
    proposals = (
        await session.execute(
            select(FeatureProposal).join(
                ProblemCluster, FeatureProposal.cluster_id == ProblemCluster.id
            )
        )
    ).scalars().all()

    scores = []
    for proposal in proposals:
        cluster = await session.get(ProblemCluster, proposal.cluster_id)
        if cluster:
            score = await calculate_priority(session, proposal, cluster)
            scores.append(score)

    await session.commit()
    logger.info("Scored %d proposals", len(scores))
    return scores


async def update_strategic_weight(
    session: AsyncSession,
    proposal_id: UUID,
    weight: float,
) -> PriorityScore | None:
    """Update the strategic weight for a proposal and recalculate its score."""
    proposal = await session.get(FeatureProposal, proposal_id)
    if not proposal:
        return None

    cluster = await session.get(ProblemCluster, proposal.cluster_id)
    if not cluster:
        return None

    # Recalculate with new weight
    score = await calculate_priority(session, proposal, cluster)
    score.strategic_weight = weight

    # Recompute final score with new weight
    score.final_score = (
        score.frequency_score * score.severity_score * weight
    ) / score.effort_estimate
    score.score_breakdown["weight"] = {"value": weight, "reason": "manual adjustment"}
    score.score_breakdown["final"] = round(score.final_score, 4)

    # Update proposal priority_score
    await session.execute(
        update(FeatureProposal)
        .where(FeatureProposal.id == proposal_id)
        .values(priority_score=score.final_score)
    )

    await session.commit()
    return score
