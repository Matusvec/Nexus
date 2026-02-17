"""Proposal generation service — strategy Section E.

Converts pain clusters into structured feature proposals using LLM,
with citation verification and version tracking.
"""

import asyncio
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.llm.client import get_client
from app.models.clusters import (
    ClusterMembership,
    FeatureProposal,
    ProblemCluster,
    ProposalCitation,
    ProposalVersion,
)
from app.models.problems import ProblemMention
from app.schemas.problems import ProblemMentionCreate
from app.utils.retry import call_with_retry

logger = logging.getLogger(__name__)
PROMPT_VERSION = "generate_proposal_v1"


def _build_proposal_prompt(
    cluster: ProblemCluster,
    members: list[ProblemMention],
) -> str:
    """Build the LLM prompt for proposal generation."""
    # Format mentions
    formatted_mentions = "\n".join(
        f"- [{m.severity}] {m.problem_statement}\n  Quote: \"{m.quote_text}\""
        for m in members
    )

    # Severity distribution
    sev_counts: dict[str, int] = {}
    for m in members:
        sev_counts[m.severity] = sev_counts.get(m.severity, 0) + 1

    return (
        "You are a senior product manager. Based on the following customer pain cluster,\n"
        "generate a structured feature proposal.\n\n"
        f"CLUSTER:\n"
        f"- Label: {cluster.label}\n"
        f"- Summary: {cluster.summary or 'N/A'}\n"
        f"- Member count: {cluster.mention_count}\n"
        f"- Severity distribution: {sev_counts}\n"
        f"- Problem mentions:\n{formatted_mentions}\n\n"
        "Return valid JSON only, with this schema:\n"
        "{\n"
        '  "feature_name": "string",\n'
        '  "one_liner": "string",\n'
        '  "user_story": "As a [persona], I want [goal] so that [benefit]",\n'
        '  "rationale": "string — MUST cite quotes using [Quote: \\"...\\"]. Do not invent quotes.",\n'
        '  "impact": "high|medium|low",\n'
        '  "effort": "S|M|L|XL",\n'
        '  "risks": [{"risk": "string", "mitigation": "string"}],\n'
        '  "success_metrics": [{"metric": "string", "target": "string"}]\n'
        "}\n"
    )


async def generate_proposal_for_cluster(
    session: AsyncSession,
    cluster_id: UUID,
) -> FeatureProposal:
    """Generate a feature proposal from a cluster using LLM.

    1. Load the cluster and its member problem mentions
    2. Call LLM with the proposal generation prompt
    3. Parse the response and create a FeatureProposal
    4. Link citations to source problem mentions
    5. Create initial proposal version snapshot
    """
    # Load cluster with members
    cluster = (
        await session.execute(
            select(ProblemCluster)
            .options(selectinload(ProblemCluster.members))
            .where(ProblemCluster.id == cluster_id)
        )
    ).scalar_one_or_none()

    if not cluster:
        raise ValueError(f"Cluster {cluster_id} not found")
    if cluster.mention_count == 0:
        raise ValueError(f"Cluster {cluster_id} has no members — cannot generate proposal")

    # Load the actual problem mentions for the cluster
    member_problem_ids = [m.problem_id for m in cluster.members]
    members_q = select(ProblemMention).where(ProblemMention.id.in_(member_problem_ids))
    members = (await session.execute(members_q)).scalars().all()

    if not members:
        raise ValueError(f"No problem mentions found for cluster {cluster_id}")

    # Generate proposal via LLM
    client = get_client()
    prompt = _build_proposal_prompt(cluster, list(members))
    raw = await call_with_retry(
        client.generate_json, prompt, PROMPT_VERSION,
        label="Proposal generation",
    )

    # Create proposal
    proposal = FeatureProposal(
        cluster_id=cluster_id,
        title=raw.get("feature_name", "Untitled Proposal"),
        description=raw.get("rationale", ""),
        impact=raw.get("impact"),
        effort=raw.get("effort"),
        metadata_={
            "one_liner": raw.get("one_liner"),
            "user_story": raw.get("user_story"),
            "risks": raw.get("risks", []),
            "success_metrics": raw.get("success_metrics", []),
        },
    )
    session.add(proposal)
    await session.flush()

    # Auto-cite all cluster members as supporting evidence
    for member in members:
        citation = ProposalCitation(
            proposal_id=proposal.id,
            problem_id=member.id,
            relevance_note=f"Cluster member (severity: {member.severity})",
        )
        session.add(citation)

    # Create initial version snapshot
    version = ProposalVersion(
        proposal_id=proposal.id,
        version_number=1,
        snapshot={
            "title": proposal.title,
            "description": proposal.description,
            "impact": proposal.impact,
            "effort": proposal.effort,
            "metadata": proposal.metadata_,
        },
        change_reason="Initial LLM generation",
    )
    session.add(version)

    await session.commit()
    logger.info(
        "Generated proposal '%s' for cluster %s with %d citations",
        proposal.title, cluster_id, len(members),
    )
    return proposal
