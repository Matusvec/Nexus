from uuid import UUID

from pydantic import BaseModel


class EmbedProblemsRequest(BaseModel):
    limit: int | None = None


class EmbedProblemsResponse(BaseModel):
    embedded_count: int
    problem_ids: list[UUID]
