from pydantic import BaseModel


class EmbedProblemsRequest(BaseModel):
    limit: int | None = None


class EmbedProblemsResponse(BaseModel):
    embedded_count: int
    message: str
