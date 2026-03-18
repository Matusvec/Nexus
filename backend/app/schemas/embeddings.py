from pydantic import BaseModel


class EmbedProblemsRequest(BaseModel):
    limit: int | None = None
