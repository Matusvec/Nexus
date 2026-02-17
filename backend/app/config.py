from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # Auth — comma-separated API keys; empty = auth disabled (dev mode)
    api_keys: str = ""

    # CORS — comma-separated origins; empty = localhost defaults
    cors_origins: str = ""

    # Rate limiting
    rate_limit_requests: int = 60
    rate_limit_window: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("database_url")
    @classmethod
    def normalize_db_url(cls, value: str) -> str:
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+asyncpg://", 1)
        return value


settings = Settings()
