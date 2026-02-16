"""Authentication middleware — Phase 4 scaffold.

Supports API key auth via `X-API-Key` header with a migration path to JWT.
When `API_KEYS` env var is not set, auth is disabled (development mode).
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_allowed_keys() -> set[str]:
    """Load API keys from settings. Returns empty set if not configured."""
    raw = getattr(settings, "api_keys", None)
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """Dependency that validates the API key.

    If no keys are configured (dev mode), all requests pass through.
    Otherwise, the X-API-Key header must match one of the configured keys.
    """
    allowed = _get_allowed_keys()
    if not allowed:
        # No keys configured — auth disabled (dev mode)
        return None

    if not api_key or api_key not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


# Type alias for use as a FastAPI dependency
RequireAuth = Annotated[str | None, Depends(verify_api_key)]
