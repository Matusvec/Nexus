"""Rate limiting middleware — Phase 4 scaffold.

Simple in-memory token-bucket rate limiter. For production, replace
with Redis-backed sliding window (e.g. via `slowapi` or custom Redis logic).
"""

import time
import logging
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter keyed by client IP.

    Args:
        app: FastAPI/Starlette app
        rate: Max requests per window
        window: Time window in seconds
    """

    def __init__(self, app, rate: int = 60, window: int = 60) -> None:
        super().__init__(app)
        self.rate = rate
        self.window = window
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._last_prune: float = 0.0
        self._prune_interval: float = 300.0  # prune stale IPs every 5 minutes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # O10 fix: periodically prune stale IP buckets to prevent memory leak
        if now - self._last_prune > self._prune_interval:
            self._prune_stale_buckets(now)

        # Clean old entries
        bucket = self._buckets[client_ip]
        cutoff = now - self.window
        self._buckets[client_ip] = [t for t in bucket if t > cutoff]
        bucket = self._buckets[client_ip]

        if len(bucket) >= self.rate:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return Response(
                content='{"detail":"Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(self.window),
                    "X-RateLimit-Limit": str(self.rate),
                    "X-RateLimit-Remaining": "0",
                },
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.rate - len(bucket))
        )
        return response

    def _prune_stale_buckets(self, now: float) -> None:
        """Remove IP buckets that have no recent requests (O10 fix)."""
        cutoff = now - self.window
        stale_ips = [ip for ip, ts_list in self._buckets.items() if not ts_list or ts_list[-1] < cutoff]
        for ip in stale_ips:
            del self._buckets[ip]
        if stale_ips:
            logger.debug("Pruned %d stale rate-limit buckets", len(stale_ips))
        self._last_prune = now
