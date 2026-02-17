"""Shared retry-with-backoff utility and rate limiting for LLM API calls."""

import asyncio
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter for outgoing API calls.

    Ensures no more than ``max_calls`` calls within any rolling ``period``
    seconds. Awaiting ``acquire()`` blocks until a slot is available.
    """

    def __init__(self, max_calls: int = 55, period: float = 60.0) -> None:
        self._max_calls = max_calls
        self._period = period
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self._period
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self._max_calls:
                wait = self._timestamps[0] - cutoff
                logger.debug("Rate limiter: sleeping %.1fs to stay under %d RPM", wait, self._max_calls)
                await asyncio.sleep(wait)
            self._timestamps.append(time.monotonic())


# Default global rate limiter (55 calls/min — under Gemini's 60 RPM free tier)
_global_limiter = RateLimiter(max_calls=55, period=60.0)


async def call_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    label: str = "LLM call",
    rate_limit: bool = True,
) -> Any:
    """Call *func* in a thread with exponential backoff retries.

    Args:
        func: Synchronous callable to run via ``asyncio.to_thread``.
        *args: Positional arguments forwarded to *func*.
        max_retries: Total number of attempts before giving up.
        initial_delay: Seconds to sleep after the first failure (doubles each retry).
        label: Human-readable label for log messages.
        rate_limit: Whether to acquire the global rate limiter before each call.

    Returns:
        The return value of *func*.

    Raises:
        The last exception if all retries are exhausted.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            if rate_limit:
                await _global_limiter.acquire()
            return await asyncio.to_thread(func, *args)
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            logger.info(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                label, attempt, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
            delay *= 2
