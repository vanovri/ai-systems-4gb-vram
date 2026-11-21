"""Sliding window rate limiter (Benchmark 2: off-by-one capacity bug)."""

from __future__ import annotations

from collections import deque


class SlidingWindowRateLimiter:
    """Allows at most `max_requests` per client within `window_seconds`."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, client_id: str, now: float) -> bool:
        bucket = self._buckets.setdefault(client_id, deque())
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) > self.max_requests:
            return False

        bucket.append(now)
        return True

    def remaining(self, client_id: str, now: float) -> int:
        bucket = self._buckets.get(client_id)
        if not bucket:
            return self.max_requests
        cutoff = now - self.window_seconds
        active = sum(1 for ts in bucket if ts > cutoff)
        return max(0, self.max_requests - active)
