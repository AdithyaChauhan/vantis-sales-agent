"""In-memory rate limiter with fixed windows (one Render instance, one worker; see PLAN.md part 3.11).

Each key (a client IP) gets, per limit, a window start time and a count. The window starts at the
key's first request and the count resets when the window has passed. At a window boundary up to twice
the limit can pass in a short burst; that is accepted for this demo. Counts live in this process only,
so they reset when the service restarts or sleeps; several instances would need a shared store (Redis).
"""
import threading
import time
from typing import Callable

MAX_KEYS = 10_000  # above this, expired keys are dropped so memory stays bounded


class RateLimiter:
    """At most `count` requests per `seconds` for each key, for every (count, seconds) in `limits`."""

    def __init__(self, limits: list[tuple[int, float]], clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.clock = clock
        self._windows: dict[str, list[list[float]]] = {}  # key -> [[window_start, count], ...] per limit
        self._lock = threading.Lock()  # sync endpoints run in a thread pool

    def hit(self, key: str) -> float | None:
        """Count one request for `key`. If a limit is full, count nothing and return the seconds until
        that limit frees up; otherwise return None."""
        with self._lock:
            now = self.clock()
            windows = self._current(key, now)
            wait = self._wait(windows, now)
            if wait is None:
                for window in windows:
                    window[1] += 1
            return wait

    def blocked(self, key: str) -> float | None:
        """The seconds until `key` may make another request, or None. Counts nothing."""
        with self._lock:
            now = self.clock()
            return self._wait(self._current(key, now), now)

    def reset(self) -> None:
        """Forget every key (tests)."""
        with self._lock:
            self._windows.clear()

    def _current(self, key: str, now: float) -> list[list[float]]:
        """The key's windows, with any that have passed restarted at `now`."""
        if key not in self._windows:
            if len(self._windows) >= MAX_KEYS:
                self._prune(now)
            self._windows[key] = [[now, 0] for _ in self.limits]
        windows = self._windows[key]
        for window, (_, seconds) in zip(windows, self.limits):
            if now - window[0] >= seconds:
                window[0], window[1] = now, 0
        return windows

    def _wait(self, windows: list[list[float]], now: float) -> float | None:
        waits = [
            start + seconds - now
            for (start, used), (count, seconds) in zip(windows, self.limits)
            if used >= count
        ]
        return max(waits) if waits else None

    def _prune(self, now: float) -> None:
        """Drop keys whose windows have all passed (they would start fresh anyway)."""
        expired = [
            key for key, windows in self._windows.items()
            if all(now - start >= seconds for (start, _), (_, seconds) in zip(windows, self.limits))
        ]
        for key in expired:
            del self._windows[key]
