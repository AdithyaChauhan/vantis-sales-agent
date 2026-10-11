"""RateLimiter with a fake clock (no sleeping)."""
from app.core import rate_limit
from app.core.rate_limit import RateLimiter


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_says_how_long_to_wait():
    clock = Clock()
    limiter = RateLimiter([(3, 60)], clock=clock)
    assert [limiter.hit("a") for _ in range(3)] == [None, None, None]
    clock.now += 20
    assert limiter.hit("a") == 40  # the window started at the first request


def test_window_resets_after_it_passes():
    clock = Clock()
    limiter = RateLimiter([(1, 60)], clock=clock)
    assert limiter.hit("a") is None
    assert limiter.hit("a") == 60
    clock.now += 60
    assert limiter.hit("a") is None


def test_keys_are_counted_separately():
    limiter = RateLimiter([(1, 60)], clock=Clock())
    assert limiter.hit("a") is None
    assert limiter.hit("b") is None
    assert limiter.hit("a") == 60


def test_day_limit_still_blocks_after_the_minute_limit_frees_up():
    clock = Clock()
    limiter = RateLimiter([(2, 60), (3, 86_400)], clock=clock)
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a") == 60
    clock.now += 60
    assert limiter.hit("a") is None  # third of the day
    clock.now += 60
    assert limiter.hit("a") == 86_400 - 120


def test_a_blocked_request_is_not_counted():
    clock = Clock()
    limiter = RateLimiter([(1, 60), (2, 86_400)], clock=clock)
    assert limiter.hit("a") is None
    for _ in range(5):
        assert limiter.hit("a") is not None  # refused by the minute limit, never counted for the day
    clock.now += 60
    assert limiter.hit("a") is None  # the second of the day still fits


def test_blocked_counts_nothing():
    limiter = RateLimiter([(1, 60)], clock=Clock())
    for _ in range(3):
        assert limiter.blocked("a") is None
    assert limiter.hit("a") is None
    assert limiter.blocked("a") == 60


def test_expired_keys_are_pruned(monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_KEYS", 3)
    clock = Clock()
    limiter = RateLimiter([(1, 60)], clock=clock)
    for key in "abc":
        limiter.hit(key)
    clock.now += 61
    limiter.hit("d")  # over MAX_KEYS: a, b and c have expired and are dropped
    assert set(limiter._windows) == {"d"}


def test_reset_forgets_everything():
    limiter = RateLimiter([(1, 60)], clock=Clock())
    limiter.hit("a")
    limiter.reset()
    assert limiter.hit("a") is None
