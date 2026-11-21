from limiter import SlidingWindowRateLimiter


def test_enforces_exact_max_requests() -> None:
    rl = SlidingWindowRateLimiter(max_requests=3, window_seconds=10.0)
    assert rl.allow("alice", now=1.0) is True
    assert rl.allow("alice", now=2.0) is True
    assert rl.allow("alice", now=3.0) is True
    # 4th request inside the 10s window MUST be rejected
    assert rl.allow("alice", now=4.0) is False
    assert rl.remaining("alice", now=4.0) == 0


def test_allows_after_window_expires() -> None:
    rl = SlidingWindowRateLimiter(max_requests=2, window_seconds=5.0)
    assert rl.allow("bob", now=10.0) is True
    assert rl.allow("bob", now=11.0) is True
    assert rl.allow("bob", now=12.0) is False
    # At t=15.0, request at t=10.0 has expired (15 - 5 == 10)
    assert rl.allow("bob", now=15.0) is True


def test_clients_are_isolated() -> None:
    rl = SlidingWindowRateLimiter(max_requests=1, window_seconds=10.0)
    assert rl.allow("c1", now=1.0) is True
    assert rl.allow("c1", now=2.0) is False
    assert rl.allow("c2", now=2.0) is True
