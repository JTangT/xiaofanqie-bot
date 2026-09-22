"""Tests for the network resilience layer (net.py)."""

from __future__ import annotations

import asyncio

import pytest
from telegram.error import BadRequest, Forbidden, InvalidToken, NetworkError, RetryAfter, TimedOut

from net import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_SIZE,
    build_get_updates_request,
    build_request,
    call_with_retry,
)


def test_build_request_overrides_ptb_defaults():
    """The whole point is to replace PTB's latency-hostile defaults."""
    import inspect

    from telegram.request import HTTPXRequest

    defaults = inspect.signature(HTTPXRequest.__init__).parameters
    # Sanity-check the assumption this module is built on.
    assert defaults["connection_pool_size"].default == 1
    assert defaults["connect_timeout"].default == 5.0

    request = build_request()
    kwargs = request._client_kwargs
    timeout = kwargs["timeout"]

    assert kwargs["limits"].max_connections == DEFAULT_POOL_SIZE > 1
    assert timeout.connect == DEFAULT_CONNECT_TIMEOUT > 5.0
    assert timeout.pool == 30.0 > 1.0
    assert timeout.write == 120.0 > 5.0
    assert request._media_write_timeout == 300.0 > 20.0


def test_get_updates_read_timeout_exceeds_poll_timeout():
    """Otherwise every idle long poll looks like a dead connection."""
    poll_timeout = 20.0
    request = build_get_updates_request(poll_timeout)
    assert request._client_kwargs["timeout"].read > poll_timeout


# --------------------------------------------------------------------------
# call_with_retry
# --------------------------------------------------------------------------


def test_returns_immediately_on_success():
    calls = []

    async def scenario():
        async def ok():
            calls.append(1)
            return "value"

        return await call_with_retry(ok, description="test", base_delay=0.01)

    assert asyncio.run(scenario()) == "value"
    assert len(calls) == 1


def test_retries_timed_out_then_succeeds():
    """A transient timeout must not surface to the user."""
    attempts = []

    async def scenario():
        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise TimedOut("connect timed out")
            return "recovered"

        return await call_with_retry(
            flaky, attempts=4, base_delay=0.01, description="test"
        )

    assert asyncio.run(scenario()) == "recovered"
    assert len(attempts) == 3


def test_retries_network_error():
    attempts = []

    async def scenario():
        async def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise NetworkError("connection reset")
            return "ok"

        return await call_with_retry(
            flaky, attempts=3, base_delay=0.01, description="test"
        )

    assert asyncio.run(scenario()) == "ok"
    assert len(attempts) == 2


def test_gives_up_after_all_attempts():
    attempts = []

    async def scenario():
        async def always_fail():
            attempts.append(1)
            raise TimedOut("never works")

        with pytest.raises(TimedOut):
            await call_with_retry(
                always_fail, attempts=3, base_delay=0.01, description="test"
            )

    asyncio.run(scenario())
    assert len(attempts) == 3


def test_non_retryable_error_propagates_immediately():
    """A 400-class error will never succeed, so do not waste retries."""
    attempts = []

    async def scenario():
        async def bad_request():
            attempts.append(1)
            raise BadRequest("chat not found")

        with pytest.raises(BadRequest):
            await call_with_retry(
                bad_request, attempts=4, base_delay=0.01, description="test"
            )

    asyncio.run(scenario())
    assert len(attempts) == 1


def test_retry_after_is_honoured():
    """Telegram's explicit wait must be respected, not ignored."""
    attempts = []

    async def scenario():
        async def rate_limited():
            attempts.append(1)
            if len(attempts) == 1:
                raise RetryAfter(0)  # 0 + 1.0s internal allowance
            return "ok"

        return await call_with_retry(
            rate_limited, attempts=3, base_delay=0.01, description="test"
        )

    assert asyncio.run(scenario()) == "ok"
    assert len(attempts) == 2


def test_retry_after_exhausting_attempts_raises():
    async def scenario():
        async def always_limited():
            raise RetryAfter(0)

        with pytest.raises(RetryAfter):
            await call_with_retry(
                always_limited, attempts=2, base_delay=0.01, description="test"
            )

    asyncio.run(scenario())


def test_backoff_grows_and_is_capped():
    """Verify the delay sequence stays bounded by max_delay."""
    import net

    delays = []
    real_sleep = asyncio.sleep

    async def fake_sleep(d):
        delays.append(d)
        await real_sleep(0)

    async def scenario():
        async def always_fail():
            raise TimedOut("x")

        original_sleep = net.asyncio.sleep
        net.asyncio.sleep = fake_sleep
        try:
            with pytest.raises(TimedOut):
                await call_with_retry(
                    always_fail,
                    attempts=5,
                    base_delay=1.0,
                    max_delay=4.0,
                    description="test",
                )
        finally:
            net.asyncio.sleep = original_sleep

    asyncio.run(scenario())
    # 4 sleeps for 5 attempts, all within [0, max_delay]
    assert len(delays) == 4
    assert all(0 <= d <= 4.0 for d in delays)


# --------------------------------------------------------------------------
# Upload serialisation
# --------------------------------------------------------------------------


def test_upload_lock_serialises_transfers():
    """Concurrent uploads were measured to fail far more often, so the bot
    guards send_document with a lock. Verify it actually serialises."""
    import bot

    events = []

    async def scenario():
        async def fake_upload(name, hold):
            async with bot._upload_lock:
                events.append(f"{name}-start")
                await asyncio.sleep(hold)
                events.append(f"{name}-end")

        # Second task is much faster; without a lock it would finish first.
        await asyncio.gather(fake_upload("a", 0.05), fake_upload("b", 0.001))

    asyncio.run(scenario())
    assert events == ["a-start", "a-end", "b-start", "b-end"]


def test_badrequest_is_not_retried_despite_subclassing_networkerror():
    """Regression guard for a real trap.

    PTB defines ``BadRequest(NetworkError)``, so ``except NetworkError`` alone
    silently retries permanent errors. Verify our predicate rejects them.
    """
    from net import is_retryable

    assert issubclass(BadRequest, NetworkError), "assumption of the trap"
    assert is_retryable(BadRequest("chat not found")) is False
    assert is_retryable(Forbidden("blocked")) is False
    assert is_retryable(InvalidToken("bad token")) is False
    # Genuine transient failures stay retryable.
    assert is_retryable(TimedOut("connect timed out")) is True
    assert is_retryable(NetworkError("connection reset")) is True


def test_forbidden_propagates_without_retry():
    from telegram.error import Forbidden

    attempts = []

    async def scenario():
        async def blocked():
            attempts.append(1)
            raise Forbidden("bot was blocked by the user")

        with pytest.raises(Forbidden):
            await call_with_retry(
                blocked, attempts=4, base_delay=0.01, description="test"
            )

    asyncio.run(scenario())
    assert len(attempts) == 1


# --------------------------------------------------------------------------
# Proxy support
# --------------------------------------------------------------------------


def test_proxy_from_env_precedence(monkeypatch):
    from net import proxy_from_env

    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    assert proxy_from_env() is None

    monkeypatch.setenv("HTTPS_PROXY", "http://fallback:8080")
    assert proxy_from_env() == "http://fallback:8080"

    # TELEGRAM_PROXY wins over the conventional variables.
    monkeypatch.setenv("TELEGRAM_PROXY", "http://explicit:7890")
    assert proxy_from_env() == "http://explicit:7890"


def test_empty_proxy_env_is_treated_as_unset(monkeypatch):
    """`TELEGRAM_PROXY=` must not become a broken proxy URL."""
    from net import proxy_from_env

    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TELEGRAM_PROXY", "   ")
    assert proxy_from_env() is None


def test_http_proxy_is_applied_to_request(monkeypatch):
    monkeypatch.setenv("TELEGRAM_PROXY", "http://127.0.0.1:7890")
    for var in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    request = build_request()
    assert request._client_kwargs["proxy"] == "http://127.0.0.1:7890"


def test_get_updates_request_also_uses_proxy(monkeypatch):
    monkeypatch.setenv("TELEGRAM_PROXY", "http://127.0.0.1:7890")
    for var in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    request = build_get_updates_request(20.0)
    assert request._client_kwargs["proxy"] == "http://127.0.0.1:7890"


def test_no_proxy_configured_means_direct(monkeypatch):
    for var in ("TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    assert build_request()._client_kwargs["proxy"] is None


def test_proxy_credentials_are_redacted_in_logs():
    from net import _redact

    assert _redact("http://user:secret@proxy:8080") == "http://***@proxy:8080"
    assert _redact("http://proxy:8080") == "http://proxy:8080"
    assert _redact("socks5://127.0.0.1:1080") == "socks5://127.0.0.1:1080"
