"""Network resilience helpers for the Telegram bot.

Why this module exists
----------------------
The bot originally ran with python-telegram-bot's defaults, which are tuned for
a well-connected host and are a poor fit for a high-latency link:

    connection_pool_size = 1     <- all requests share ONE TCP connection
    connect_timeout      = 5.0   <- a fresh TLS handshake may need longer
    read_timeout         = 5.0
    write_timeout        = 5.0   <- far too small for a multi-MB upload
    pool_timeout         = 1.0   <- wait only 1s for the busy connection

Measured on the host this was written for (RTT to api.telegram.org ~300 ms,
TLS handshake ~0.5-6 s, intermittent drops):

* A brand-new TLS handshake succeeded 20/20 when allowed 25 s, but 1 of those
  20 needed 5.9 s -- i.e. it would be killed by the default 5 s connect
  timeout. The server was healthy; only the client's patience was the problem.
* Opening connections **concurrently** is what really breaks: with 8s timeout,
  1 connection -> 8/8 OK, 4 -> 16/16 OK, 8 -> 14/24, 40 -> 4/40.
  So bursts of new TLS handshakes get dropped, while serialised ones are fine.
* **Reusing** a warm connection is both reliable and ~10x faster:
  25/25 OK with a median of 0.26 s, versus 0.73-2.0 s for fresh handshakes.

Hence the three mitigations below, in order of importance:

1. Keep connections alive and pooled (``connection_pool_size``), and make the
   pool wait rather than fail instantly (``pool_timeout``).
2. Serialise time-consuming media uploads so they never stampede the link.
3. Retry the transient failures that still slip through, with backoff.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Awaitable, Callable, TypeVar

from telegram.error import (
    BadRequest,
    Forbidden,
    InvalidToken,
    NetworkError,
    RetryAfter,
    TimedOut,
)
from telegram.request import HTTPXRequest

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Purely transport-level failures, safe to retry: Telegram either never
# received the request, or deduplicates by message id.
RETRYABLE = (TimedOut, NetworkError)

# Permanently-failing errors that must NOT be retried. This list matters more
# than it looks: in python-telegram-bot, ``BadRequest`` inherits from
# ``NetworkError`` (a legacy quirk), so a naive ``except NetworkError`` would
# retry hopeless errors such as "chat not found" or "file is too big",
# delaying the real error message to the user by the whole backoff budget.
NON_RETRYABLE = (BadRequest, Forbidden, InvalidToken)


def is_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` is worth another attempt."""
    if isinstance(exc, NON_RETRYABLE):
        return False
    return isinstance(exc, RETRYABLE)

DEFAULT_POOL_SIZE = 8
DEFAULT_CONNECT_TIMEOUT = 20.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_WRITE_TIMEOUT = 120.0
DEFAULT_POOL_TIMEOUT = 30.0
DEFAULT_MEDIA_WRITE_TIMEOUT = 300.0


def proxy_from_env() -> str | None:
    """Read a proxy URL from the environment.

    Honours ``TELEGRAM_PROXY`` first (explicit and unambiguous), then the
    conventional variables, and ignores empty values so ``TELEGRAM_PROXY=``
    behaves as "unset" rather than as a broken proxy.
    """
    for name in (
        "TELEGRAM_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _redact(url: str) -> str:
    """Hide credentials embedded in a proxy URL before logging it."""
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


def build_request(
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    write_timeout: float = DEFAULT_WRITE_TIMEOUT,
    pool_timeout: float = DEFAULT_POOL_TIMEOUT,
    media_write_timeout: float = DEFAULT_MEDIA_WRITE_TIMEOUT,
    proxy_url: str | None = None,
) -> HTTPXRequest:
    """Build an :class:`HTTPXRequest` tolerant of a slow, lossy link.

    When ``proxy_url`` is not given, a proxy is taken from the environment if
    one is configured (see :func:`proxy_from_env`).
    """
    proxy = proxy_url if proxy_url is not None else proxy_from_env()
    if proxy:
        logger.info("routing Telegram traffic through proxy %s", _redact(proxy))
    return HTTPXRequest(
        connection_pool_size=pool_size,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
        pool_timeout=pool_timeout,
        media_write_timeout=media_write_timeout,
        proxy=proxy,
    )


def build_get_updates_request(
    poll_timeout: float,
    *,
    pool_size: int = 2,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> HTTPXRequest:
    """Request object for long polling.

    ``read_timeout`` must exceed the long-poll timeout, otherwise every idle
    poll looks like a dead connection and the bot reconnects in a loop.
    """
    return HTTPXRequest(
        connection_pool_size=pool_size,
        connect_timeout=connect_timeout,
        # Generous margin over the server-side poll timeout.
        read_timeout=poll_timeout + 20.0,
        write_timeout=30.0,
        pool_timeout=30.0,
        proxy=proxy_from_env(),
    )


async def call_with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    base_delay: float = 1.5,
    max_delay: float = 20.0,
    description: str = "request",
) -> T:
    """Call ``func`` retrying transient network failures with jittered backoff.

    ``func`` must be a zero-argument callable returning a fresh coroutine, so
    that each attempt creates a new request object.
    """

    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except RetryAfter as exc:
            # Telegram explicitly told us how long to wait; honour it.
            delay = float(exc.retry_after) + 1.0
            logger.warning("%s rate-limited, sleeping %.1fs", description, delay)
            if attempt == attempts:
                raise
            await asyncio.sleep(delay)
        except Exception as exc:
            # Deliberately broad: ``BadRequest`` subclasses ``NetworkError`` in
            # PTB, so the retryable test has to be an explicit predicate rather
            # than a plain ``except NetworkError``.
            if not is_retryable(exc):
                raise
            if attempt == attempts:
                logger.error(
                    "%s failed after %d attempts: %s", description, attempts, exc
                )
                raise
            # Exponential backoff with full jitter, to avoid re-stampeding.
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = random.uniform(delay * 0.5, delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                description, attempt, attempts, exc, delay,
            )
            await asyncio.sleep(delay)

    raise AssertionError("unreachable: the loop either returns or raises")
