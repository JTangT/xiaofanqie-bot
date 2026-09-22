#!/usr/bin/env python3
"""Container health check: verify the bot can actually reach Telegram.

A plain "is the process alive" check would not have caught the failures this
project hit in practice -- the bot stays running while every request times out
on TLS handshake. So this performs a real, cheap ``getMe`` call.

Exits 0 when the token is valid and the API is reachable; non-zero otherwise.

Note: Docker does not restart a container merely because it is unhealthy
(that requires Swarm or an autoheal sidecar), so this is a diagnostic signal
for ``docker ps`` / monitoring rather than a restart trigger.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 8.0


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("unhealthy: TELEGRAM_BOT_TOKEN is not set", file=sys.stderr)
        return 1

    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # A 401 here means the network is fine but the token is wrong, which is
        # worth distinguishing from a connectivity failure.
        print(f"unhealthy: Telegram returned HTTP {exc.code} (check the token)", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - any failure means unhealthy
        print(f"unhealthy: cannot reach Telegram API: {exc}", file=sys.stderr)
        return 1

    if not payload.get("ok"):
        print(f"unhealthy: unexpected API response: {payload}", file=sys.stderr)
        return 1

    username = payload.get("result", {}).get("username", "?")
    print(f"healthy: authenticated as @{username}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
