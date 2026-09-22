# syntax=docker/dockerfile:1

# 小番茄图片解混淆 Telegram Bot
#
# Build:
#   docker build -t xiaofanqie-bot .
#
# Run (the token is passed at runtime and never baked into an image layer):
#   docker run -d --name xiaofanqie-bot \
#     -e TELEGRAM_BOT_TOKEN="123456:ABC..." \
#     --restart unless-stopped \
#     xiaofanqie-bot
#
# Or with compose (reads the token from a .env file):
#   docker compose up -d --build
#
# Behind a proxy on a slow link (see README, "网络超时"):
#   docker run -d -e TELEGRAM_BOT_TOKEN=... -e TELEGRAM_PROXY=http://host:7890 ...

FROM python:3.13-slim-bookworm

# PYTHONDONTWRITEBYTECODE avoids stray __pycache__ files in the container;
# PYTHONUNBUFFERED makes logs show up immediately in `docker logs`.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies are installed before the source is copied, so editing bot.py
# does not invalidate this (slow) layer. numpy and Pillow ship manylinux
# wheels, so no compiler or build toolchain is needed.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# Only the application code is copied. .dockerignore keeps .venv, .git and
# caches out of the build context, so nothing extra can leak in here.
COPY image_confuse.py net.py bot.py cli.py healthcheck.py ./

# Two steps that both matter:
#
# 1. chmod 0644: COPY preserves the source file mode, so a checkout with a
#    restrictive umask (files as 0600) would leave the non-root user unable to
#    read its own source, failing with
#    "PermissionError: '/app/image_confuse.py'". Normalising the modes makes
#    the image work regardless of the local umask.
# 2. An unprivileged user: the bot needs only the network and memory, never
#    root and never a writable application directory.
RUN chmod 0755 /app \
    && chmod 0644 /app/*.py \
    && useradd --create-home --shell /usr/sbin/nologin --uid 10001 botuser
USER botuser

# Performs a real getMe call instead of only checking the process is alive:
# on a slow link the bot stays up while every request times out, which is
# exactly the failure mode this project hit. `docker ps` then shows "unhealthy".
HEALTHCHECK --interval=60s --timeout=15s --start-period=20s --retries=3 \
    CMD ["python", "healthcheck.py"]

# Long-polling mode: no ports to publish and no volume required.
CMD ["python", "bot.py"]
