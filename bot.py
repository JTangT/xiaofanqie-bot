"""Telegram bot: receive a confused image, un-confuse it, send it back.

Usage:
    export TELEGRAM_BOT_TOKEN="123456:ABC..."
    python bot.py

Send the image as a **file/document** (Telegram's "Send as file"), not as a
photo. Telegram re-encodes and downscales photos, which destroys the
pixel-level permutation and makes the image impossible to restore.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from image_confuse import (
    AlgorithmError,
    MAX_PIXELS,
    decrypt_image,
    encrypt_image,
)
from net import build_get_updates_request, build_request, call_with_retry

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("xiaofanqie-bot")

# Telegram Bot API limits.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # getFile hard limit
MAX_SEND_BYTES = 50 * 1024 * 1024  # sendDocument limit

# Long-poll timeout handed to getUpdates. Kept modest so a dead connection is
# noticed quickly on a flaky link rather than hanging for a minute.
POLL_TIMEOUT = 20.0

# Decrypting is CPU-bound and holds the GIL for the numpy work plus the pure
# Python curve generation, so run it off the event loop in a small pool.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="confuse")

# Uploading a multi-MB file on a high-latency link opens a fresh TLS
# connection and holds it for a while. Running several at once was measured to
# drop the success rate from ~100% to ~15%, so uploads are serialised here.
_upload_lock = asyncio.Lock()

PHOTO_HINT = (
    "⚠️ *请用「文件」方式发送图片，不要用「照片」。*\n\n"
    "Telegram 会对「照片」自动压缩并缩放尺寸，像素位置被破坏，"
    "解混淆后只会得到乱码。\n\n"
    "正确做法：\n"
    "1\\. 点击输入框的 📎 回形针\n"
    "2\\. 选择「文件 / File」而不是「相册 / Gallery」\n"
    "3\\. 选中图片发送\n\n"
    "在 Telegram Desktop 上也可以直接把图片文件拖进聊天窗口。"
)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _looks_like_image(mime: str | None) -> bool:
    """Whether a document's MIME type could plausibly be an image.

    Telegram frequently labels uploaded documents as
    ``application/octet-stream`` or omits the MIME type entirely, so those are
    treated as "maybe" and handed to the decoder rather than rejected.
    """
    if not mime:
        return True
    mime = mime.lower().strip()
    if mime in ("application/octet-stream", "binary/octet-stream"):
        return True
    return mime.startswith("image/")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🍅 *小番茄图片解混淆 Bot*\n\n"
        "把混淆过的图片以「文件」形式发给我，我会还原后把 PNG 文件发回。\n\n"
        "命令：\n"
        "/start — 显示这条帮助\n"
        "/id — 显示你的 chat id\n"
        "/encrypt — 切换：默认解密；回复图片时改为加密\n\n"
        "⚠️ 必须用「文件」发送，不能用「照片」，否则 Telegram 的压缩会破坏像素。"
        "图片宽度或高度需大于 1。",
        parse_mode="Markdown",
    )


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    await update.message.reply_text(f"chat id: `{chat.id}`", parse_mode="Markdown")


async def cmd_encrypt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle per-chat mode: decrypt (default) or encrypt."""
    chat_id = update.effective_chat.id
    modes = context.application.bot_data.setdefault("modes", {})
    new_mode = "encrypt" if modes.get(chat_id) != "encrypt" else "decrypt"
    modes[chat_id] = new_mode
    label = "加密" if new_mode == "encrypt" else "解密"
    await update.message.reply_text(f"当前模式：*{label}*", parse_mode="Markdown")


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Photos cannot be reliably restored — tell the user to send a file."""
    await update.message.reply_text(PHOTO_HINT, parse_mode="Markdown")


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    document = message.document
    chat_id = message.chat_id

    if document.file_size and document.file_size > MAX_DOWNLOAD_BYTES:
        await message.reply_text(
            f"文件太大（{_human_size(document.file_size)}），"
            f"Bot API 下载上限为 {_human_size(MAX_DOWNLOAD_BYTES)}。"
        )
        return

    # Only reject types that are *clearly* not images. Telegram frequently
    # labels uploaded documents as application/octet-stream (or omits the MIME
    # type entirely), so anything ambiguous is passed through and left to the
    # decoder -- otherwise legitimate images get rejected here.
    if not _looks_like_image(document.mime_type):
        await message.reply_text(
            f"这不是图片文件（{document.mime_type}），请发送 PNG / JPG / WEBP 等图片。"
        )
        return

    mode = context.application.bot_data.get("modes", {}).get(chat_id, "decrypt")
    action = "加密" if mode == "encrypt" else "解密"

    status = await message.reply_text(f"⏳ 正在{action}，请稍候…")
    await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)

    started = time.monotonic()

    async def _download() -> bytes:
        tg_file = await document.get_file()
        return bytes(await tg_file.download_as_bytearray())

    try:
        raw = await call_with_retry(_download, description="download", attempts=3)
    except TelegramError as exc:
        logger.warning("download failed: %s", exc)
        await status.edit_text(f"❌ 下载文件失败：{exc}")
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("download failed unexpectedly")
        await status.edit_text(f"❌ 下载文件失败：{exc}")
        return

    try:
        result_bytes, out_name, width, height = await asyncio.get_running_loop().run_in_executor(
            _executor, _process, raw, mode, document.file_name
        )
    except AlgorithmError as exc:
        await status.edit_text(f"❌ 处理失败：{exc}")
        return
    except Exception:  # noqa: BLE001 - report anything unexpected to the user
        logger.exception("processing failed")
        await status.edit_text("❌ 处理失败：文件可能不是有效图片，或已损坏。")
        return

    elapsed = time.monotonic() - started

    if len(result_bytes) > MAX_SEND_BYTES:
        await status.edit_text(
            f"⚠️ 结果文件 {_human_size(len(result_bytes))} 超过 Telegram "
            f"{_human_size(MAX_SEND_BYTES)} 上限，无法发送。"
        )
        return

    async def _upload():
        # Serialise uploads: concurrent large transfers were measured to fail
        # far more often than serialised ones on a high-latency link.
        async with _upload_lock:
            return await context.bot.send_document(
                chat_id=chat_id,
                document=io.BytesIO(result_bytes),
                filename=out_name,
                caption=(
                    f"✅ {action}完成 · {width}×{height} · "
                    f"{_human_size(document.file_size or len(raw))} → "
                    f"{_human_size(len(result_bytes))} · {elapsed:.1f}s"
                ),
                reply_to_message_id=message.message_id,
            )

    try:
        await call_with_retry(_upload, description="upload", attempts=4)
    except TelegramError as exc:
        logger.warning("upload failed: %s", exc)
        await status.edit_text(
            f"❌ 发送结果失败（网络超时）：{exc}\n"
            "图片已处理完成，请重新发送一次试试。"
        )
        return

    try:
        await status.delete()
    except BadRequest:
        pass


def _process(raw: bytes, mode: str, file_name: str | None):
    """Decode, transform, and re-encode. Runs in a worker thread."""
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(raw))
        # Read the header and validate size *before* decoding pixels, so a
        # crafted "decompression bomb" cannot expand into memory first.
        width, height = image.size
    except UnidentifiedImageError as exc:
        raise AlgorithmError("无法识别的图片格式") from exc
    except OSError as exc:
        raise AlgorithmError(f"无法读取图片：{exc}") from exc

    if width < 2 or height < 2:
        raise AlgorithmError(f"图片尺寸过小：{width}×{height}")
    if width * height > MAX_PIXELS:
        raise AlgorithmError(
            f"图片太大：{width}×{height} = {width * height:,} 像素 "
            f"（上限 {MAX_PIXELS:,}）"
        )

    try:
        image.load()
    except OSError as exc:
        # Truncated or corrupt pixel data.
        raise AlgorithmError(f"图片数据损坏或不完整：{exc}") from exc

    result = encrypt_image(image) if mode == "encrypt" else decrypt_image(image)

    buffer = io.BytesIO()
    # PNG keeps the restored pixels exact; JPEG would add another generation
    # of lossy compression on top of the input's.
    result.save(buffer, format="PNG", optimize=True)
    data = buffer.getvalue()

    stem = os.path.splitext(os.path.basename(file_name or "image"))[0] or "image"
    suffix = "_encrypted" if mode == "encrypt" else "_decrypted"
    return data, f"{stem}{suffix}.png", width, height


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("update %s caused error", update, exc_info=context.error)


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print(
            "ERROR: TELEGRAM_BOT_TOKEN is not set.\n"
            "Get a token from @BotFather, then:\n"
            '  export TELEGRAM_BOT_TOKEN="123456:ABC..."\n'
            "  python bot.py",
            file=sys.stderr,
        )
        return 1

    application: Application = (
        ApplicationBuilder()
        .token(token)
        # Bounded concurrency: this is a CPU-bound image tool, and letting many
        # updates run at once both saturates the link and thrashes the disk.
        .concurrent_updates(4)
        # Explicit, latency-tolerant transports instead of PTB's defaults
        # (pool=1, connect/read/write=5s, pool_timeout=1s). See net.py.
        .request(build_request())
        .get_updates_request(build_get_updates_request(POLL_TIMEOUT))
        .build()
    )
    application.bot_data["modes"] = {}

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_start))
    application.add_handler(CommandHandler("id", cmd_id))
    application.add_handler(CommandHandler("encrypt", cmd_encrypt))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    # Telegram often reports real images as application/octet-stream, so accept
    # any document and let the decoder decide (see on_document).
    application.add_handler(MessageHandler(filters.Document.ALL, on_document))
    application.add_error_handler(on_error)

    logger.info("Bot started, polling for updates…")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        poll_interval=1.0,
        timeout=POLL_TIMEOUT,
        # Keep retrying the initial getMe on a flaky link instead of aborting.
        bootstrap_retries=5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
