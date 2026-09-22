"""Tests for the bot's image-processing pipeline and the CLI.

These do not contact Telegram; they exercise the same ``_process`` function the
``on_document`` handler calls, plus argument handling for ``cli.py``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

import cli
from bot import _process
from image_confuse import AlgorithmError, encrypt_image


def _png_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def test_bot_decrypts_an_encrypted_document_exactly():
    """The full bot path: upload confused PNG -> get back the original pixels."""
    rng = np.random.default_rng(2024)
    original = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)

    confused = _png_bytes(np.array(encrypt_image(Image.fromarray(original))))

    data, name, width, height = _process(confused, "decrypt", "cat.png")

    assert name == "cat_decrypted.png"
    assert (width, height) == (320, 240)
    restored = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    assert np.array_equal(restored, original)


def test_bot_encrypt_mode_round_trips():
    rng = np.random.default_rng(77)
    original = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    plain = _png_bytes(original)

    encrypted, name, _, _ = _process(plain, "encrypt", "photo.jpg")
    assert name == "photo_encrypted.png"

    decrypted, _, _, _ = _process(encrypted, "decrypt", "photo_encrypted.png")
    assert np.array_equal(
        np.array(Image.open(io.BytesIO(decrypted)).convert("RGB")), original
    )


def test_bot_output_is_png_and_lossless():
    rng = np.random.default_rng(11)
    original = rng.integers(0, 256, (50, 70, 3), dtype=np.uint8)
    plain = _png_bytes(original)

    data, name, _, _ = _process(plain, "encrypt", "a.png")
    assert name.endswith(".png")
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_bot_rejects_non_image_bytes():
    with pytest.raises(AlgorithmError):
        _process(b"definitely not an image", "decrypt", "x.png")


def test_bot_rejects_degenerate_dimensions():
    """A 1xA or Ax1 strip cannot carry the permutation meaningfully."""
    image = Image.new("RGB", (1, 10))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with pytest.raises(AlgorithmError):
        _process(buffer.getvalue(), "decrypt", "strip.png")


def test_bot_handles_missing_filename():
    rng = np.random.default_rng(3)
    plain = _png_bytes(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    _, name, _, _ = _process(plain, "decrypt", None)
    assert name == "image_decrypted.png"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, (90, 150, 3), dtype=np.uint8)
    src = tmp_path / "src.png"
    Image.fromarray(original).save(src)

    enc = tmp_path / "enc.png"
    dec = tmp_path / "dec.png"
    assert cli.main(["encrypt", str(src), "-o", str(enc)]) == 0
    assert cli.main(["decrypt", str(enc), "-o", str(dec)]) == 0

    restored = np.array(Image.open(dec).convert("RGB"))
    assert np.array_equal(restored, original)


def test_cli_batch_into_outdir(tmp_path):
    rng = np.random.default_rng(1)
    srcs = []
    for i in range(3):
        path = tmp_path / f"img{i}.png"
        Image.fromarray(rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)).save(path)
        srcs.append(path)

    outdir = tmp_path / "out"
    assert cli.main(["decrypt", *map(str, srcs), "-d", str(outdir)]) == 0
    assert sorted(p.name for p in outdir.iterdir()) == [
        "img0_decrypted.png",
        "img1_decrypted.png",
        "img2_decrypted.png",
    ]


def test_cli_reports_failure_for_non_image(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("not an image")
    assert cli.main(["decrypt", str(bad)]) == 1


def test_cli_jpeg_flag_produces_jpeg(tmp_path):
    rng = np.random.default_rng(8)
    src = tmp_path / "src.png"
    Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(src)

    outdir = tmp_path / "out"
    assert cli.main(["encrypt", str(src), "-d", str(outdir), "--jpeg"]) == 0
    produced = list(outdir.iterdir())
    assert len(produced) == 1
    assert Image.open(produced[0]).format == "JPEG"


def test_cli_rejects_conflicting_output_flags():
    with pytest.raises(SystemExit):
        cli.main(["encrypt", "a.png", "-o", "b.png", "-d", "out"])


# --------------------------------------------------------------------------
# MIME handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mime,expected",
    [
        ("image/png", True),
        ("image/jpeg", True),
        ("image/webp", True),
        ("IMAGE/PNG", True),
        (None, True),  # Telegram omitted it: let the decoder try
        ("", True),
        ("application/octet-stream", True),  # very common for real uploads
        ("binary/octet-stream", True),
        ("application/pdf", False),
        ("text/plain", False),
        ("application/zip", False),
        ("video/mp4", False),
    ],
)
def test_looks_like_image(mime, expected):
    from bot import _looks_like_image

    assert _looks_like_image(mime) is expected


def test_human_size():
    from bot import _human_size

    assert _human_size(512) == "512 B"
    assert _human_size(2048) == "2.0 KB"
    assert _human_size(20 * 1024 * 1024) == "20.0 MB"


def test_bot_rejects_oversized_image_before_decoding():
    """A huge canvas must be refused from the header, not decoded first."""
    import image_confuse

    # A real 2x2 PNG, but we pretend its dimensions exceed the limit.
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")

    original_limit = image_confuse.MAX_PIXELS
    try:
        image_confuse.MAX_PIXELS = 2  # 2x2 = 4 pixels > 2
        import bot as bot_module

        bot_module.MAX_PIXELS = 2
        with pytest.raises(AlgorithmError, match="太大"):
            _process(buf.getvalue(), "decrypt", "big.png")
    finally:
        image_confuse.MAX_PIXELS = original_limit
        import bot as bot_module

        bot_module.MAX_PIXELS = original_limit


def test_bot_rejects_truncated_image_data():
    """Header parses fine but pixel data is cut off."""
    rng = np.random.default_rng(6)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
        buf, format="PNG"
    )
    truncated = buf.getvalue()[:60]
    with pytest.raises(AlgorithmError):
        _process(truncated, "decrypt", "trunc.png")
