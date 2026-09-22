"""Tests for the Python port of the 小番茄图片混淆 algorithm.

Includes a differential test that runs the *original* JavaScript (extracted
from the HTML tool) under Node and asserts the Python permutation is identical.
That test is skipped automatically when node or the reference HTML is missing.

Run with:  .venv/bin/python -m pytest -v
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_confuse import (
    AlgorithmError,
    decrypt_array,
    decrypt_image,
    encrypt_array,
    encrypt_image,
    gilbert2d,
    gilbert_curve_indices,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REFERENCE_NAME = "小番茄图片混淆工具.html"


def _find_reference_html() -> Path | None:
    """Locate the original HTML tool for the differential tests.

    The reference file is NOT bundled with this repository (the upstream
    project ships no licence, so redistributing it is avoided). It is looked
    for in a few sensible places instead, and the differential tests skip
    cleanly when it cannot be found.
    """
    candidates = []
    env = os.environ.get("REFERENCE_HTML", "").strip()
    if env:
        candidates.append(Path(env))
    # A sibling checkout of the upstream repo, and a local reference/ dir.
    candidates.append(_REPO_ROOT.parent / "xiaofanqie-image-confuse" / _REFERENCE_NAME)
    candidates.append(_REPO_ROOT / "reference" / _REFERENCE_NAME)
    candidates.append(_REPO_ROOT / "reference" / "xiaofanqie-image-confuse.html")
    candidates.append(_REPO_ROOT / _REFERENCE_NAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


REFERENCE_HTML = _find_reference_html()
_NEEDS_REFERENCE = pytest.mark.skipif(
    REFERENCE_HTML is None,
    reason=(
        "reference HTML not found; clone MadLife77/xiaofanqie-image-confuse "
        "next to this repo or set REFERENCE_HTML=<path>"
    ),
)
_NEEDS_NODE = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

SIZES = [(1, 1), (2, 1), (1, 2), (7, 5), (16, 16), (37, 23), (3, 9), (64, 64), (100, 100)]


# --------------------------------------------------------------------------
# Curve properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width,height", SIZES)
def test_curve_is_a_permutation(width, height):
    """The curve must visit every pixel exactly once."""
    curve = gilbert_curve_indices(width, height)
    assert len(curve) == width * height
    assert len(set(curve.tolist())) == width * height
    assert curve.min() == 0
    assert curve.max() == width * height - 1


@pytest.mark.parametrize("width,height", [(7, 5), (37, 23), (100, 100), (13, 40)])
def test_curve_is_continuous(width, height):
    """Consecutive curve points must be 4-neighbours (the curve must not jump)."""
    flat = gilbert_curve_indices(width, height).tolist()
    coords = [(i % width, i // width) for i in flat]
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        assert abs(x1 - x2) + abs(y1 - y2) == 1, (
            f"curve jumps between {(x1, y1)} and {(x2, y2)}"
        )


def test_reference_order_matches_known_js_output():
    """Spot-check early curve points against the original JS tool's output.

    Expected values below were captured by running the reference
    ``gilbert2d`` from the HTML under Node.
    """
    # 4x2 -> (0,0) (0,1) (1,1) (1,0) (2,0)
    assert gilbert_curve_indices(4, 2).tolist()[:5] == [0, 4, 5, 1, 2]
    # 2x4 -> (0,0) (1,0) (1,1) (0,1) (0,2)
    assert gilbert_curve_indices(2, 4).tolist()[:5] == [0, 1, 3, 2, 4]
    # 5x3 -> (0,0) (0,1) (0,2) (1,2) (1,1)
    assert gilbert_curve_indices(5, 3).tolist()[:5] == [0, 5, 10, 11, 6]


# --------------------------------------------------------------------------
# Round-trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width,height", SIZES)
def test_array_roundtrip(width, height):
    rng = np.random.default_rng(1234)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    restored = decrypt_array(encrypt_array(arr))
    assert np.array_equal(restored, arr)


@pytest.mark.parametrize("width,height", SIZES)
def test_encrypt_actually_changes_image(width, height):
    """Encryption must not be a no-op (except for a 1x1 image)."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    encrypted = encrypt_array(arr)
    if width * height == 1:
        assert np.array_equal(encrypted, arr)
    else:
        assert not np.array_equal(encrypted, arr)


def test_double_encrypt_and_double_decrypt():
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, (37, 23, 3), dtype=np.uint8)
    assert np.array_equal(decrypt_array(decrypt_array(encrypt_array(encrypt_array(arr)))), arr)


def test_grayscale_and_rgba_arrays():
    rng = np.random.default_rng(5)
    gray = rng.integers(0, 256, (16, 24), dtype=np.uint8)
    assert np.array_equal(decrypt_array(encrypt_array(gray)), gray)

    rgba = rng.integers(0, 256, (16, 24, 4), dtype=np.uint8)
    assert np.array_equal(decrypt_array(encrypt_array(rgba)), rgba)


def test_invalid_shapes_rejected():
    with pytest.raises(AlgorithmError):
        encrypt_array(np.zeros((4, 4, 3, 2), dtype=np.uint8))
    with pytest.raises(AlgorithmError):
        gilbert2d(0, 10)
    with pytest.raises(AlgorithmError):
        gilbert2d(10, -1)


# --------------------------------------------------------------------------
# PIL integration
# --------------------------------------------------------------------------


def test_image_roundtrip_lossless(tmp_path):
    rng = np.random.default_rng(2024)
    arr = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    original = Image.fromarray(arr)

    encrypted = encrypt_image(original)
    decrypted = decrypt_image(encrypted)

    assert np.array_equal(np.array(decrypted), arr)


def test_mode_preserved_rgba():
    arr = np.zeros((20, 30, 4), dtype=np.uint8)
    arr[..., 3] = 128
    arr[5, 7] = [10, 20, 30, 200]
    original = Image.fromarray(arr, mode="RGBA")
    decrypted = decrypt_image(encrypt_image(original))
    assert decrypted.mode == "RGBA"
    assert np.array_equal(np.array(decrypted), arr)


def test_exif_orientation_is_applied():
    """A 90-degree EXIF rotation must be normalised the way a browser would."""
    arr = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    image = Image.fromarray(arr)
    exif = image.getexif()
    exif[274] = 6  # Orientation: rotate 90 CW
    buffer = __import__("io").BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)

    with Image.open(buffer) as reloaded:
        out = decrypt_image(encrypt_image(reloaded))
    # Orientation 6 means the stored image is rotated relative to display, so
    # the processed result must come back transposed relative to the raw pixels.
    assert out.size == (3, 4)


def test_png_roundtrip_through_disk_preserves_every_pixel(tmp_path):
    """The bot ships PNG precisely so the restored pixels survive intact."""
    rng = np.random.default_rng(31337)
    arr = rng.integers(0, 256, (61, 83, 3), dtype=np.uint8)

    encrypted = encrypt_image(Image.fromarray(arr))
    enc_path = tmp_path / "enc.png"
    encrypted.save(enc_path)

    with Image.open(enc_path) as loaded:
        decrypted = decrypt_image(loaded)

    assert np.array_equal(np.array(decrypted), arr)


# --------------------------------------------------------------------------
# Differential test against the original JavaScript
# --------------------------------------------------------------------------


def _extract_js() -> str:
    html = REFERENCE_HTML.read_text(encoding="utf-8")
    start = html.index("function gilbert2d")
    end = html.index("// 加密函数")
    return html[start:end]


@_NEEDS_NODE
@_NEEDS_REFERENCE
@pytest.mark.parametrize("width,height", SIZES)
def test_matches_original_javascript(width, height):
    """Python permutation must be byte-identical to the reference JS tool."""
    script = textwrap.dedent(
        f"""
        {_extract_js()}
        const width = {width}, height = {height};
        const curve = gilbert2d(width, height);
        const n = width * height;
        const offset = Math.round((Math.sqrt(5) - 1) / 2 * n);
        const perm = [];
        for (let i = 0; i < n; i++) {{
            const op = curve[i], np_ = curve[(i + offset) % n];
            perm.push([op[0] + op[1] * width, np_[0] + np_[1] * width]);
        }}
        process.stdout.write(JSON.stringify({{offset, perm}}));
        """
    )
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    js = json.loads(result.stdout)

    curve = gilbert_curve_indices(width, height)
    assert js["offset"] == round((math.sqrt(5) - 1) / 2 * width * height)

    js_src = [int(p[0]) for p in js["perm"]]
    js_dst = [int(p[1]) for p in js["perm"]]
    assert js_src == curve.tolist()
    assert js_dst == np.roll(curve, -js["offset"]).tolist()


@_NEEDS_NODE
@_NEEDS_REFERENCE
def test_matches_original_javascript_many_sizes():
    """Broad differential sweep: 40 varied sizes, curves and offsets must match.

    This is the test that catches the ``Math.floor`` / truncating-division trap,
    which only shows up on non-square images with negative deltas.
    """
    sizes = [(w, h) for w in range(1, 9) for h in range(1, 9)]
    sizes += [(37, 23), (23, 37), (100, 100), (101, 57), (57, 101), (200, 3), (3, 200)]

    script = textwrap.dedent(
        f"""
        {_extract_js()}
        const sizes = {json.dumps(sizes)};
        const out = {{}};
        for (const [w, h] of sizes) {{
            const curve = gilbert2d(w, h);
            const n = w * h;
            out[`${{w}}x${{h}}`] = {{
                offset: Math.round((Math.sqrt(5) - 1) / 2 * n),
                src: curve.map(p => p[0] + p[1] * w),
            }};
        }}
        process.stdout.write(JSON.stringify(out));
        """
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=300, check=True
    )
    js = json.loads(result.stdout)

    for (width, height) in sizes:
        ref = js[f"{width}x{height}"]
        curve = gilbert_curve_indices(width, height)
        assert ref["offset"] == round((math.sqrt(5) - 1) / 2 * width * height), (
            f"offset mismatch for {width}x{height}"
        )
        assert ref["src"] == curve.tolist(), f"curve mismatch for {width}x{height}"


@_NEEDS_NODE
@_NEEDS_REFERENCE
@pytest.mark.parametrize("width,height", [(7, 5), (16, 16), (37, 23), (3, 9), (64, 48)])
def test_full_pixel_loops_match_javascript(width, height):
    """Strongest check: run the reference encrypt/decrypt loops on real pixels.

    Reimplements the reference ``encrypt``/``decrypt`` inner loops verbatim
    (including the ``4 * (x + y * width)`` RGBA offset arithmetic) over Node,
    then asserts Python produces identical output bytes.
    """
    script = textwrap.dedent(
        f"""
        {_extract_js()}
        const width = {width}, height = {height};
        const n = width * height;
        const data = new Uint8Array(n * 4);
        for (let i = 0; i < n; i++) {{
            data[4 * i + 0] = i % 256;
            data[4 * i + 1] = (i * 7) % 256;
            data[4 * i + 2] = (i * 13) % 256;
            data[4 * i + 3] = 255;
        }}
        function conv(src, inverse) {{
            const curve = gilbert2d(width, height);
            const offset = Math.round((Math.sqrt(5) - 1) / 2 * n);
            const out = new Uint8Array(src.length);
            for (let i = 0; i < n; i++) {{
                const old_pos = curve[i];
                const new_pos = curve[(i + offset) % n];
                const old_p = 4 * (old_pos[0] + old_pos[1] * width);
                const new_p = 4 * (new_pos[0] + new_pos[1] * width);
                if (inverse) out.set(src.slice(new_p, new_p + 4), old_p);
                else out.set(src.slice(old_p, old_p + 4), new_p);
            }}
            return out;
        }}
        const enc = conv(data, false);
        process.stdout.write(JSON.stringify({{
            enc: Array.from(enc),
            dec: Array.from(conv(enc, true)),
            orig: Array.from(data),
        }}));
        """
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=300, check=True
    )
    js = json.loads(result.stdout)

    source = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(width * height):
        source.reshape(-1, 4)[i] = [i % 256, (i * 7) % 256, (i * 13) % 256, 255]

    js_enc = np.array(js["enc"], dtype=np.uint8).reshape(height, width, 4)
    js_dec = np.array(js["dec"], dtype=np.uint8).reshape(height, width, 4)

    # Python must reproduce the reference transform from the same input...
    assert np.array_equal(encrypt_array(source), js_enc), f"encrypt differs for {width}x{height}"
    # ...and must be able to invert the reference's *own* encrypted output.
    assert np.array_equal(decrypt_array(js_enc), js_dec), f"decrypt differs for {width}x{height}"
    # The reference round-trip is lossless, and so is ours.
    assert np.array_equal(js_dec, source)
    assert np.array_equal(decrypt_array(encrypt_array(source)), source)
