"""小番茄图片混淆 — Python port of the Hilbert-curve image confusion algorithm.

Reference implementation: `小番茄图片混淆工具.html` (MadLife77/xiaofanqie-image-confuse),
functions `gilbert2d`, `generate2d`, `encrypt`, `decrypt`.

How it works
------------
The image is *not* encrypted in any cryptographic sense. A generalized Hilbert
curve visits every pixel of the (arbitrary, non-square) image exactly once,
producing a permutation of pixel positions. Pixels are then cyclically shifted
along that curve by a fixed offset:

    offset = round((sqrt(5) - 1) / 2 * width * height)   # ~0.618 * N, golden ratio

    encrypt:  dst[curve[(i + offset) % N]] = src[curve[i]]
    decrypt:  dst[curve[i]] = src[curve[(i + offset) % N]]

Because the offset is derived purely from width and height, there is no key:
anyone who knows the dimensions can invert it. This module reproduces the
reference behaviour exactly (verified byte-for-byte against the original
JavaScript, see tests/).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

__all__ = [
    "gilbert2d",
    "gilbert_curve_indices",
    "encrypt_array",
    "decrypt_array",
    "encrypt_image",
    "decrypt_image",
    "AlgorithmError",
    "MAX_PIXELS",
]

# Refuse absurd allocations (a Telegram document can be up to 20 MB of highly
# compressible data, which would otherwise expand into an enormous canvas).
# At this size an RGBA image plus its output and the index table need roughly
# 1 GB, which is about the practical ceiling for a small VPS.
MAX_PIXELS = 50_000_000

# Distinct (width, height) index tables kept in memory. Each costs
# 4 * width * height bytes.
_CACHE_SIZE = 8


class AlgorithmError(ValueError):
    """Raised when an image cannot be processed by the algorithm."""


def gilbert2d(width: int, height: int) -> list[tuple[int, int]]:
    """Generate the generalized Hilbert curve as a list of ``(x, y)`` points.

    Direct port of the reference ``gilbert2d`` / ``generate2d``. Visits every
    pixel of the ``width`` x ``height`` rectangle exactly once, and works for
    non-square and odd-sized images.
    """
    if width <= 0 or height <= 0:
        raise AlgorithmError(f"invalid dimensions: {width}x{height}")

    coordinates: list[tuple[int, int]] = []

    def generate(x: int, y: int, ax: int, ay: int, bx: int, by: int) -> None:
        w = abs(ax + ay)
        h = abs(bx + by)
        dax = (ax > 0) - (ax < 0)
        day = (ay > 0) - (ay < 0)
        dbx = (bx > 0) - (bx < 0)
        dby = (by > 0) - (by < 0)

        if h == 1:
            for _ in range(w):
                coordinates.append((x, y))
                x += dax
                y += day
            return

        if w == 1:
            for _ in range(h):
                coordinates.append((x, y))
                x += dbx
                y += dby
            return

        # NOTE: Python's // floors toward negative infinity, which is exactly
        # what JavaScript's Math.floor does. Do NOT use int(a / 2) here: that
        # truncates toward zero and silently corrupts the curve whenever a
        # delta is negative (which happens for any non-square image).
        ax2, ay2 = ax // 2, ay // 2
        bx2, by2 = bx // 2, by // 2
        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)

        if 2 * w > 3 * h:
            if (w2 % 2) and (w > 2):
                ax2 += dax
                ay2 += day
            generate(x, y, ax2, ay2, bx, by)
            generate(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        else:
            if (h2 % 2) and (h > 2):
                bx2 += dbx
                by2 += dby
            generate(x, y, bx2, by2, ax2, ay2)
            generate(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
            generate(
                x + (ax - dax) + (bx2 - dbx),
                y + (ay - day) + (by2 - dby),
                -bx2,
                -by2,
                -(ax - ax2),
                -(ay - ay2),
            )

    if width >= height:
        generate(0, 0, width, 0, 0, height)
    else:
        generate(0, 0, 0, height, width, 0)

    return coordinates


@lru_cache(maxsize=_CACHE_SIZE)
def gilbert_curve_indices(width: int, height: int) -> np.ndarray:
    """Hilbert curve as flat pixel indices ``x + y * width``.

    Cached, because generating the curve in pure Python is by far the most
    expensive part of the algorithm (~0.7 s for 1080p, ~5 s for 12 MP).

    The array is stored in the narrowest signed integer type that can hold the
    largest index, which halves the cache's memory footprint versus int64.
    """
    count = width * height
    dtype = np.int32 if count <= np.iinfo(np.int32).max else np.int64
    coordinates = gilbert2d(width, height)
    flat = np.fromiter(
        (x + y * width for x, y in coordinates), dtype=dtype, count=count
    )
    return flat


def _confusion_offset(pixel_count: int) -> int:
    """``round((sqrt(5) - 1) / 2 * N)``.

    ``round`` half-to-even (Python) matches ``Math.round`` half-up (JS) for
    the values produced here, but the result is checked against the reference
    implementation by the test-suite for a range of image sizes.

    Note the golden ratio here is (sqrt(5) - 1) / 2 ~= 0.618, i.e. 1/phi
    rather than the more common (1 + sqrt(5)) / 2 ~= 1.618.
    """
    return round((math.sqrt(5) - 1) / 2 * pixel_count)


def _apply(arr: np.ndarray, inverse: bool) -> np.ndarray:
    """Apply the pixel permutation to a ``(H, W)`` or ``(H, W, C)`` array."""
    if arr.ndim == 2:
        height, width = arr.shape
        channels = None
    elif arr.ndim == 3:
        height, width, channels = arr.shape
    else:
        raise AlgorithmError(f"expected a 2D or 3D array, got shape {arr.shape}")

    pixel_count = width * height
    if pixel_count > MAX_PIXELS:
        raise AlgorithmError(
            f"image too large: {width}x{height} = {pixel_count:,} pixels "
            f"(limit {MAX_PIXELS:,})"
        )

    curve = gilbert_curve_indices(width, height)
    offset = _confusion_offset(pixel_count)

    # Matches the reference loop exactly (verified against the original JS):
    #   encrypt: dst[curve[i]] = src[curve[(i + offset) % N]]
    #   decrypt: dst[curve[i]] = src[curve[(i - offset) % N]]
    shifted = np.roll(curve, -offset if inverse else offset)

    flat = arr.reshape(-1, channels) if channels is not None else arr.reshape(-1)
    out = np.empty_like(flat)
    out[curve] = flat[shifted]

    return out.reshape(arr.shape)


def encrypt_array(arr: np.ndarray) -> np.ndarray:
    """Confuse an image array (``uint8``, ``(H, W)`` or ``(H, W, C)``)."""
    return _apply(np.ascontiguousarray(arr), inverse=False)


def decrypt_array(arr: np.ndarray) -> np.ndarray:
    """Undo the confusion on an image array."""
    return _apply(np.ascontiguousarray(arr), inverse=True)


def _to_array(image):
    """Convert a PIL image to an RGB(A) numpy array, applying EXIF rotation."""
    from PIL import ImageOps

    # Browsers silently apply the EXIF orientation tag when drawing to canvas;
    # Pillow does not, so do it explicitly to stay compatible with the web tool.
    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")
    return np.array(image)


def decrypt_image(image):
    """Decrypt a PIL image, returning a new PIL image.

    Output is lossless (PNG-friendly) so no further generation loss is added.
    """
    from PIL import Image

    return Image.fromarray(decrypt_array(_to_array(image)))


def encrypt_image(image):
    """Encrypt a PIL image, returning a new PIL image."""
    from PIL import Image

    return Image.fromarray(encrypt_array(_to_array(image)))
