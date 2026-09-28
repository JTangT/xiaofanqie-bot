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

import array
import math
import os
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
#
# Peak memory for the *whole* pipeline (decode -> permute -> re-encode PNG) was
# measured at ~40 bytes per pixel: 122 MB at 2 MP, 478 MB at 12 MP. 8 MP keeps
# the worst case near 320 MB, which fits a 512 MB container with headroom, and
# still covers everything Telegram realistically delivers -- a compressed photo
# is ~1 MP and a 4K screenshot is ~8 MP. Override with the MAX_PIXELS
# environment variable if your host has RAM to spare.
DEFAULT_MAX_PIXELS = 8_000_000

try:
    MAX_PIXELS = int(os.environ.get("MAX_PIXELS", "").strip() or DEFAULT_MAX_PIXELS)
    if MAX_PIXELS <= 0:
        raise ValueError
except ValueError:
    MAX_PIXELS = DEFAULT_MAX_PIXELS

# Distinct (width, height) index tables kept in memory. Each costs 4 bytes per
# pixel (46 MB at 12 MP), so keep this small on memory-constrained hosts. Two
# is enough in practice: photos from one device tend to share dimensions.
_CACHE_SIZE = 2
_CACHE_SIZE = 8


class AlgorithmError(ValueError):
    """Raised when an image cannot be processed by the algorithm."""


def _build_curve(width: int, height: int) -> array.array:
    """Hilbert curve as flat pixel indices ``x + y * width``, in visit order.

    Direct port of the reference ``gilbert2d`` / ``generate2d``. Visits every
    pixel exactly once; works for non-square and odd-sized images.

    Memory matters a lot here. The obvious way to collect the coordinates is a
    Python ``list`` of ``(x, y)`` tuples, and that is a bomb: at 12 MP the list
    alone costs ~1.2 GB (roughly 100 bytes per tuple), which is enough to OOM
    a small VPS and take the whole host down with it. An ``array.array`` of
    int64 is contiguous and costs 8 bytes per pixel instead (~96 MB at 12 MP).

    ``gilbert2d`` below decodes its ``(x, y)`` pairs from this array, so there
    is exactly one implementation of the curve and the two views can never
    drift apart.
    """
    if width <= 0 or height <= 0:
        raise AlgorithmError(f"invalid dimensions: {width}x{height}")

    flat: array.array = array.array("q")

    def generate(x: int, y: int, ax: int, ay: int, bx: int, by: int) -> None:
        w = abs(ax + ay)
        h = abs(bx + by)
        dax = (ax > 0) - (ax < 0)
        day = (ay > 0) - (ay < 0)
        dbx = (bx > 0) - (bx < 0)
        dby = (by > 0) - (by < 0)

        if h == 1:
            for _ in range(w):
                flat.append(x + y * width)
                x += dax
                y += day
            return

        if w == 1:
            for _ in range(h):
                flat.append(x + y * width)
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

    return flat


def gilbert2d(width: int, height: int) -> list[tuple[int, int]]:
    """Generate the generalized Hilbert curve as a list of ``(x, y)`` points.

    Convenience/inspection view over :func:`_build_curve`, kept for parity with
    the reference implementation's API. Prefer :func:`gilbert_curve_indices`
    for real work -- this one materialises a tuple per pixel and therefore has
    the large memory footprint described above.
    """
    flat = _build_curve(width, height)
    return [(int(i) % width, int(i) // width) for i in flat]


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
    # _build_curve returns an int64 array.array; copy into the narrow dtype so
    # the cached array is as small as possible. The array.array itself is
    # released as soon as this function returns.
    return np.frombuffer(_build_curve(width, height), dtype=np.int64).astype(
        dtype, copy=True
    )


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
