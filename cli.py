#!/usr/bin/env python3
"""Command-line interface for the 小番茄图片混淆 algorithm.

Examples:
    python cli.py decrypt confused.jpg -o restored.png
    python cli.py encrypt photo.png -o confused.png
    python cli.py decrypt batch/*.jpg -d out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from image_confuse import AlgorithmError, decrypt_image, encrypt_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="小番茄图片混淆 / 解混淆命令行工具",
    )
    parser.add_argument("mode", choices=["encrypt", "decrypt"], help="操作模式")
    parser.add_argument("inputs", nargs="+", type=Path, help="输入图片（可多个）")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出文件（仅在单个输入时可用）",
    )
    parser.add_argument(
        "-d",
        "--outdir",
        type=Path,
        help="输出目录（多个输入时使用，默认为输入文件所在目录）",
    )
    parser.add_argument(
        "--jpeg",
        action="store_true",
        help="输出 JPEG（有损，默认输出无损 PNG）",
    )
    args = parser.parse_args(argv)

    from PIL import Image, UnidentifiedImageError

    if args.output and len(args.inputs) > 1:
        parser.error("-o/--output 只能用于单个输入文件，多文件请用 -d/--outdir")
    if args.output and args.outdir:
        parser.error("-o/--output 与 -d/--outdir 不能同时使用")

    suffix = "_encrypted" if args.mode == "encrypt" else "_decrypted"
    failures = 0

    for src in args.inputs:
        if not src.is_file():
            print(f"[跳过] 不是文件: {src}", file=sys.stderr)
            failures += 1
            continue

        if args.output:
            dst = args.output
        else:
            outdir = args.outdir or src.parent
            outdir.mkdir(parents=True, exist_ok=True)
            ext = ".jpg" if args.jpeg else ".png"
            dst = outdir / f"{src.stem}{suffix}{ext}"

        try:
            with Image.open(src) as image:
                image.load()
                width, height = image.size
                result = (
                    encrypt_image(image)
                    if args.mode == "encrypt"
                    else decrypt_image(image)
                )

            if args.jpeg:
                result.save(dst, format="JPEG", quality=95)
            else:
                result.save(dst, format="PNG", optimize=True)

            print(f"[完成] {src} ({width}x{height}) -> {dst}")
        except AlgorithmError as exc:
            print(f"[失败] {src}: {exc}", file=sys.stderr)
            failures += 1
        except UnidentifiedImageError:
            print(f"[失败] {src}: 无法识别的图片格式", file=sys.stderr)
            failures += 1
        except OSError as exc:
            print(f"[失败] {src}: 读取或写入失败 ({exc})", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
