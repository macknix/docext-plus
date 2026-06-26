"""Image preparation for the VLM.

docext's prompt builder hard-codes the data URL mime type as ``image/jpeg``
(see docext/core/prompts.py), so the bytes we hand it must actually be JPEG.
TIFFs - especially bitonal/grayscale scans - therefore have to be converted
first. We also downscale very large scans so the request stays within the
model's context window.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def set_max_image_pixels(max_pixels: int | None) -> None:
    """Configure Pillow's decompression-bomb guard.

    Research scans can legitimately exceed Pillow's ~89 MP default, which emits a
    ``DecompressionBombWarning``. We raise the limit to ``max_pixels`` so normal
    large scans pass quietly; Pillow still raises ``DecompressionBombError`` above
    2x this value, so a guard against pathological inputs remains. ``None``
    disables the check entirely.
    """
    Image.MAX_IMAGE_PIXELS = max_pixels


def list_documents(input_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return sorted document paths in ``input_dir`` matching ``extensions``."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    exts = {e.lower() for e in extensions}
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return files


def tiff_to_jpeg(
    src: Path,
    dst_dir: Path,
    max_size: int = 2048,
    quality: int = 95,
    rotate: float = 0.0,
) -> Path:
    """Convert a (single-page) TIFF to a JPEG suitable for the VLM.

    - Uses the first frame only (inputs are expected to be single page).
    - Flattens to RGB so JPEG encoding always succeeds.
    - Optionally rotates ``rotate`` degrees clockwise (canvas expanded, new
      corners filled white).
    - Downscales so the longest side is at most ``max_size`` (aspect preserved).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        # Take the first frame for multi-frame TIFFs; harmless for single page.
        try:
            img.seek(0)
        except EOFError:
            pass

        img = img.convert("RGB")

        if rotate:
            # PIL rotates counter-clockwise for positive angles; negate so the
            # config value means clockwise (the intuitive direction). expand keeps
            # the whole page; white fill suits document scans.
            img = img.rotate(-rotate, expand=True, fillcolor="white")

        longest = max(img.size)
        if max_size and longest > max_size:
            scale = max_size / longest
            new_size = (round(img.width * scale), round(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        dst = dst_dir / f"{src.stem}.jpg"
        img.save(dst, format="JPEG", quality=quality)
    return dst
