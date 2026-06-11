"""Preprocessing helpers for keypoint images."""

from __future__ import annotations

from PIL import Image, ImageOps

from .image_io import image_metadata, open_rgb, png_bytes


def preprocess_image_bytes(image_bytes: bytes, *, target_size: int = 160) -> tuple[bytes, dict[str, object]]:
    if target_size < 32:
        raise ValueError("target_size must be at least 32")
    image = open_rgb(image_bytes)
    source_meta = image_metadata(image)
    prepared = ImageOps.contain(image, (target_size, target_size))
    canvas = Image.new("RGB", (target_size, target_size), (250, 250, 247))
    offset = ((target_size - prepared.width) // 2, (target_size - prepared.height) // 2)
    canvas.paste(prepared, offset)
    return png_bytes(canvas), {
        "source": source_meta,
        "output": image_metadata(canvas, target_size=target_size),
        "letterbox_offset": {"x": int(offset[0]), "y": int(offset[1])},
    }
