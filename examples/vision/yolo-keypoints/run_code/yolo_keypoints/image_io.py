"""Image loading and PNG serialization helpers."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image, ImageOps


def open_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_metadata(image: Image.Image, **extra: Any) -> dict[str, Any]:
    payload = {"width": int(image.width), "height": int(image.height), "mode": image.mode}
    payload.update(extra)
    return payload
