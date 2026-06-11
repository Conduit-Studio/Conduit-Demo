"""Overlay rendering for keypoint predictions."""

from __future__ import annotations

import json
from typing import Any

from PIL import ImageDraw

from .image_io import open_rgb, png_bytes

SKELETON = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("nose", "left_eye"), ("nose", "right_eye"),
]


def render_overlay(image_bytes: bytes, prediction: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    image = open_rgb(image_bytes)
    draw = ImageDraw.Draw(image)
    points = {
        str(kp["name"]): (float(kp["x"]), float(kp["y"]))
        for kp in prediction.get("keypoints", [])
        if float(kp.get("confidence", 0.0)) > 0.1
    }
    for start, end in SKELETON:
        if start in points and end in points:
            draw.line((*points[start], *points[end]), fill=(178, 86, 47), width=3)
    for name, (x, y) in points.items():
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(37, 99, 143), outline=(255, 255, 255))
    draw.rectangle((6, 6, 154, 27), fill=(255, 255, 255), outline=(205, 207, 201))
    draw.text((10, 11), f"{prediction.get('label', 'object')} {prediction.get('confidence', 0):.2f}", fill=(36, 46, 56))
    summary = {
        "image_id": prediction.get("image_id"),
        "label": prediction.get("label"),
        "confidence": prediction.get("confidence"),
        "keypoint_count": len(points),
    }
    # Verify JSON serializability before writing labels downstream.
    json.dumps(summary)
    return png_bytes(image), summary
