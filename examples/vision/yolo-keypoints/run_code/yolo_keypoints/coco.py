"""COCO keypoint parsing for the tiny demo dataset."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def image_id_from_key(key: str) -> str:
    stem = PurePosixPath(key).stem
    for suffix in ("-processed", "-overlay", "-keypoints"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def annotation_for_image(coco: dict[str, Any], image_key: str) -> dict[str, Any]:
    image_id = image_id_from_key(image_key)
    image = next((item for item in coco.get("images", []) if PurePosixPath(str(item.get("file_name", ""))).stem == image_id), None)
    if image is None:
        raise ValueError(f"COCO annotations do not include image {image_id!r}")
    annotation = next((item for item in coco.get("annotations", []) if item.get("image_id") == image.get("id")), None)
    if annotation is None:
        raise ValueError(f"COCO annotations do not include keypoints for {image_id!r}")
    return annotation


def decode_keypoints(annotation: dict[str, Any], *, y_offset: int = 20) -> list[dict[str, Any]]:
    raw = list(annotation.get("keypoints", []))
    if len(raw) != len(KEYPOINT_NAMES) * 3:
        raise ValueError("COCO keypoints must contain x, y, visibility triples for 17 keypoints")
    decoded: list[dict[str, Any]] = []
    for index, name in enumerate(KEYPOINT_NAMES):
        x, y, visibility = raw[index * 3 : index * 3 + 3]
        decoded.append({
            "name": name,
            "x": float(x),
            "y": float(y) + y_offset,
            "visibility": int(visibility),
            "confidence": 0.93 if int(visibility) > 0 else 0.0,
        })
    return decoded
