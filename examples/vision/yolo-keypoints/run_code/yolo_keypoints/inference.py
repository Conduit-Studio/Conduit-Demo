"""Deterministic YOLO-style keypoint inference for the public demo.

The demo does not download a model during verification. It emits a COCO-shaped
pose prediction from the checked-in annotation file, while keeping the same
input and output contract a real YOLO pose model would use.
"""

from __future__ import annotations

from typing import Any

from .coco import (
    annotation_for_image,
    decode_keypoints,
)


def estimate_keypoints(image_id: str, annotations: dict[str, Any]) -> dict[str, Any]:
    # Deterministic demo: derive the pose from the checked-in COCO annotations (keyed by
    # image_id) so verification needs no model download. A real YOLO-pose model would instead
    # decode the image pixels — give the node an `image` file port and read it here.
    annotation = annotation_for_image(annotations, image_id)
    keypoints = decode_keypoints(annotation)
    visible = [kp for kp in keypoints if kp["visibility"] > 0]
    confidence = round(sum(float(kp["confidence"]) for kp in visible) / max(len(visible), 1), 3)
    return {
        "image_id": image_id,
        "label": "person",
        "model": {"name": "yolo-keypoints-demo", "version": "pose-demo-v1"},
        "confidence": confidence,
        "source_annotation_id": annotation.get("id"),
        "keypoints": keypoints,
    }
