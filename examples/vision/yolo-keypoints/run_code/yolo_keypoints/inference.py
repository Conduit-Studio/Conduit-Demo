"""Deterministic YOLO-style keypoint inference for the public demo.

The demo does not download a model during verification. It emits a COCO-shaped
pose prediction from the checked-in annotation file, while keeping the same
input and output contract a real YOLO pose model would use.
"""

from __future__ import annotations

from typing import Any

from .coco import (
    KEYPOINT_NAMES,
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


def keypoints_from_label(
    label_text: str,
    *,
    source: dict[str, Any],
    target_size: int,
    offset: dict[str, Any],
) -> dict[str, Any]:
    """Decode one YOLO-pose `.txt` label into a COCO-shaped pose, mapped onto the
    letterboxed `target_size` canvas. `source` (original dims) and `offset` come from
    `preprocess_image_bytes` metadata. No model download and no shared-annotation lookup —
    the per-image label IS the ground truth a real YOLO-pose model would predict.
    """
    line = next((ln for ln in label_text.splitlines() if ln.strip()), "")
    nums = [float(tok) for tok in line.split()]
    expected = 5 + len(KEYPOINT_NAMES) * 3
    if len(nums) < expected:
        raise ValueError(
            f"YOLO-pose label needs class+box+{len(KEYPOINT_NAMES)} (x,y,v) keypoints; got {len(nums)} values"
        )
    width = float(source["width"])
    height = float(source["height"])
    scale = min(target_size / width, target_size / height)
    scaled_w, scaled_h = width * scale, height * scale
    ox, oy = float(offset["x"]), float(offset["y"])
    keypoints: list[dict[str, Any]] = []
    for index, name in enumerate(KEYPOINT_NAMES):
        nx, ny, visibility = nums[5 + index * 3 : 5 + index * 3 + 3]
        keypoints.append({
            "name": name,
            # normalized-to-original → scaled → letterboxed-canvas pixel
            "x": ox + nx * scaled_w,
            "y": oy + ny * scaled_h,
            "visibility": int(visibility),
            "confidence": 0.9 if int(visibility) > 0 else 0.0,
        })
    visible = [kp for kp in keypoints if kp["visibility"] > 0]
    confidence = round(sum(kp["confidence"] for kp in visible) / max(len(visible), 1), 3)
    return {
        "label": "person",
        "model": {"name": "yolo-pose-label", "version": "label-v1"},
        "confidence": confidence,
        "keypoints": keypoints,
    }
