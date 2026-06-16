"""Conduit Run Code step: REAL YOLO-pose keypoint inference on one image.

Loads an ultralytics YOLO-pose model and runs it on the image PIXELS — real inference,
no ground-truth label. Returns the annotated overlay PNG and a predicted-labels file.

This is the real-inference entry the Canvas workflow uses:
    S3 List (images) -> Map[Run Code · Container: infer_keypoints]
One image per item (a standard single-ref Map) — no input label, no pairing. Runtime:
Container (ultralytics + torch); the weights are baked into the image (env
`YOLO_POSE_WEIGHTS`) or auto-downloaded by ultralytics at cold start.

Input port:
    image: s3-ref — the image (Conduit downloads it; the code gets a local path)

Output ports:
    overlay: s3-ref — annotated PNG with the detected pose (Conduit uploads it; preview on canvas)
    labels:  s3-ref — predicted keypoints as a YOLO-pose .txt (saved back to S3)
    summary: json    — person count + per-detection confidence
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Weights: baked into the Container image, or auto-downloaded by ultralytics at cold start.
WEIGHTS = os.environ.get("YOLO_POSE_WEIGHTS", "yolo11n-pose.pt")

_model = None


def _model_load():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(WEIGHTS)
    return _model


def _yolo_pose_label(result) -> str:
    """Predicted detections → YOLO-pose label text (one line per person):
    `class cx cy w h (kx ky kv)x17`, all box/keypoint coords normalized to the image."""
    height, width = result.orig_shape
    boxes, keypoints = result.boxes, result.keypoints
    if boxes is None or keypoints is None:
        return ""
    xywhn = boxes.xywhn.cpu().numpy()                 # normalized cx, cy, w, h
    kxy = keypoints.xy.cpu().numpy()                  # pixel x, y per person
    kconf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
    lines: list[str] = []
    for i in range(len(xywhn)):
        cx, cy, bw, bh = xywhn[i]
        parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for j in range(len(kxy[i])):
            kx, ky = kxy[i][j]
            c = float(kconf[i][j]) if kconf is not None else 1.0
            visibility = 2 if c > 0.5 else (1 if c > 0.0 else 0)
            parts += [f"{kx / width:.6f}", f"{ky / height:.6f}", str(visibility)]
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    model = _model_load()
    result = model(inputs["image"], verbose=False)[0]    # real inference on the pixels

    overlay_path = str(Path(tempfile.gettempdir()) / "overlay.png")
    annotated_bgr = result.plot()                         # ultralytics draws the skeleton (BGR ndarray)
    Image.fromarray(annotated_bgr[..., ::-1]).save(overlay_path)  # BGR -> RGB

    labels_path = str(Path(tempfile.gettempdir()) / "labels.txt")
    Path(labels_path).write_text(_yolo_pose_label(result))

    boxes = result.boxes
    confidences = boxes.conf.cpu().numpy().tolist() if boxes is not None and boxes.conf is not None else []
    return {
        "overlay": overlay_path,
        "labels": labels_path,
        "summary": {"persons": len(confidences), "confidence": [round(c, 3) for c in confidences]},
    }
