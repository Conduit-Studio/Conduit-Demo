"""Conduit Run Code step: the full per-image keypoint pipeline in ONE body (record-of-refs).

Reads TWO per-item files — the image and its per-image YOLO-pose label — and runs
preprocess -> keypoints -> render in sequence inside a single Map iteration. There is no
Map boundary between the steps, so each image's intermediate data (the processed image,
the pose) flows through ordinary local variables.

This is the record-mode shape: the Map item is a ``{image, label}`` record, "record items"
is ON, and Conduit downloads both refs and hands the body two local file paths.

Inputs (Map "record items" — both are file ports):
    image: s3-ref — the raw image (Conduit hands a local path)
    label: s3-ref — its per-image YOLO-pose ``.txt`` label (a local path)

Output ports:
    overlay: s3-ref — the overlay PNG (Conduit uploads it)
    summary: json   — a compact keypoint summary
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from yolo_keypoints.preprocessing import preprocess_image_bytes
from yolo_keypoints.inference import keypoints_from_label
from yolo_keypoints.rendering import render_overlay

TARGET_SIZE = 160


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    image_bytes = Path(inputs["image"]).read_bytes()   # record field 1 (file port)
    label_text = Path(inputs["label"]).read_text()     # record field 2 (file port)

    # preprocess -> keypoints -> render, all per-image, in one body (no Map boundary)
    processed_bytes, metadata = preprocess_image_bytes(image_bytes, target_size=TARGET_SIZE)
    prediction = keypoints_from_label(
        label_text,
        source=metadata["source"],
        target_size=TARGET_SIZE,
        offset=metadata["letterbox_offset"],
    )
    overlay_bytes, summary = render_overlay(processed_bytes, prediction)

    out_path = str(Path(tempfile.gettempdir()) / "overlay.png")
    Path(out_path).write_bytes(overlay_bytes)
    return {"overlay": out_path, "summary": summary}
