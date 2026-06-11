"""Conduit Run Code step: render keypoint labels into a visual preview artifact.

Inputs:
    image/preprocessed: S3 ref for the processed image.
    predictions: S3 ref for prediction JSON, or an inline prediction object.
    output_bucket: bucket where overlay and label artifacts should be written.
    output_prefix: S3 prefix for overlay PNG files.

Output ports:
    overlay: S3 ref to a PNG preview.
    labels: S3 ref to a small JSON summary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from yolo_keypoints.coco import image_id_from_key
from yolo_keypoints.rendering import render_overlay
from yolo_keypoints.s3 import S3Ref, join_key, output_bucket, parse_ref, read_bytes, read_json, require_ref, write_bytes, write_json


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    image_ref = require_ref(inputs, "image", aliases=("preprocessed",))
    prediction = _prediction_from_input(inputs, local_s3_root=inputs.get("_local_s3_root"))
    local_s3_root = inputs.get("_local_s3_root")
    local_output_root = inputs.get("_local_output_root")
    prefix = str(inputs.get("output_prefix") or "data/outputs/overlays/")
    image_bytes = read_bytes(image_ref, local_s3_root=local_s3_root)
    overlay_bytes, summary = render_overlay(image_bytes, prediction)
    image_id = image_id_from_key(image_ref.key)
    bucket = output_bucket(inputs, image_ref)
    overlay_ref = S3Ref(bucket=bucket, key=join_key(prefix, f"{image_id}-overlay.png"))
    labels_ref = S3Ref(bucket=bucket, key=join_key(prefix, f"{image_id}-labels.json"))
    write_bytes(overlay_ref, overlay_bytes, content_type="image/png", local_output_root=local_output_root)
    write_json(labels_ref, {"summary": summary, "prediction": prediction}, local_output_root=local_output_root)
    return {
        "overlay": overlay_ref.as_dict(),
        "labels": labels_ref.as_dict(),
        "preview": overlay_ref.as_dict(),
        "summary": summary,
    }


def _prediction_from_input(inputs: dict[str, Any], *, local_s3_root: str | None) -> dict[str, Any]:
    value = inputs.get("predictions") or inputs.get("prediction")
    ref = parse_ref(value)
    if ref is not None:
        return read_json(ref, local_s3_root=local_s3_root)
    if isinstance(value, dict) and isinstance(value.get("prediction"), dict):
        return value["prediction"]
    if isinstance(value, dict) and value.get("keypoints"):
        return value
    raise ValueError("inputs['predictions'] must be an S3 ref or an inline prediction object")
