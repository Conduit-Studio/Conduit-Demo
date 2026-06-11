"""Conduit Run Code step: run YOLO-style keypoint inference on one image.

Inputs:
    image/preprocessed/object: S3 ref for the preprocessed image.
    annotations: S3 ref for a COCO-style keypoint annotation JSON.
    output_bucket: bucket where the prediction JSON should be written.
    output_prefix: S3 prefix for prediction JSON files.

Output port:
    predictions: S3 ref to the label JSON artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from yolo_keypoints.coco import image_id_from_key
from yolo_keypoints.inference import estimate_keypoints
from yolo_keypoints.s3 import S3Ref, join_key, output_bucket, read_json, require_ref, write_json


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    image_ref = require_ref(inputs, "image", aliases=("preprocessed", "object"))
    annotations_ref = require_ref(inputs, "annotations")
    local_s3_root = inputs.get("_local_s3_root")
    local_output_root = inputs.get("_local_output_root")
    prefix = str(inputs.get("output_prefix") or "data/predictions/")
    annotations = read_json(annotations_ref, local_s3_root=local_s3_root)
    prediction = estimate_keypoints(image_ref.key, annotations)
    image_id = image_id_from_key(image_ref.key)
    out_ref = S3Ref(bucket=output_bucket(inputs, image_ref), key=join_key(prefix, f"{image_id}-keypoints.json"))
    write_json(out_ref, prediction, local_output_root=local_output_root)
    return {
        "predictions": out_ref.as_dict(),
        "prediction": prediction,
    }
