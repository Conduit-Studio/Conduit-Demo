"""Conduit Run Code step: preprocess one image for YOLO keypoint inference.

Inputs:
    image/object/source: S3 ref or s3:// URI for one copied source image.
    output_bucket: bucket where the processed image should be written.
    output_prefix: S3 prefix for processed images.

Output port:
    preprocessed: S3 ref to the processed PNG artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from typing import Any

_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from yolo_keypoints.preprocessing import preprocess_image_bytes
from yolo_keypoints.s3 import S3Ref, join_key, output_bucket, read_bytes, require_ref, write_bytes


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    image_ref = require_ref(inputs, "image", aliases=("object", "source"))
    local_s3_root = inputs.get("_local_s3_root")
    local_output_root = inputs.get("_local_output_root")
    target_size = int(inputs.get("target_size", 160))
    prefix = str(inputs.get("output_prefix") or "data/processed/images/")
    image_bytes = read_bytes(image_ref, local_s3_root=local_s3_root)
    processed_bytes, metadata = preprocess_image_bytes(image_bytes, target_size=target_size)
    image_id = PurePosixPath(image_ref.key).stem
    out_ref = S3Ref(bucket=output_bucket(inputs, image_ref), key=join_key(prefix, f"{image_id}-processed.png"))
    write_bytes(out_ref, processed_bytes, content_type="image/png", local_output_root=local_output_root)
    return {
        "preprocessed": out_ref.as_dict(),
        "image_id": image_id,
        "metadata": metadata,
    }
