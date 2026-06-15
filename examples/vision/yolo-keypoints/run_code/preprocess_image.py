"""Conduit Run Code step: preprocess one image for YOLO keypoint inference.

Inputs:
    image: file port — local path to one raw source image (Conduit downloads it).
    image_name: json — the original filename, used to derive the logical image id.
    target_size: json — letterbox target edge length (default 160).

Output ports:
    preprocessed: file port — local path to the processed PNG (Conduit uploads it).
    image_id: json — the logical image id, threaded to downstream nodes.
    metadata: json — preprocessing metadata.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from yolo_keypoints.coco import image_id_from_name
from yolo_keypoints.preprocessing import preprocess_image_bytes


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    image_path = inputs["image"]                                   # file port -> local path
    image_name = str(inputs.get("image_name") or "image.png")      # json: original filename (for id)
    target_size = int(inputs.get("target_size", 160))
    image_bytes = Path(image_path).read_bytes()
    processed_bytes, metadata = preprocess_image_bytes(image_bytes, target_size=target_size)
    image_id = image_id_from_name(image_name)
    out_path = str(Path(tempfile.gettempdir()) / f"{image_id}-processed.png")
    Path(out_path).write_bytes(processed_bytes)
    return {"preprocessed": out_path, "image_id": image_id, "metadata": metadata}
