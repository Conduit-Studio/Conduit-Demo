"""Conduit Run Code step: run YOLO-style keypoint inference on one image.

Inputs:
    annotations: file port — local path to a COCO-style keypoint annotation JSON.
    image_id: json — logical image id (threaded from preprocess).

A real YOLO-pose model would also take an `image` file port and run inference on the
pixels; this deterministic demo derives the pose from the annotations + image_id.

Output ports:
    predictions: file port — local path to the prediction JSON (Conduit uploads it).
    prediction: json — the inline prediction object.
    image_id: json — the logical image id, threaded downstream.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from yolo_keypoints.inference import estimate_keypoints


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    anno_path = inputs["annotations"]       # file port (COCO json)
    image_id = str(inputs["image_id"])      # json (threaded from preprocess)
    annotations = json.loads(Path(anno_path).read_text())
    prediction = estimate_keypoints(image_id, annotations)
    out_path = str(Path(tempfile.gettempdir()) / f"{image_id}-keypoints.json")
    Path(out_path).write_text(json.dumps(prediction))
    return {"predictions": out_path, "prediction": prediction, "image_id": image_id}
