"""Conduit Run Code step: render keypoint labels into a visual preview artifact.

Inputs:
    image: file port — local path to the processed image (Conduit downloads it).
    image_id: json — logical image id (threaded from upstream).
    prediction: json — an inline prediction object, OR
    predictions: file port — local path to a prediction JSON.

Output ports:
    overlay: file port — local path to a PNG preview (Conduit uploads it).
    labels: file port — local path to a small JSON summary (Conduit uploads it).
    summary: json — a compact label summary.
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

from yolo_keypoints.rendering import render_overlay


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    image_path = inputs["image"]            # file port (processed image)
    image_id = str(inputs["image_id"])      # json
    prediction = _prediction_from_input(inputs)   # see below
    overlay_bytes, summary = render_overlay(Path(image_path).read_bytes(), prediction)
    overlay_path = str(Path(tempfile.gettempdir()) / f"{image_id}-overlay.png")
    labels_path = str(Path(tempfile.gettempdir()) / f"{image_id}-labels.json")
    Path(overlay_path).write_bytes(overlay_bytes)
    Path(labels_path).write_text(json.dumps({"summary": summary, "prediction": prediction}))
    return {"overlay": overlay_path, "labels": labels_path, "summary": summary}


def _prediction_from_input(inputs: dict[str, Any]) -> dict[str, Any]:
    # accept an inline "prediction" dict, OR a "predictions" file port (local path to prediction json)
    value = inputs.get("prediction")
    if isinstance(value, dict) and value.get("keypoints"):
        return value
    path = inputs.get("predictions")
    if isinstance(path, str) and path:
        return json.loads(Path(path).read_text())
    raise ValueError("render needs inputs['prediction'] (inline) or inputs['predictions'] (file path)")
