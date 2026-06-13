"""Conduit Run Code step: rank the YOLO fine-tune sweep and pick the winner.

Drop this on the canvas as a Run Code node (Container/CPU — it's pure Python, no GPU).
Wire the Map's output into its `results` input; wire its `best.model` output into a
register/deploy step or read `best.value` in a Choice gate.

Input ports
    results : the Map output — a list of per-trial records, each like
              { "name": "s-1280",
                "model": { "bucket": "...", "key": ".../model.tar.gz" },   # s3-ref
                "metrics": { "mAP50-95": 0.71, "mAP50": 0.89 },
                "hyperparameters": { "weights": "yolov8s.pt", "imgsz": 1280, ... } }
    metric  : (optional) which metric to rank by; defaults to "mAP50-95".

Output ports
    best    : { name, metric, value, model:s3-ref, hyperparameters } — the winning trial.
    ranking : trials sorted best→worst (name + score) for the run log / a Slack notify.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the sibling package importable when Conduit runs this file as the entrypoint.
_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from yolo_finetune.schemas import TrialResult  # noqa: E402
from yolo_finetune.selection import DEFAULT_METRIC, rank, select_best  # noqa: E402


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    results = inputs.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("inputs['results'] must be the non-empty Map output of the sweep")

    metric = str(inputs.get("metric") or DEFAULT_METRIC)
    best = select_best(results, metric)
    ranked = rank([TrialResult.from_item(i) for i in results], metric)

    return {
        "best": best.as_dict(),
        "ranking": [{"name": r.name, metric: r.metrics.get(metric)} for r in ranked],
    }
