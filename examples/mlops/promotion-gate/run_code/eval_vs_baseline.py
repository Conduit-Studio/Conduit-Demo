"""Conduit Run Code node: evaluate a candidate model vs the current-prod baseline.

Drop this on the canvas as a Run Code node (Container/CPU — pure Python, no GPU). Wire the
candidate's `metrics` (e.g. a Train Model's `metrics` output) and the `baseline` (a Config·JSON
value holding the current-prod metric) into its inputs; wire its `beats` output into a Choice gate
and its `report` output into the Wait for Approval node's `value` (the evidence the reviewer sees).

Input ports
    metrics  : the candidate's eval metrics, e.g. { "accuracy": 0.96 }
    baseline : the current-prod metric to beat, e.g. { "accuracy": 0.93 }
    metric   : (optional) which metric to compare; defaults to "accuracy".
    minDelta : (optional) minimum improvement required to count as beating; defaults to 0.
    modelPackageArn : (optional) carried into the report so the approve branch knows what to promote.

Output ports
    report : { metric, candidate, baseline, delta, beats, modelPackageArn } — shown to the reviewer.
    beats  : bool — did the candidate beat the baseline? (drives the Choice gate)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the sibling package importable when Conduit runs this file as the entrypoint.
_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from promotion_gate.evaluation import evaluate  # noqa: E402


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    metrics = inputs.get("metrics")
    # Input-wins: a wired `baseline` takes precedence; otherwise fall back to the BASELINE
    # env constant (the single-root deploy shape — the baseline is folded into this node).
    baseline = inputs.get("baseline")
    if baseline is None:
        raw = os.environ.get("BASELINE", "")
        baseline = json.loads(raw) if raw else None
    if not isinstance(metrics, dict) or not isinstance(baseline, dict):
        raise ValueError("inputs['metrics'] and inputs['baseline'] must both be metric objects")
    return evaluate(
        metrics,
        baseline,
        metric=str(inputs.get("metric") or "accuracy"),
        min_delta=float(inputs.get("minDelta") or 0.0),
        model_package_arn=inputs.get("modelPackageArn"),
    )
