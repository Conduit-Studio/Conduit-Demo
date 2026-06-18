"""Conduit Run Code step: rank the YOLO fine-tune sweep and pick the winner.

Drop this on the canvas as a Run Code node (Container/CPU — it's pure Python, no GPU).

The sweep's Map exposes its per-trial outputs as SEPARATE, index-aligned lists (it can't
build a per-element object) and does NOT echo the trial item — so wire all three:
    trials  ◄ Config·JSON.trials   — the swept grid (name + hyperparameters per trial)
    metrics ◄ Map.metrics          — per-trial metrics  (json[])
    models  ◄ Map.model            — per-trial model artifacts (model-artifact[])
They are index-aligned (the Map iterates the trials in order), so trial i, metric i and
model i belong together. `select_best` zips them, ranks, and returns the winner WITH its
model + name + hyperparameters — wire `best.model` into a register/deploy step or read
`best.value` in a Choice gate.

(For the local mirror or a single combined feed, pass `results` instead — a list of
{name, model, metrics, hyperparameters} records; if present it takes precedence and
trials/metrics/models are ignored.)

Input ports
    trials  : json[] — the swept grid (each row carries name + hyperparameters).
    metrics : json[] — per-trial metrics, index-aligned with trials.
    models  : model-artifact[] — per-trial model refs, index-aligned with trials. Optional:
              at Verify time the model-artifact port isn't supplied, so best.model is empty
              (ranking by metric still works).
    results : (alternative) a list of combined per-trial records; if given, takes precedence.
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


def _zip_trials(trials: list, metrics: list, models: list) -> list[dict[str, Any]]:
    """Reunite the Map's separate, index-aligned per-trial lists into combined records.

    `TrialResult.from_item` reads `trial` (→ name + hyperparameters), `metrics` and `model`,
    so each zipped record is {trial, metrics, model}. `models` may be shorter than the trial
    grid (e.g. empty at Verify time) — a missing model just yields an empty ref.
    """
    count = max(len(trials), len(metrics))
    return [
        {
            "trial": trials[i] if i < len(trials) else {},      # name + swept hyperparameters
            "metrics": metrics[i] if i < len(metrics) else {},
            "model": models[i] if i < len(models) else {},
        }
        for i in range(count)
    ]


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    results = inputs.get("results")
    if not (isinstance(results, list) and results):
        # Canvas shape: three index-aligned lists from Config.trials + Map.metrics + Map.model.
        trials = inputs.get("trials")
        metrics = inputs.get("metrics")
        if isinstance(trials, list) and isinstance(metrics, list) and (trials or metrics):
            results = _zip_trials(trials, metrics, inputs.get("models") or [])
    if not (isinstance(results, list) and results):
        raise ValueError(
            "select_best needs either `results` (combined per-trial records) or index-aligned "
            "`trials` + `metrics` (+ optional `models`) from the sweep Map."
        )

    metric = str(inputs.get("metric") or DEFAULT_METRIC)
    best = select_best(results, metric)
    ranked = rank([TrialResult.from_item(item) for item in results], metric)

    return {
        "best": best.as_dict(),
        "ranking": [{"name": r.name, metric: r.metrics.get(metric)} for r in ranked],
    }
