"""Conduit Run Code step: rank the YOLO fine-tune sweep and pick the winner.

Drop this on the canvas as a Run Code node (Container/CPU — it's pure Python, no GPU).

The sweep's Map emits a single self-describing `results` array — one element per trial,
each carrying the trial config, its trained model artifact, and its eval metrics:

    results ◄ Map.results   — list of { item: <trial config>, index, model, metrics }

`select_best` flattens each element (extracting name + hyperparameters from `item`),
ranks by the chosen metric, and returns the winner WITH its model + name +
hyperparameters — wire `best.model` into a register/deploy step or read `best.value`
in a Choice gate.

Input ports
    results : json[]  — the self-describing per-trial records emitted by the sweep Map.
                        Each element: { item: {name, ...hyperparameters}, index, model, metrics }.
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
    if not (isinstance(results, list) and results):
        raise ValueError(
            "select_best requires `results` — the self-describing per-trial array from "
            "the sweep Map (each element: { item: <trial config>, model, metrics })."
        )

    # Flatten each self-describing element into the shape `from_item` expects:
    # { name, hyperparameters, metrics, model }.  The trial config lives under "item"
    # (which carries name + all hyperparameters); model and metrics are siblings.
    flat = []
    for r in results:
        item = r.get("item") or {}
        flat.append({
            "name": item.get("name") or "trial",
            "hyperparameters": item,
            "metrics": r.get("metrics") or {},
            "model": r.get("model") or {},
        })

    metric = str(inputs.get("metric") or DEFAULT_METRIC)
    best = select_best(flat, metric)
    ranked = rank([TrialResult.from_item(i) for i in flat], metric)

    return {
        "best": best.as_dict(),
        "ranking": [{"name": r.name, metric: r.metrics.get(metric)} for r in ranked],
    }
