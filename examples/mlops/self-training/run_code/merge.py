"""Conduit Run Code step: the loop body's ROUND SUMMARY (and the loop's sink).

Drop this on the canvas as a Run Code node (Container/CPU — pure Python, no GPU).
It is the LAST step of the loop body:

    flow.loop[ train -> pseudo_label -> select_confident -> merge ]
                                                            ^^^^^

`merge` produces the loop's UPDATED vars for the next round:

  1. Folds this round's accepted batch into the dataset — labelled GROWS by the
     batch, pool SHRINKS by exactly the batch's ids (delegated to the pure,
     unit-tested `self_training.merge.merge_round`).
  2. Re-emits the round's `metrics`. The `flow.loop` stop condition reads
     `$.loopState.metrics.accuracy`, so the loop's stop-only `metrics` var is
     `updatedBy` this output. (Registering the final model after the loop is a
     follow-on — see config/loop.json — not in this example.)

CANVAS RULE — a code.run node's port names must be UNIQUE across inputs+outputs, so the
updated dataset goes out under DIFFERENT names than it came in: the inputs are the CURRENT
`trainingData` / `pool`, the outputs are the NEXT round's `trainingDataNext` / `poolNext`.
The loop vars wire `trainingData.updatedBy = trainingDataNext` and `pool.updatedBy = poolNext`.

Input ports
    trainingData : json[] — current labelled rows {id, label, path}. (Loop var `trainingData`.)
    pool         : json[] — current unlabeled pool rows {id, path} (label absent). (Loop var `pool`.)
    batch        : json[] — select_confident's kept pseudo-labels {id, label, confidence}.
    metricsIn    : json — this round's metrics, e.g. {"accuracy": 0.71} (re-emitted as `metrics`).
                   (Distinct input name so inputs and outputs never collide.)

Output ports (these feed the loop vars the next round re-feeds + the loop sink)
    trainingDataNext : json[] — updated labelled set (trainingData + batch). → loop var `trainingData`.
    poolNext         : json[] — updated pool (pool - batch ids).            → loop var `pool`.
    metrics          : json — this round's metrics (drives the stop check).  → loop var `metrics`.
    new_confident    : number — count added this round (0 ⇒ plateau).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the sibling package importable when Conduit runs this file as the entrypoint.
_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from self_training.merge import merge_round  # noqa: E402


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    training_data = inputs.get("trainingData") or []
    pool = inputs.get("pool") or []
    batch = inputs.get("batch") or []
    if not isinstance(training_data, list) or not isinstance(pool, list) or not isinstance(batch, list):
        raise ValueError("merge needs list-valued `trainingData`, `pool`, and `batch`.")

    next_training_data, next_pool = merge_round(training_data, pool, batch)
    return {
        "trainingDataNext": next_training_data,   # → loop var `trainingData` (updatedBy)
        "poolNext": next_pool,                     # → loop var `pool` (updatedBy)
        "metrics": inputs.get("metricsIn"),        # → loop var `metrics`; stop reads metrics.accuracy
        "new_confident": len(batch),
    }
