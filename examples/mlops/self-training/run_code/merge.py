"""Conduit Run Code step: the loop body's ROUND SUMMARY (and the loop's sink).

Drop this on the canvas as a Run Code node (Container/CPU — pure Python, no GPU).
It is the LAST step of the loop body:

    flow.loop[ train -> pseudo_label -> select_confident -> merge ]
                                                            ^^^^^

`merge` does two jobs at once:

  1. Folds this round's accepted batch into the dataset for the NEXT round —
     labelled GROWS by the batch, pool SHRINKS by exactly the batch's ids
     (delegated to the pure, unit-tested `self_training.merge.merge_round`).
  2. PASSES THROUGH the round's `model` + `metrics` so the loop's updated vars
     carry the latest artifact + accuracy forward. The `flow.loop` stop condition
     reads `$.loopState.metrics.accuracy`, so threading `metrics` through here is
     what lets the loop halt on a target accuracy. (Registering the final model
     after the loop is a follow-on — see config/loop.json — not in this example.)

Input ports
    labeled : json[] — current labelled rows {id, label, path}.
    pool    : json[] — current unlabeled pool rows {id, path} (label absent).
    batch   : json[] — select_confident's kept pseudo-labels {id, label, confidence}.
    model   : model-artifact / s3-ref — this round's trained model (passthrough).
    metrics : json — this round's metrics, e.g. {"accuracy": 0.71} (passthrough).

Output ports (these are the loop vars the next round re-feeds + the loop sink)
    labeled       : json[] — updated labelled set (labeled + batch).
    pool          : json[] — updated pool (pool - batch ids).
    model         : model-artifact / s3-ref — passthrough of this round's model.
    metrics       : json — passthrough of this round's metrics (drives the stop check).
    new_confident : number — count added this round (0 ⇒ plateau).
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
    labeled = inputs.get("labeled") or []
    pool = inputs.get("pool") or []
    batch = inputs.get("batch") or []
    if not isinstance(labeled, list) or not isinstance(pool, list) or not isinstance(batch, list):
        raise ValueError("merge needs list-valued `labeled`, `pool`, and `batch`.")

    new_labeled, new_pool = merge_round(labeled, pool, batch)
    return {
        "labeled": new_labeled,
        "pool": new_pool,
        "model": inputs.get("model"),       # passthrough — the loop carries the artifact forward
        "metrics": inputs.get("metrics"),   # passthrough — the stop condition reads metrics.accuracy
        "new_confident": len(batch),
    }
