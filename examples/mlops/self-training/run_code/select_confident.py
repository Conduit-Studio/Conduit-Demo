"""Conduit Run Code step: keep this round's high-confidence pseudo-labels.

Drop this on the canvas as a Run Code node (Container/CPU — it's pure Python, no GPU).
It runs INSIDE the loop body, right after pseudo_label:

    flow.loop[ train  ->  pseudo_label  ->  select_confident  ->  merge ]
                                            ^^^^^^^^^^^^^^^^^^
The pseudo-label step emits one prediction per pooled example; select_confident keeps
only those whose softmax confidence clears `threshold` — these become the new labelled
batch that `merge` folds into the labelled set for the next round. A high bar is the
guard against confirmation bias (a confident-but-wrong pseudo-label poisons retraining),
so the threshold is a tunable loop input, not a constant.

Input ports
    preds     : json[] — predictions from pseudo_label, each {id, label, confidence}.
    threshold : (optional) min softmax max-prob to accept; defaults to 0.95. May be a string.

Output ports
    batch         : json[] — the kept predictions {id, label, confidence}, conf-desc sorted.
    new_confident : number — how many were kept this round (0 ⇒ the loop has plateaued).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the sibling package importable when Conduit runs this file as the entrypoint.
_RUN_CODE_DIR = Path(__file__).resolve().parent
if str(_RUN_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_CODE_DIR))

from self_training.selection import DEFAULT_THRESHOLD, select_confident  # noqa: E402


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    preds = inputs.get("preds")
    if not isinstance(preds, list):
        raise ValueError("select_confident needs `preds`: a list of {id, label, confidence} predictions.")

    raw_threshold = inputs.get("threshold")
    threshold = DEFAULT_THRESHOLD if raw_threshold is None else float(raw_threshold)

    batch = select_confident(preds, threshold)
    return {"batch": batch, "new_confident": len(batch)}
