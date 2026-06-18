"""Run the model-promotion gate locally, end-to-end — the local mirror of the deployed Conduit graph:

    Train Model → Run Code(eval_vs_baseline) → Choice(beats baseline?) → ⏸ Wait for Approval → Notify

It calls the REAL `train_and_eval()` (a real scikit-learn model) and the EXACT SAME
`eval_vs_baseline.main()` the Run Code node runs, applies the same Choice, and then simulates the
human approval gate. On the canvas the gate is a DURABLE pause (the SFN execution suspends for up
to `timeoutSeconds`); locally we simulate the reviewer's decision interactively, or via --decision.

Run from examples/mlops/promotion-gate:

    uv venv .venv && uv pip install --python .venv/bin/python -r model/requirements.txt
    .venv/bin/python scripts/run_gate_local.py --decision approve     # → PROMOTED
    .venv/bin/python scripts/run_gate_local.py --decision reject      # → REJECTED
    .venv/bin/python scripts/run_gate_local.py                        # → prompts for the decision
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
# Import the demo's real code: the candidate model + the Run Code node's eval.
sys.path.insert(0, str(EXAMPLE_DIR / "model"))
sys.path.insert(0, str(EXAMPLE_DIR / "run_code"))

from candidate import train_and_eval  # noqa: E402  (model/candidate.py)
from eval_vs_baseline import main as eval_main  # noqa: E402  (run_code/eval_vs_baseline.py)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", choices=["approve", "reject", "expire"],
                    help="simulate the reviewer's decision (default: prompt interactively)")
    ap.add_argument("--seed", type=int, default=0, help="candidate training seed")
    args = ap.parse_args()
    cfg = json.loads((EXAMPLE_DIR / "config" / "gate.json").read_text(encoding="utf-8"))
    metric = cfg["metric"]

    # ---- Train Model: a real candidate + its real metric ----
    candidate = train_and_eval(seed=args.seed)
    print(f"\n[Train Model]  candidate metrics: {candidate['metrics']}  ({candidate['modelPackageArn']})")

    # ---- Run Code: eval_vs_baseline (the EXACT node code) ----
    out = eval_main({
        "metrics": candidate["metrics"],
        "baseline": cfg["baseline"],
        "metric": metric,
        "minDelta": cfg.get("minDelta", 0.0),
        "modelPackageArn": candidate["modelPackageArn"],
    })
    r = out["report"]
    print(f"[Run Code · eval_vs_baseline]  candidate {r['candidate']} vs baseline {r['baseline']} "
          f"→ delta {r['delta']:+}, beats={r['beats']}")

    # ---- Choice: beats baseline? ----
    if not out["beats"]:
        print(f"[Choice]  default → [Notify]  underperformed ({r['candidate']} < {r['baseline']} + minDelta {r['minDelta']}) — not promoting.\n")
        return 0
    print(f"[Choice]  pass → entering the approval gate (durable on the canvas; timeout {cfg['timeoutSeconds']}s)…")

    # ---- Wait for Approval: the human gate (durable pause on the canvas; simulated here) ----
    decision = args.decision or input("[Wait for Approval]  Approve promotion? [approve/reject/expire]: ").strip().lower()
    if decision == "approve":
        print(f"[approved → Notify]  🚀 PROMOTED — {candidate['modelPackageArn']} → Approved.\n")
    elif decision == "reject":
        print("[rejected → Notify]  🛑 REJECTED — held; reason logged.\n")
    elif decision == "expire":
        print("[expired → Notify]  ⏰ EXPIRED — no decision within the window; escalating.\n")
    else:
        print(f"unknown decision {decision!r} (expected approve/reject/expire)\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
