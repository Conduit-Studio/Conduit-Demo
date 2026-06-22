# Model Promotion Gate — human sign-off (durable approval-wait)

A retrain produces a **candidate** model. You do **not** auto-deploy it to production — a human
reviews the evidence (candidate metric vs the current-prod baseline) and **approves or rejects**
before it goes live. On the Conduit canvas the pipeline **durably pauses** at that gate (days OK)
until the decision comes in; this example is the **local mirror** of that graph, with the EXACT
SAME code the canvas Run Code node runs.

This is the responsible-MLOps default, and the same human-gated-deploy discipline the self-training
loop needs. The new primitive it exercises is Conduit's **Wait for Approval** node (`flow.waitApproval`).

## The graph (and its local mirror)

```
Train Model → Run Code(eval_vs_baseline) → Choice(beats baseline?) → ⏸ Wait for Approval → Notify
                                                ├─ no  → Notify "underperformed"        (EXIT)
                                                └─ yes → ⏸ pause ─┬─ approve → Notify "promoted"
                                                                  ├─ reject  → Notify "rejected"
                                                                  └─ expire  → Notify "expired"
```

`scripts/run_gate_local.py` runs this end-to-end locally: a **real** scikit-learn model is trained
and evaluated (`model/candidate.py`), the **exact same** `eval_vs_baseline.main()` the Run Code node
runs compares it to the baseline, the same Choice auto-rejects an underperformer, and the human gate
is simulated (interactively, or via `--decision`). On the canvas the gate is a *durable* SFN pause
that resumes on the reviewer's click — see the build guide
(`Conduit/docs/conduit-example-workflows-build-and-mockdata.md`, §9).

## Run it locally first (real model, no faking)

```bash
cd examples/mlops/promotion-gate
uv venv .venv && uv pip install --python .venv/bin/python -r model/requirements.txt

.venv/bin/python scripts/run_gate_local.py --decision approve   # candidate beats baseline → PROMOTED
.venv/bin/python scripts/run_gate_local.py --decision reject    # → REJECTED (held)
.venv/bin/python scripts/run_gate_local.py                      # prompts for the decision

# unit tests for the pure comparison logic:
.venv/bin/python -m unittest tests/test_evaluation.py
```

The candidate is a **real** model (a LogisticRegression on the real `digits` dataset) — its accuracy
is a genuine held-out metric, not a hardcoded number. The gate is **model-agnostic**: swap
`model/candidate.py` for any real training run (e.g. the YOLO fine-tune in `../yolo-finetune`, whose
winning mAP becomes the candidate metric — set `config/gate.json` `metric` to `"mAP50-95"`).

## Map to the canvas

| Local | Canvas node | Notes |
|---|---|---|
| `train/finetune.py` `train(hyperparameters, channels)` | **Train Model** (build-from-code) | real sklearn; reads `digits.csv` from the `dataset` S3 channel. (`model/candidate.py` is the local-mirror equivalent run by `run_gate_local.py`.) |
| `train/export_digits.py` → S3 | **Config·JSON** `dataset_uri` (`s3-ref`) → Train Model `dataset` | the digits CSV uploaded to S3, wired into the Train Model's `dataset` channel |
| `config/gate.json` `baseline` | **Config·JSON** `baseline` | the current-prod metric to beat |
| `run_code/eval_vs_baseline.py` | **Run Code** `eval_vs_baseline` | inputs `metrics`,`baseline`; outputs `report`,`beats` |
| the `beats` branch | **Choice** `beats_baseline?` | rule `pass` (beats truthy) vs built-in `default` |
| the human prompt | **Wait for Approval** `promotion_gate` | the durable pause; `report` wired into its `value` |
| each print | **Notify** ×4 | promoted / rejected / expired / underperformed |

## Layout

```
train/finetune.py              # build-from-code Train Model node: train(hyperparameters, channels) -> {model, metrics}
train/export_digits.py         # exports sklearn digits -> digits.csv for upload to the `dataset` S3 channel
train/requirements.txt         # scikit-learn, numpy, joblib (installed into the built training image)
model/candidate.py             # def train_and_eval(seed) -> {metrics, modelPackageArn}  (local-mirror, built-in digits)
model/requirements.txt         # scikit-learn
run_code/eval_vs_baseline.py   # the Run Code node: main(inputs) -> {report, beats}
run_code/promotion_gate/evaluation.py   # pure, unit-tested compare logic
config/gate.json               # the Config·JSON values (baseline, metric, minDelta, timeout, notify channel)
scripts/run_gate_local.py      # local end-to-end driver (real model → eval → choice → simulated gate)
tests/test_evaluation.py
```
