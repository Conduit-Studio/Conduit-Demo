# Self-Training Loop (CIFAR-10, the `flow.loop` probe)

Start from a small **labelled seed** (~10% of the data) plus a large **unlabeled pool**
(~90%), then teach the model to label its own data: **train → pseudo-label the pool → keep
the high-confidence predictions → merge them into the labelled set → retrain**, round after
round, until held-out accuracy plateaus or hits a target. This is the canonical use of
Conduit's loop primitive — a graph whose body feeds itself.

Everything here is **REAL**: a torchvision **ResNet** fine-tuned on **real CIFAR-10**
photographs, with **real softmax-confidence** pseudo-labelling. The only deterministic,
unit-tested part is the keep/merge bookkeeping (`run_code/self_training/`); the training and
inference are genuine — no stubbed training, no replayed labels, no canned metrics.

## The loop, on the canvas

    Config·JSON  →  flow.loop[ train → pseudo_label → select_confident → merge ]  →  Notify

- **Config·JSON** (`config/loop.json`) holds the s3-ref data locations + the loop knobs
  (`target_acc`, `conf_threshold`, `max_rounds`).
- **flow.loop** re-feeds the body each round. Its `vars` are `labeled` (init from
  `labeled_seed`, updated by `merge.labeled`), `pool` (init from `unlabeled_pool`, updated
  by `merge.pool`), and `metrics` (updated by `merge.metrics`). It stops when
  `$.loopState.metrics.accuracy >= target_acc` or `maxRounds` is reached.
- The body is the four steps below; **`merge` is the body's sink** and carries the round's
  `model` + `metrics` forward so the loop can both halt on accuracy and hand the latest
  artifact to the next round.

> **Follow-on (not in this example):** registering the FINAL model into the SageMaker Model
> Registry *after* the loop ends is a separate slice — this example builds the loop itself.

This uses Conduit's **build-from-code** Train Model node: you author
`train(hyperparameters, channels)` + a `requirements.txt`, and Conduit builds + digest-pins
the GPU training image — **no Dockerfile, no `/opt/ml` plumbing**. The same `train()` runs
locally and in your account.

## The four loop-body steps (exact contracts)

| step | file | input → output |
|------|------|----------------|
| **train** | `train/finetune.py` | `train(hyperparameters, channels)` → `{"model": <dir>, "metrics": {"accuracy": float}}`. Channels: `labeled` (current labelled split), `test` (held-out eval). HPs: `epochs`, `batch`, `lr`, `arch` (default `resnet18`). |
| **pseudo_label** | `run_code/pseudo_label.py` | `main({"model", "pool"})` → `{"preds": [{"id", "label", "confidence"}, …]}`. Loads the model, runs REAL inference over the pool, `confidence` = softmax max-prob. |
| **select_confident** | `run_code/select_confident.py` | `main({"preds", "threshold"})` → `{"batch": [preds with conf ≥ threshold], "new_confident": <count>}`. |
| **merge** | `run_code/merge.py` | `main({"labeled", "pool", "batch", "model", "metrics"})` → `{"labeled": labeled+batch, "pool": pool−batch ids, "model": <passthrough>, "metrics": <passthrough>, "new_confident": <count>}`. |

The pure, unit-tested logic lives in `run_code/self_training/`: `select_confident(preds,
threshold)` and `merge_round(labeled, pool, batch) -> (labeled', pool')` (labelled grows,
pool shrinks by exactly the batch's ids, no duplicates). No torch in that package, so the
tests run with no GPU.

## On-disk data format

A **split** is a directory holding `index.csv` — a manifest with header
`id,image_path,label` — plus the referenced PNGs under `images/`:

```
data/labeled_seed/    index.csv  +  images/*.png    (label = CIFAR-10 class 0-9)
data/unlabeled_pool/  index.csv  +  images/*.png    (label EMPTY — the loop fills it in)
data/test/            index.csv  +  images/*.png    (label = class; held-out eval)
```

`image_path` is relative to the split dir; `label` is the integer class (empty for the
pool). This is the simplest real format — a CSV index + real PNGs — readable without torch.
`prepare_cifar_subset.py` also writes a private `data/_pool_truth.csv` (the pool's true
labels) **only** so the local driver can report pseudo-label accuracy as a sanity check; the
training/inference code never reads it and it is never uploaded.

## Run it locally first (real training + inference, no faking)

The same functions Conduit runs in your account also run locally:

```bash
cd examples/mlops/self-training
uv venv .venv && uv pip install --python .venv/bin/python torch torchvision pillow numpy
# (for an NVIDIA GPU, install a torch build matching your driver's CUDA, e.g.:
#  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchvision)

# 1. Real CIFAR-10 subset → 10% labelled seed / 90% unlabeled pool / held-out test:
.venv/bin/python scripts/prepare_cifar_subset.py --limit 2000

# 2. Run the self-training loop locally (real ResNet per round, real pseudo-labels):
.venv/bin/python scripts/run_loop_local.py --rounds 5 --epochs 3 --threshold 0.95 --target 0.80 --device cuda
```

`run_loop_local.py` calls the EXACT SAME `train()`, `pseudo_label`, `select_confident`, and
`merge` the deployed graph runs, prints per-round accuracy + labelled/pool sizes +
`new_confident`, and stops on target / plateau / maxRounds — the local mirror of
`Config → flow.loop[…] → Notify`. Use a smaller `--limit` for a faster proof.

Run the pure-logic tests (no torch, no GPU):

```bash
python -m unittest tests/test_selection.py
```

## The honest ML caveat

Self-training risks **confirmation bias**: a confident-but-wrong pseudo-label gets baked
into the labelled set and the model reinforces its own mistake. The guards here are
**(1) a high confidence threshold** (`conf_threshold` — only near-certain predictions are
accepted) and **(2) evaluating on a held-out `test` set every round** (so accuracy is
measured against real labels the model never trained on, and the loop stops if it isn't
genuinely improving). The local driver additionally reports the accepted pseudo-labels'
accuracy against the pool's hidden truth as a sanity check.

## Layout

```
train/finetune.py                 # def train(hyperparameters, channels) -> {model, metrics}  (build-from-code, REAL ResNet)
train/requirements.txt            # torch/torchvision/pillow/numpy — Conduit installs into the image it builds
config/loop.json                  # Config·JSON values + the documented flow.loop config (vars + stop)
run_code/pseudo_label.py          # Run Code: REAL inference over the pool → {preds}
run_code/select_confident.py      # Run Code: keep conf >= threshold → {batch, new_confident}
run_code/merge.py                 # Run Code: round summary — grow labelled / shrink pool / passthrough model+metrics
run_code/self_training/           # pure, unit-tested logic (selection.py, merge.py, schemas.py) — no torch
run_code/requirements.txt
scripts/prepare_cifar_subset.py   # real CIFAR-10 → labeled_seed / unlabeled_pool / test
scripts/run_loop_local.py         # local end-to-end loop driver (real training + inference each round)
tests/test_selection.py           # unit tests for the pure select/merge logic
```
