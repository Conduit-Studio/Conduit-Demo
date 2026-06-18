# YOLO Fine-tune Sweep, Eval Gate & Lineage

This demo is a **Deploy** Conduit workflow: it fine-tunes several YOLO object
detectors on your own bounding-box dataset, sweeps model sizes + hyperparameters in
parallel, evaluates each by **mAP**, picks the winner, and — because every top-level
`Train Model` registers automatically — leaves a reproducible **lineage record** in
your SageMaker Model Registry (dataset fingerprint + the run's actual hyperparameters
+ the digest-pinned training image), sitting at `PendingManualApproval`.

It uses Conduit's **build-from-code** Train Model node: you author a
`train(hyperparameters, channels)` function and a `requirements.txt`, and Conduit
builds + digest-pins the GPU SageMaker training image for you — **no Dockerfile, no
`/opt/ml` plumbing, no `build_and_push.sh`**. The same `train()` runs locally on your
GPU and in your AWS account.

Unlike the Run-only examples, training is long-running, so this graph is compiled to a
**Step Functions** state machine and **Deployed** (not Run synchronously). The sweep is
an inline **Map** — one SageMaker training job per trial, fanned out concurrently.

The dataset is a small **real COCO** subset — `person` + `car` boxes — downloaded by a
script (no synthetic data). Swap in your own images + YOLO labels for a real run.

## Problem

A team has a labelled bounding-box dataset and wants the best YOLO detector they can
get — but "best" depends on model size, image size, and learning rate, and they need
to be able to answer *six months later* exactly which data + which hyperparameters
produced the model now serving traffic. Hand-running `yolo train` four times in a
notebook gives you neither the parallelism nor the provenance.

## Solution

Build the sweep once on the Conduit canvas and Deploy it:

1. **Config · JSON** holds the sweep grid (`config/sweep.json`): four trials over
   yolov8n/s/m × imgsz/lr/epochs, plus the shared `dataset` S3 ref.
2. **Map** fans the grid out — one branch per trial.
3. **Train Model · GPU** (inside the Map, **build-from-code**) runs each trial as a
   SageMaker training job. You author `train(hyperparameters, channels)` in
   `train/finetune.py`; Conduit builds the ultralytics GPU image, runs your function
   with the trial's hyperparameters + the shared `dataset` channel, copies the model
   dir you return into `model.tar.gz`, and scrapes the `metrics` you return
   (`mAP50-95` / `mAP50`) into the job's metrics → the node's `metrics` output.
4. **Run Code · `select_best`** ranks the Map's results by mAP and picks the winner.
5. **Choice · `eval_gate`** routes on `best.value` — promote only if it clears a mAP
   threshold, otherwise notify and stop.
6. **Lineage** — each top-level `Train Model` already registers its model with a full
   lineage record; you read it back via `GET /api/workflows/:id/lineage` (or approve it
   in the SageMaker console to flip `ApprovalStatus`, which is the eval gate's seam).

All training logs and failures stay in the Output panel; the registry is the durable
record of what was produced.

## Layout

```text
examples/mlops/yolo-finetune/
  example.md
  README.md
  config/
    sweep.json                  # the Config·JSON node's value: trials grid + dataset ref + metric
  train/                        # build-from-code training (NO Dockerfile — Conduit builds the image)
    finetune.py                 #   def train(hyperparameters, channels) -> {model, metrics}
    requirements.txt            #   ultralytics — Conduit installs these into the image it builds
  run_code/                     # the select_best Conduit Run Code node (Container/CPU, pure stdlib)
    select_best.py              #   main(inputs): rank the sweep, pick the winner
    requirements.txt
    yolo_finetune/
      __init__.py
      schemas.py                #   TrialResult / Best
      selection.py              #   pure rank/select logic (unit-tested)
  scripts/
    download_coco_subset.py     # download a real COCO person+car subset (80 train / 20 val)
    convert_coco_to_yolo_bbox.py# COCO boxes -> YOLO labels + data.yaml
    run_sweep_local.py          # run the whole sweep locally with REAL GPU training
  data/dataset/
    data.yaml                   # YOLO dataset config (committed; images/ + labels/ are generated)
    images/{train,val}/         # generated (gitignored)
    labels/{train,val}/         # generated (gitignored; YOLO format: "class cx cy w h")
  tests/
    test_selection.py           # pure selection logic
    test_finetune.py            # finetune helpers (coercion / run-name / data.yaml discovery)
```

## Run it locally first (real training, no faking)

The exact `train()` Conduit runs in your account also runs locally on your GPU. This is
how the example was validated before any Deploy:

```bash
cd examples/mlops/yolo-finetune
uv venv .venv && uv pip install --python .venv/bin/python "ultralytics>=8.2"
# NVIDIA GPU: install a torch build matching your driver's CUDA (e.g. cu128):
#   uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 1. Real COCO person+car subset (80 train / 20 val):
.venv/bin/python scripts/download_coco_subset.py --limit 100
.venv/bin/python scripts/convert_coco_to_yolo_bbox.py

# 2. Run the sweep locally — subset the grid + cap epochs/workers for a fast GPU proof:
.venv/bin/python scripts/run_sweep_local.py --trials n-640 s-640 --epochs 40 --batch 8 --workers 4 --device 0
```

`run_sweep_local.py` calls the same `train()` per trial, ranks with the same
`select_best` the Run Code node runs, and applies the eval gate — the local mirror of
`Config·JSON → Map[Train Model] → select_best → Choice`.

**Verified local run** (RTX 3060, 80 train / 20 val real COCO images, 40 epochs/trial):

| trial | model    | mAP50-95 | mAP50 |
|-------|----------|----------|-------|
| n-640 | yolov8n  | 0.306    | 0.485 |
| **s-640** | **yolov8s** | **0.500** | **0.687** |

→ **winner `s-640`**, eval gate **PASS → promote** (0.500 ≥ 0.10 baseline). The full
4-trial grid in `config/sweep.json` (incl. `s-1280` / `yolov8m`) targets a cloud GPU via
the Canvas Deploy; locally we subset to what fits a laptop GPU.

## Build it in Conduit (the Deploy workflow)

Wire up: `Config · JSON (sweep)` → `Map` → `Train Model · GPU` →
`Run Code (select_best)` → `Choice (eval_gate)` → `Notify`.

- **Config · JSON (sweep)** — paste `config/sweep.json` (the trials grid + the shared
  `dataset` S3 ref). Output: `spec`.
- **Map (sweep_finetune)** — fan out over `spec.trials`; shared input `spec.dataset`.
  Each iteration gets one trial (via `items`) + the same `dataset`. Body = Train Model.
- **Train Model · GPU (finetune_yolo)** — **Image source: Build from code.** Paste
  `train/finetune.py` (the `train(hyperparameters, channels)` function) and
  `train/requirements.txt` (`ultralytics>=8.2`). Input channel `dataset` ← the shared
  `spec.dataset`; HyperParameters ← the per-trial item (`weights`, `imgsz`, `lr0`,
  `epochs`); declared metric `mAP50-95`. Pick a GPU tier (e.g. `ml.g5.xlarge`). On
  deploy Conduit builds the image (CodeBuild → ECR, content-addressed) and pins it by
  `@sha256:` digest — the same digest the lineage record records.
- **Run Code (select_best)** — wire **`Map.results`** into `select_best`'s single
  `results` input — one self-describing `{item, index, model, metrics}` record per trial;
  `select_best` ranks them and returns the winner.
- **Choice (eval_gate)** — route on `best.value >= <baseline>`: pass → Notify (promote),
  default → Notify (hold).

**Deploy.** Click **Deploy** — the graph compiles to a Step Functions sweep (an inline
Map of training jobs). Each trial trains on its own scoped IAM; the Map runs them
concurrently (cap to your GPU quota).

**Lineage.** After a deployed execution, read the registered model(s):

```bash
curl -s https://<your-conduit-host>/api/workflows/<workflowId>/lineage | jq
# -> [{ modelPackageArn, approvalStatus: "PendingManualApproval",
#       datasetFingerprint, paramHash, trainingImage: "...conduit-images@sha256:...",
#       modelArtifactUri, trainingJobName, ... }]
```

Approve the winner in the SageMaker console (or via the API) to flip `ApprovalStatus` —
that's the hook the eval gate / a downstream deploy reads.

## Local smoke test (no AWS, no GPU)

The selection + helper logic is pure and unit-tested:

```bash
cd examples/mlops/yolo-finetune
.venv/bin/python tests/test_selection.py
.venv/bin/python tests/test_finetune.py
```

## Scope notes (what's Foundation vs. fast-follow)

- **Lineage on a top-level `Train Model` is live today** — the compiler injects
  capture → register after each top-level training Task.
- **Per-trial lineage inside the Map** (registering all four trials, not just a
  top-level train) is the Map-aware fast-follow. Until it lands, register the
  **winning** config as a second, top-level `Train Model` to get its lineage record.
- **Auto-promotion** (the eval gate flipping `ApprovalStatus` automatically) is the
  eval-gate slice; today the gate routes/notifies and approval is a human action.
