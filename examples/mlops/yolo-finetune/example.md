# YOLO Fine-tune Sweep, Eval Gate & Lineage

This demo is a **Deploy** Conduit workflow: it fine-tunes several YOLO object
detectors on your own bounding-box dataset, sweeps model sizes + hyperparameters in
parallel, evaluates each by **mAP**, picks the winner, and — because every top-level
`Train Model` registers automatically — leaves a reproducible **lineage record** in
your SageMaker Model Registry (dataset fingerprint + the run's actual hyperparameters
+ the digest-pinned training image), sitting at `PendingManualApproval`.

Unlike the Run-only examples, training is long-running, so this graph is compiled to a
**Step Functions** state machine and **Deployed** (not Run synchronously). The sweep is
an inline **Map** — one SageMaker training job per trial, fanned out concurrently.

It ships a tiny **synthetic** 2-class dataset (circle / rectangle) so the whole thing
runs end-to-end on a laptop's worth of data. Replace it with your real images + YOLO
labels for a real run.

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
3. **Train Model · GPU** (inside the Map) runs each trial as a SageMaker training job
   against your BYO ultralytics container (`container/`), reading the trial's
   hyperparameters and the shared dataset channel, emitting `model` (the artifact) and
   `metrics` (scraped `mAP50-95` / `mAP50`).
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
  container/                    # the BYO SageMaker training image (the ONE container Conduit doesn't build)
    Dockerfile                  #   ultralytics on a CUDA PyTorch base
    train.py                    #   SageMaker entrypoint: HP + dataset channel -> train + val -> model.tar.gz + mAP
    requirements.txt
  run_code/                     # the select_best Conduit Run Code node (Container/CPU, pure stdlib)
    select_best.py              #   main(inputs): rank the sweep, pick the winner
    requirements.txt
    yolo_finetune/
      __init__.py
      schemas.py                #   TrialResult / Best
      selection.py              #   pure rank/select logic (unit-tested)
  scripts/
    make_demo_dataset.py        # generate the synthetic 2-class bbox dataset under data/dataset/
    build_and_push.sh           # build + push the training image to ECR -> prints the @sha256 to pin
  data/dataset/
    data.yaml                   # YOLO dataset config (committed; the generator fills the rest)
    images/{train,val}/         # generated (gitignored)
    labels/{train,val}/         # generated (gitignored; YOLO format: "class cx cy w h")
  tests/
    test_selection.py
```

## Build it (end to end)

**1. Dataset.** Generate the synthetic set and upload it to the S3 ref in `sweep.json`:

```bash
uv run python scripts/make_demo_dataset.py
aws s3 sync data/dataset/ s3://<your-bucket>/examples/mlops/yolo-finetune/data/dataset/
```

(For a real run, replace `data/dataset/` with your images + YOLO labels and keep the
same `data.yaml` shape.)

**2. Training image.** Build + push the BYO container and copy the printed digest:

```bash
AWS_REGION=us-east-1 bash scripts/build_and_push.sh
# ✅ Pin THIS in the Train Model node:  <acct>.dkr.ecr.us-east-1.amazonaws.com/conduit-yolo@sha256:...
```

Pin the **`@sha256:` digest**, never a `:latest` tag — Conduit's lineage record pins the
image by digest so a registered model is genuinely reproducible.

**3. Canvas.** Wire up: `Config · JSON (sweep)` → `Map` → `Train Model · GPU` →
`Run Code (select_best)` → `Choice (eval_gate)` → `Notify`.

- **Train Model** node: set the image to the pinned digest; map the trial fields to
  hyperparameters (`weights`, `imgsz`, `lr0`, `epochs`); add a metric definition
  `mAP50-95` with regex `mAP50-95: ([0-9.]+)` (train.py prints exactly that line); point
  the `dataset` input channel at the sweep's `dataset` ref. Pick a GPU instance
  (e.g. `ml.g5.xlarge`).
- **select_best** node: wire the Map's output into its `results` input.
- **eval_gate** (Choice): route on `best.value >= 0.5` (tune the threshold).

**4. Deploy.** Click **Deploy** — the graph compiles to a Step Functions sweep (an
inline Map of training jobs). Each trial trains on its own scoped IAM; the Map runs
them concurrently (cap to your GPU quota).

**5. Lineage.** After a deployed execution, read the registered model(s):

```bash
curl -s https://<your-conduit-host>/api/workflows/<workflowId>/lineage | jq
# -> [{ modelPackageArn, approvalStatus: "PendingManualApproval",
#       datasetFingerprint, paramHash, trainingImage: "...@sha256:...",
#       modelArtifactUri, trainingJobName, ... }]
```

Approve the winner in the SageMaker console (or via the API) to flip
`ApprovalStatus` → that's the hook the eval gate / a downstream deploy reads.

## Local smoke test

No AWS needed for the selection logic or the dataset generator:

```bash
uv run python -m unittest examples/mlops/yolo-finetune/tests/test_selection.py
uv run python examples/mlops/yolo-finetune/scripts/make_demo_dataset.py --train 20 --val 4
```

## Scope notes (what's Foundation vs. fast-follow)

- **Lineage on a top-level `Train Model` is live today** — the compiler injects
  capture → register after each top-level training Task.
- **Per-trial lineage inside the Map** (registering all four trials, not just a
  top-level train) is the Map-aware fast-follow. Until it lands, register the
  **winning** config as a second, top-level `Train Model` to get its lineage record.
- **Auto-promotion** (the eval gate flipping `ApprovalStatus` automatically) is the
  eval-gate slice; today the gate routes/notifies and approval is a human action.
