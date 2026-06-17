# YOLO Fine-tune Sweep (+ eval + lineage)

Fine-tune a YOLO **bounding-box detector** on a small **real COCO** subset (person + car),
**sweep** model sizes + hyperparameters, evaluate each by mAP, and pick the winner — then
register the winning config into your SageMaker Model Registry with a reproducible lineage
record. See [example.md](example.md) for the full Conduit Deploy workflow.

This example uses Conduit's **build-from-code** Train Model node: you author a
`train(hyperparameters, channels)` function and a `requirements.txt`, and Conduit builds +
digest-pins the GPU training image for you — **no Dockerfile, no `/opt/ml` plumbing**.

## Where the data lives (two S3 locations, by purpose)

- **Canonical demo dataset:** `s3://try-conduit-app/examples/vision/YOLO-Finetuning/data/dataset/` (the YOLO family, beside the YOLO-Inference example).
- **Training channel:** the `Config·JSON` `dataset` / `sweep.json` points at a **`conduit-*`** bucket (e.g. `s3://conduit-staging-<account>/mlops/yolo-finetune/data/dataset/`) — the SageMaker execution role (`conduit-sagemaker-exec-*`) only reads `arn:aws:s3:::conduit-*`, so training **cannot** read `try-conduit-app`. Sync the dataset into a `conduit-*` bucket before deploying a sweep:
  ```bash
  aws s3 sync data/dataset/ s3://conduit-staging-<account>/mlops/yolo-finetune/data/dataset/
  ```
The example **code** stays under `examples/mlops/yolo-finetune/` (it's an MLOps sweep+lineage example, doc §8); the scripts generate the dataset locally and are path-agnostic.

## Run it locally first (real training, no faking)

The same `train()` that Conduit runs in your account also runs locally on your GPU:

```bash
cd examples/mlops/yolo-finetune
uv venv .venv && uv pip install --python .venv/bin/python "ultralytics>=8.2"
# (for an NVIDIA GPU, install a torch build matching your driver's CUDA, e.g.:
#  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchvision)

# 1. Real COCO person+car subset (80 train / 20 val):
.venv/bin/python scripts/download_coco_subset.py --limit 100
.venv/bin/python scripts/convert_coco_to_yolo_bbox.py

# 2. Run the sweep locally (subset the grid + cap epochs for a fast GPU proof):
.venv/bin/python scripts/run_sweep_local.py --trials n-640 s-640 --epochs 40 --batch 8 --device 0
```

`run_sweep_local.py` calls the same `train()` per trial, ranks the trials with the same
`select_best` the Run Code node runs, and applies the eval gate — the local mirror of
`Config·JSON → Map[Train Model] → select_best → Choice`. The full 4-trial grid in
`config/sweep.json` (incl. `s-1280` / `yolov8m`) targets a cloud GPU via the Canvas Deploy.

## Layout

```
train/finetune.py              # def train(hyperparameters, channels) -> {model, metrics}  (build-from-code)
train/requirements.txt         # ultralytics — Conduit installs these into the image it builds
config/sweep.json              # the Config·JSON sweep grid + dataset ref
run_code/select_best.py        # Run Code node: rank trials, pick the winner
run_code/yolo_finetune/        # pure, unit-tested selection logic (selection.py, schemas.py)
scripts/download_coco_subset.py    # real COCO 2017 val person+car subset
scripts/convert_coco_to_yolo_bbox.py
scripts/run_sweep_local.py     # local end-to-end sweep driver (real GPU training)
data/dataset/data.yaml         # YOLO dataset config (images/ + labels/ are generated, gitignored)
tests/test_selection.py
```
