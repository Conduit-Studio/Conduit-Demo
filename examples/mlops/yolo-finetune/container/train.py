#!/usr/bin/env python3
"""SageMaker training entrypoint: fine-tune a YOLO detector and report mAP.

Runs INSIDE the BYO training container (see Dockerfile). This is the image the
Conduit **Train Model** node points at (pinned by `@sha256:` digest). SageMaker's
BYO-training contract:

  * hyperparameters arrive as JSON at /opt/ml/input/config/hyperparameters.json
    (every value is a STRING — coerce);
  * the `dataset` input channel is mounted at /opt/ml/input/data/dataset
    (SageMaker also exports SM_CHANNEL_DATASET);
  * write the model artifact to /opt/ml/model — SageMaker tars it into model.tar.gz
    (which becomes the Train Model node's `model` output);
  * print metrics so a MetricDefinitions regex can scrape them into the job's
    FinalMetricDataList → the node's `metrics` output → Conduit's lineage record.

Conduit's Train Model node passes the per-trial hyperparameters ({weights, imgsz,
lr0, epochs}) and the dataset channel; the auto-injected capture step then
fingerprints the dataset + the job's ACTUAL hyperparameters into the model's
lineage record. Nothing here talks to Conduit — it's a plain SageMaker job.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HP_PATH = Path("/opt/ml/input/config/hyperparameters.json")
DATASET_DIR = Path(os.environ.get("SM_CHANNEL_DATASET", "/opt/ml/input/data/dataset"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DIR = Path("/opt/ml/output/data")


def _num(value: object, cast):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def read_hyperparameters() -> dict:
    """Read + coerce SageMaker hyperparameters (all delivered as strings)."""
    raw = json.loads(HP_PATH.read_text()) if HP_PATH.exists() else {}
    return {
        "weights": str(raw.get("weights", "yolov8n.pt")),
        "imgsz": _num(raw.get("imgsz", 640), int) or 640,
        "lr0": _num(raw.get("lr0", 0.01), float) or 0.01,
        "epochs": _num(raw.get("epochs", 50), int) or 50,
        "batch": _num(raw.get("batch", 16), int) or 16,
    }


def find_data_yaml(dataset_dir: Path) -> Path:
    """Locate the YOLO dataset config inside the mounted channel."""
    for name in ("data.yaml", "dataset.yaml"):
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(dataset_dir.rglob("*.yaml"))
    if not matches:
        raise FileNotFoundError(f"no data.yaml found under {dataset_dir}")
    return matches[0]


def main() -> int:
    hp = read_hyperparameters()
    data_yaml = find_data_yaml(DATASET_DIR)
    print(
        f"[train] weights={hp['weights']} imgsz={hp['imgsz']} lr0={hp['lr0']} "
        f"epochs={hp['epochs']} batch={hp['batch']} data={data_yaml}",
        flush=True,
    )

    # Heavy import kept inside main() so `python train.py --help` stays cheap.
    from ultralytics import YOLO

    model = YOLO(hp["weights"])
    model.train(
        data=str(data_yaml),
        imgsz=hp["imgsz"],
        lr0=hp["lr0"],
        epochs=hp["epochs"],
        batch=hp["batch"],
        project=str(OUTPUT_DIR),
        name="finetune",
        exist_ok=True,
        verbose=True,
    )

    metrics = model.val(data=str(data_yaml), imgsz=hp["imgsz"], verbose=True)
    # ultralytics DetMetrics: box.map = mAP@0.5:0.95, box.map50 = mAP@0.5.
    map50_95 = float(getattr(metrics.box, "map", 0.0))
    map50 = float(getattr(metrics.box, "map50", 0.0))

    # Emit on their own lines so SageMaker MetricDefinitions can scrape them, e.g.
    #   { "Name": "mAP50-95", "Regex": "mAP50-95: ([0-9.]+)" }
    print(f"mAP50-95: {map50_95:.5f}", flush=True)
    print(f"mAP50: {map50:.5f}", flush=True)

    # Persist the best weights as the model artifact; SageMaker tars /opt/ml/model.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best = OUTPUT_DIR / "finetune" / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, MODEL_DIR / "best.pt")
    else:
        print(f"[train] WARNING: expected weights at {best} not found", flush=True)
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps({"mAP50-95": map50_95, "mAP50": map50, "hyperparameters": hp}, indent=2)
    )
    print("[train] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
