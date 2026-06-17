"""Build-from-code training function for the YOLO bbox fine-tune.

This is the WHOLE training contract under Conduit's **build-from-code** Train Model
node: you author `train(hyperparameters, channels)` and a `requirements.txt`, and
Conduit builds + digest-pins the GPU SageMaker image for you (curated AWS PyTorch
GPU Deep Learning Container base + your requirements + this code + Conduit's wrapper).
There is **no Dockerfile and no `/opt/ml` plumbing** — Conduit's wrapper:

  * hands `train()` the job's hyperparameters (the per-trial item from the sweep Map)
    and the local channel paths (e.g. the `dataset` channel mounted from S3);
  * copies the model directory you return into `/opt/ml/model` (→ `model.tar.gz`,
    which becomes the Train Model node's `model` output);
  * scrapes the `metrics` dict you return into the job's metrics → the node's
    `metrics` output → Conduit's lineage record.

So nothing here talks to AWS or SageMaker. The exact same `train()` runs:
  * locally, called by `scripts/run_sweep_local.py` (real GPU fine-tune, no faking);
  * in your account, inside the image Conduit builds for the Train Model node.

SageMaker delivers every hyperparameter as a STRING, so we coerce. `channels["dataset"]`
is a local directory holding the YOLO dataset (`images/`, `labels/`, `data.yaml`).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _coerce(value: Any, cast, default):
    """Best-effort cast (SageMaker hyperparameters all arrive as strings)."""
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def find_data_yaml(dataset_dir: Path) -> Path:
    """Locate the YOLO dataset config inside the mounted `dataset` channel."""
    for name in ("data.yaml", "dataset.yaml"):
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(dataset_dir.rglob("*.yaml"))
    if not matches:
        raise FileNotFoundError(f"no data.yaml found under {dataset_dir}")
    return matches[0]


def _run_name(hp: dict[str, Any]) -> str:
    """A unique, deterministic run-dir name so concurrent/looped trials never collide.

    Prefer the sweep trial's `name` (e.g. "s-1280"); otherwise derive from the config.
    """
    name = hp.get("name")
    if name:
        return str(name)
    weights = str(hp.get("weights", "yolo")).removesuffix(".pt")
    return f"{weights}-{hp.get('imgsz', 640)}"


def train(hyperparameters: dict[str, Any], channels: dict[str, str]) -> dict[str, Any]:
    """Fine-tune a YOLO detector on the `dataset` channel and report real mAP.

    Args:
        hyperparameters: the per-trial config — keys `weights`, `imgsz`, `lr0`,
            `epochs` (+ optional `batch`, `device`, `name`). Values may be strings.
        channels: maps channel name → local path. `channels["dataset"]` is the YOLO
            dataset directory (contains `data.yaml`).

    Returns:
        {"model": <weights dir path>, "metrics": {"mAP50-95": float, "mAP50": float}}
        `model` is the directory Conduit copies into `/opt/ml/model` (it holds
        `best.pt`/`last.pt`); `metrics` carries the real validation mAP.
    """
    weights = str(hyperparameters.get("weights", "yolov8n.pt"))
    imgsz = _coerce(hyperparameters.get("imgsz", 640), int, 640)
    lr0 = _coerce(hyperparameters.get("lr0", 0.01), float, 0.01)
    epochs = _coerce(hyperparameters.get("epochs", 50), int, 50)
    batch = _coerce(hyperparameters.get("batch", 16), int, 16)  # -1 = ultralytics AutoBatch
    workers = _coerce(hyperparameters.get("workers", 8), int, 8)  # dataloader workers (lower on small-RAM hosts)
    device = hyperparameters.get("device")  # None → ultralytics auto-selects GPU if present

    dataset_dir = Path(channels.get("dataset") or next(iter(channels.values()))).resolve()
    data_yaml = find_data_yaml(dataset_dir)

    # Deterministic output location; one sub-dir per trial so a looped sweep doesn't clobber.
    runs_dir = Path(os.environ.get("CONDUIT_RUNS_DIR", "runs/detect")).resolve()
    run_name = _run_name(hyperparameters)

    print(
        f"[train] weights={weights} imgsz={imgsz} lr0={lr0} epochs={epochs} batch={batch} "
        f"workers={workers} device={device or 'auto'} data={data_yaml} run={run_name}",
        flush=True,
    )

    # Heavy import kept inside the function so importing this module (tests, the wrapper
    # discovering `train`) stays cheap and torch-free.
    from ultralytics import YOLO

    model = YOLO(weights)
    train_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        imgsz=imgsz,
        lr0=lr0,
        epochs=epochs,
        batch=batch,
        workers=workers,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
    if device is not None:
        train_kwargs["device"] = device

    # model.train() runs a final validation on best.pt and RETURNS those metrics
    # (ultralytics DetMetrics: box.map = mAP@0.5:0.95, box.map50 = mAP@0.5). Read them
    # directly — calling model.val() again is redundant and, on a small-RAM host, the
    # second val dataloader can OOM-kill the process.
    results = model.train(**train_kwargs)
    map50_95 = float(getattr(results.box, "map", 0.0))
    map50 = float(getattr(results.box, "map50", 0.0))

    weights_dir = runs_dir / run_name / "weights"
    print(f"mAP50-95: {map50_95:.5f}", flush=True)
    print(f"mAP50: {map50:.5f}", flush=True)
    print(f"[train] weights at {weights_dir}", flush=True)

    return {
        "model": str(weights_dir),
        "metrics": {"mAP50-95": map50_95, "mAP50": map50},
    }


def _main() -> int:
    """Run a single fine-tune locally for a quick smoke (mirrors one sweep trial)."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune one YOLO trial locally.")
    parser.add_argument("--dataset", required=True, help="dataset dir holding data.yaml")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default=None, help="run name (defaults to derived)")
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU")
    args = parser.parse_args()

    out = train(
        {
            "weights": args.weights,
            "imgsz": args.imgsz,
            "lr0": args.lr0,
            "epochs": args.epochs,
            "batch": args.batch,
            "workers": args.workers,
            "name": args.name,
            "device": args.device,
        },
        {"dataset": args.dataset},
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
