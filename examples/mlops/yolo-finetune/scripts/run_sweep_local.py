"""Run the YOLO fine-tune sweep locally, end-to-end, with REAL training — no faking.

This is the local mirror of the deployed Conduit graph:

    Config·JSON(sweep)  →  Map[Train Model · build-from-code]  →  select_best  →  Choice(eval_gate)

Each trial calls the EXACT SAME `train(hyperparameters, channels)` from `train/finetune.py`
that Conduit's build-from-code Train Model node runs in your account (just on the local GPU
instead of a SageMaker job), then ranks the trials with the EXACT SAME `select_best` /
`yolo_finetune.selection` code the Run Code node runs, then applies the eval gate.

Run from examples/mlops/yolo-finetune (after downloading the dataset):

    python scripts/download_coco_subset.py --limit 100
    python scripts/convert_coco_to_yolo_bbox.py
    python scripts/run_sweep_local.py --trials n-640 s-640 --epochs 40 --batch 8

The full 4-trial grid in config/sweep.json (incl. s-1280 / yolov8m) targets a real cloud GPU
via the Canvas Deploy; for a fast local proof, subset the trials and cap epochs/imgsz/batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
# Import the demo's real code: train() from train/, selection from run_code/.
sys.path.insert(0, str(EXAMPLE_DIR / "train"))
sys.path.insert(0, str(EXAMPLE_DIR / "run_code"))

from finetune import train  # noqa: E402  (train/finetune.py)
from yolo_finetune.selection import DEFAULT_METRIC, rank, select_best  # noqa: E402


def load_sweep(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec.get("trials"), list) or not spec["trials"]:
        raise SystemExit(f"{path} has no `trials` array")
    return spec


def pick_trials(spec: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    trials = spec["trials"]
    if not names:
        return trials
    by_name = {str(t["name"]): t for t in trials}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"--trials not in sweep: {missing}; available: {list(by_name)}")
    return [by_name[n] for n in names]


def run_trial(trial: dict[str, Any], dataset_dir: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Train one trial and shape its result like the Map body's per-item output."""
    hp = {**trial, **{k: v for k, v in overrides.items() if v is not None}}
    print(f"\n===== trial {trial['name']}  hp={hp} =====", flush=True)
    out = train(hp, {"dataset": str(dataset_dir)})
    # Locally the model is a directory on disk; in the deployed graph it's an s3-ref
    # {bucket,key}. select_best treats `model` as opaque, so we carry the local path.
    return {
        "name": trial["name"],
        "model": {"path": out["model"]},
        "metrics": out["metrics"],
        "hyperparameters": hp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the YOLO fine-tune sweep locally with real training.")
    parser.add_argument("--sweep", default="config/sweep.json", help="path to the sweep grid")
    parser.add_argument("--dataset", default="data/dataset", help="dataset dir holding data.yaml")
    parser.add_argument("--trials", nargs="*", default=None, help="subset of trial names to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for every trial")
    parser.add_argument("--imgsz", type=int, default=None, help="override imgsz for every trial")
    parser.add_argument("--batch", type=int, default=None, help="override batch for every trial (-1 = AutoBatch)")
    parser.add_argument("--workers", type=int, default=None, help="dataloader workers (lower on small-RAM hosts)")
    parser.add_argument("--device", default=None, help="ultralytics device, e.g. 0 (GPU) or cpu")
    parser.add_argument("--baseline", type=float, default=0.25, help="eval-gate threshold on the ranking metric")
    args = parser.parse_args()

    sweep_path = (EXAMPLE_DIR / args.sweep) if not Path(args.sweep).is_absolute() else Path(args.sweep)
    dataset_dir = (EXAMPLE_DIR / args.dataset) if not Path(args.dataset).is_absolute() else Path(args.dataset)
    if not (dataset_dir / "data.yaml").exists():
        raise SystemExit(f"no data.yaml under {dataset_dir} — run download + convert first")

    spec = load_sweep(sweep_path)
    metric = str(spec.get("metric") or DEFAULT_METRIC)
    trials = pick_trials(spec, args.trials)
    overrides = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
    }

    print(f"Sweep: {len(trials)} trial(s) → {[t['name'] for t in trials]}; metric={metric}; baseline={args.baseline}")

    results = [run_trial(trial, dataset_dir, overrides) for trial in trials]

    # ---- select_best + eval gate (the deployed graph's Run Code + Choice) ----
    best = select_best(results, metric)
    ranked = rank([_as_trial_result(r) for r in results], metric)

    print("\n================ SWEEP RESULTS ================")
    for r in ranked:
        print(f"  {r.name:<10} {metric}={r.metrics.get(metric):.4f}  mAP50={r.metrics.get('mAP50', float('nan')):.4f}")
    print(f"\nWINNER: {best.name}  {metric}={best.value:.4f}  hp={best.hyperparameters}")
    gate_pass = best.value >= args.baseline
    verdict = "PASS → promote" if gate_pass else "HOLD → below baseline"
    print(f"EVAL GATE: {best.value:.4f} >= baseline {args.baseline}?  {gate_pass}  ({verdict})")

    summary = {
        "metric": metric,
        "baseline": args.baseline,
        "gate_pass": gate_pass,
        "best": best.as_dict(),
        "ranking": [{"name": r.name, metric: r.metrics.get(metric), "mAP50": r.metrics.get("mAP50")} for r in ranked],
        "trials": results,
    }
    out_path = EXAMPLE_DIR / "runs" / "sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(EXAMPLE_DIR)}")
    return 0


def _as_trial_result(item: dict[str, Any]):
    from yolo_finetune.schemas import TrialResult

    return TrialResult.from_item(item)


if __name__ == "__main__":
    raise SystemExit(main())
