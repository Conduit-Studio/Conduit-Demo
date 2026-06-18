"""Run the self-training loop locally, end-to-end, with REAL training + inference — no faking.

This is the local mirror of the deployed Conduit graph:

    Config·JSON  →  flow.loop[ train → pseudo_label → select_confident → merge ]  →  Notify

Each round calls the EXACT SAME functions Conduit runs in your account:
  * train()                  from train/finetune.py        — real ResNet fine-tune, real eval
  * pseudo_label.main()      from run_code/pseudo_label.py  — real inference over the pool
  * select_confident.main()  from run_code/select_confident.py — pure confidence filter
  * merge.main()             from run_code/merge.py            — pure grow/shrink + passthrough

…just on the local CPU/GPU instead of SageMaker + Lambda. The loop halts when held-out
accuracy clears `--target`, OR no new confident pseudo-labels are found (plateau), OR
`--rounds` is hit — mirroring the flow.loop stop = {$.loopState.metrics.accuracy >= target_acc}
plus maxRounds.

Because train/ + pseudo_label read split DIRECTORIES, between rounds we materialise the
grown labelled set + shrunk pool to fresh dirs under runs/ (the cloud loop threads them as
s3-refs; locally they're directories). The labelled/pool/test SHAPE is identical.

Run from examples/mlops/self-training (after preparing the data):

    python scripts/prepare_cifar_subset.py --limit 2000
    python scripts/run_loop_local.py --rounds 5 --epochs 3 --threshold 0.95 --target 0.80 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
# Import the demo's REAL code: train() from train/, the Run Code entries from run_code/.
sys.path.insert(0, str(EXAMPLE_DIR / "train"))
sys.path.insert(0, str(EXAMPLE_DIR / "run_code"))

from finetune import read_index, train  # noqa: E402  (train/finetune.py)
import pseudo_label  # noqa: E402  (run_code/pseudo_label.py)
import select_confident  # noqa: E402  (run_code/select_confident.py)
import merge  # noqa: E402  (run_code/merge.py)


def _materialise_split(rows: list[dict[str, Any]], split_dir: Path) -> Path:
    """Write rows ({id, image_path (abs), label|None}) as a fresh split dir (index.csv + image refs).

    Images aren't copied — index.csv points at the original absolute paths — so growing the
    labelled set round-over-round is cheap. This is exactly the on-disk shape train/pseudo_label read.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    index_path = split_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_path", "label"])
        for row in rows:
            label = row.get("label")
            writer.writerow([row["id"], row["image_path"], ("" if label is None else int(label))])
    return split_dir


def _load_truth(data_dir: Path) -> dict[str, int]:
    """Optional private pool ground truth → {id: label}, for a pseudo-label-accuracy sanity check."""
    truth_path = data_dir / "_pool_truth.csv"
    if not truth_path.exists():
        return {}
    with truth_path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: int(row["label"]) for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the self-training loop locally with real training + inference.")
    parser.add_argument("--data", default="data", help="data dir holding labeled_seed/, unlabeled_pool/, test/")
    parser.add_argument("--rounds", type=int, default=5, help="max self-training rounds (loop maxRounds)")
    parser.add_argument("--epochs", type=int, default=3, help="epochs per round's train()")
    parser.add_argument("--batch", type=int, default=64, help="train batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--arch", default="resnet18", help="torchvision backbone")
    parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold to accept a pseudo-label")
    parser.add_argument("--target", type=float, default=0.80, help="stop when held-out accuracy >= this")
    parser.add_argument("--limit", type=int, default=None, help="optional cap on pool size per round (faster proof)")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda, cuda:0, cpu")
    args = parser.parse_args()

    data_dir = (EXAMPLE_DIR / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    for split in ("labeled_seed", "unlabeled_pool", "test"):
        if not (data_dir / split / "index.csv").exists():
            raise SystemExit(f"missing {split}/index.csv under {data_dir} — run prepare_cifar_subset.py first")

    runs_dir = EXAMPLE_DIR / "runs" / "loop"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    test_dir = data_dir / "test"
    truth = _load_truth(data_dir)

    # Initial loop vars (the Config·JSON initFrom values), as in-memory row lists.
    labeled = read_index(data_dir / "labeled_seed", require_label=True)
    pool = read_index(data_dir / "unlabeled_pool", require_label=False)
    if args.limit is not None:
        pool = pool[: args.limit]
    print(f"Start: {len(labeled)} labelled / {len(pool)} pool / target acc {args.target} / {args.rounds} max rounds")

    history: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}

    for round_idx in range(1, args.rounds + 1):
        print(f"\n========== ROUND {round_idx}/{args.rounds} ==========", flush=True)
        round_dir = runs_dir / f"round{round_idx}"
        labeled_dir = _materialise_split(labeled, round_dir / "labeled")
        pool_dir = _materialise_split(pool, round_dir / "pool")
        model_dir = round_dir / "model"

        # ---- train (REAL) ----
        train_out = train(
            {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "arch": args.arch,
             "device": args.device, "model_dir": str(model_dir)},
            {"labeled": str(labeled_dir), "test": str(test_dir)},
        )
        accuracy = float(train_out["metrics"]["accuracy"])
        last_metrics = train_out["metrics"]

        # ---- pseudo_label (REAL inference) ----
        if pool:
            pl_out = pseudo_label.main({"model": str(model_dir), "pool": str(pool_dir)})
            preds = pl_out["preds"]
        else:
            preds = []

        # ---- select_confident (pure) ----
        sel_out = select_confident.main({"preds": preds, "threshold": args.threshold})
        batch = sel_out["batch"]

        # Sanity check: how good are the ACCEPTED pseudo-labels vs the private truth? (no faking)
        pseudo_acc = None
        if truth and batch:
            hits = sum(1 for b in batch if truth.get(b["id"]) == b["label"])
            pseudo_acc = hits / len(batch)

        # ---- merge (pure round summary; carries model + metrics forward as the loop sink) ----
        merged = merge.main({
            "labeled": labeled, "pool": pool, "batch": batch,
            "model": {"path": str(model_dir)}, "metrics": train_out["metrics"],
        })
        labeled, pool = merged["labeled"], merged["pool"]
        new_confident = merged["new_confident"]

        line = (
            f"[round {round_idx}] accuracy={accuracy:.4f}  new_confident={new_confident}  "
            f"labeled={len(labeled)}  pool={len(pool)}"
        )
        if pseudo_acc is not None:
            line += f"  pseudo_label_acc={pseudo_acc:.3f}"
        print(line, flush=True)
        history.append({
            "round": round_idx, "accuracy": accuracy, "new_confident": new_confident,
            "labeled": len(labeled), "pool": len(pool), "pseudo_label_acc": pseudo_acc,
        })

        # ---- loop stop conditions (mirror flow.loop: stop on target, plateau, or maxRounds) ----
        if accuracy >= args.target:
            print(f"\nSTOP: accuracy {accuracy:.4f} >= target {args.target} (loop stop condition met)", flush=True)
            break
        if new_confident == 0:
            print("\nSTOP: no new confident pseudo-labels (plateau)", flush=True)
            break
        if not pool:
            print("\nSTOP: pool exhausted", flush=True)
            break

    print("\n================ LOOP SUMMARY ================")
    for h in history:
        extra = f"  pseudo_acc={h['pseudo_label_acc']:.3f}" if h["pseudo_label_acc"] is not None else ""
        print(f"  round {h['round']}: acc={h['accuracy']:.4f}  new_confident={h['new_confident']}  "
              f"labeled={h['labeled']}  pool={h['pool']}{extra}")
    final_acc = history[-1]["accuracy"] if history else 0.0
    print(f"\nFinal accuracy: {final_acc:.4f}  (target {args.target} → {'MET' if final_acc >= args.target else 'not met'})")

    summary = {
        "target": args.target, "threshold": args.threshold, "rounds_run": len(history),
        "final_accuracy": final_acc, "final_metrics": last_metrics, "history": history,
    }
    out_path = runs_dir / "loop_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(EXAMPLE_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
