"""Carve a small REAL CIFAR-10 subset into a self-training layout (no synthetic data).

Downloads CIFAR-10 via torchvision (real 32x32 photographs across 10 classes), then writes
three on-disk splits under data/:

    data/labeled_seed/    # ~10% of the train pool, WITH labels (the loop's starting seed)
    data/unlabeled_pool/  # ~90% of the train pool, labels HIDDEN (the loop pseudo-labels these)
    data/test/            # a held-out, fully-labelled eval split (the loop's accuracy probe)

Each split is a directory holding `index.csv` (header `id,image_path,label`) plus the
referenced PNGs under `images/`. `image_path` is relative to the split dir. `label` is the
integer CIFAR-10 class (0-9); it is left EMPTY for the unlabeled pool. This is the simplest
real format — a CSV manifest + real PNGs — readable without torch (see ../README.md).

To prove the loop actually LEARNS from pseudo-labels, the pool keeps the true labels in a
private `data/_pool_truth.csv` (NOT fed to the model, NOT uploaded) so the local driver can
report pseudo-label accuracy as a sanity check. The training/inference code never reads it.

Run from examples/mlops/self-training:

    python scripts/prepare_cifar_subset.py --limit 2000
    # → ~10% seed / ~90% pool of `limit` train images, + `limit`-sized test split.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Carve a small real CIFAR-10 subset for self-training.")
    parser.add_argument("--limit", type=int, default=2000, help="train images to draw (split ~10%% seed / ~90%% pool)")
    parser.add_argument("--test-limit", type=int, default=None, help="test images (default: same as --limit)")
    parser.add_argument("--seed-frac", type=float, default=0.10, help="fraction of the train draw that is the labelled seed")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for a deterministic carve")
    parser.add_argument("--force", action="store_true", help="overwrite an existing data/ layout")
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    example_dir = Path(__file__).resolve().parents[1]
    data_dir = example_dir / "data"
    test_limit = args.test_limit if args.test_limit is not None else args.limit

    # torchvision is the ONLY heavy dep here — real CIFAR-10, no synthetic data.
    from torchvision.datasets import CIFAR10

    raw_dir = data_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CIFAR-10 into {raw_dir} (real photographs, 10 classes)…", flush=True)
    train_ds = CIFAR10(root=str(raw_dir), train=True, download=True)
    test_ds = CIFAR10(root=str(raw_dir), train=False, download=True)

    rng = random.Random(args.seed)

    # Deterministic draw from the train set → seed + pool.
    train_idx = list(range(len(train_ds)))
    rng.shuffle(train_idx)
    train_idx = train_idx[: args.limit]
    seed_count = max(1, round(args.limit * args.seed_frac))
    seed_idx = train_idx[:seed_count]
    pool_idx = train_idx[seed_count:]

    test_idx = list(range(len(test_ds)))
    rng.shuffle(test_idx)
    test_idx = test_idx[:test_limit]

    _write_split(data_dir / "labeled_seed", train_ds, seed_idx, prefix="seed", with_label=True)
    _write_split(data_dir / "unlabeled_pool", train_ds, pool_idx, prefix="pool", with_label=False)
    _write_split(data_dir / "test", test_ds, test_idx, prefix="test", with_label=True)

    # Private ground truth for the pool (sanity-check pseudo-labels locally; NEVER trained on / uploaded).
    truth_path = data_dir / "_pool_truth.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for i in pool_idx:
            writer.writerow([f"pool-{i}", int(train_ds[i][1])])

    print(
        f"\nWrote splits under {data_dir.relative_to(example_dir)}:\n"
        f"  labeled_seed   {len(seed_idx)} images (labelled)\n"
        f"  unlabeled_pool {len(pool_idx)} images (labels hidden)\n"
        f"  test           {len(test_idx)} images (labelled)\n"
        f"  _pool_truth.csv (private — local sanity check only)\n"
        f"Next: python scripts/run_loop_local.py --rounds 5 --epochs 3 --device cuda",
        flush=True,
    )
    return 0


def _write_split(split_dir: Path, dataset: Any, indices: list[int], *, prefix: str, with_label: bool) -> None:
    """Write `index.csv` + PNGs for one split. Real CIFAR PIL images, saved as PNG."""
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    index_path = split_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_path", "label"])
        for i in indices:
            image, label = dataset[i]  # torchvision CIFAR10 yields (PIL.Image, int)
            example_id = f"{prefix}-{i}"
            rel = f"images/{example_id}.png"
            image.save(split_dir / rel)
            writer.writerow([example_id, rel, (int(label) if with_label else "")])
    print(f"  {split_dir.name}: {len(indices)} images → {index_path.name}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
