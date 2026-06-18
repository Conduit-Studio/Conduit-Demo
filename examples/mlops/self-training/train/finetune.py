"""Build-from-code training function for the self-training loop (REAL ResNet on CIFAR-10).

This is the WHOLE training contract under Conduit's **build-from-code** Train Model node:
you author `train(hyperparameters, channels)` and a `requirements.txt`, and Conduit builds
+ digest-pins the GPU SageMaker image for you (curated AWS PyTorch GPU Deep Learning
Container base + your requirements + this code + Conduit's wrapper). There is **no
Dockerfile and no `/opt/ml` plumbing** — Conduit's wrapper:

  * hands `train()` the job's hyperparameters and the local channel paths (the `train`
    and `validation` channels mounted from S3 — these are the Train Model node's real
    channel ports `trainingData`→channel `train` and `validationData`→channel `validation`);
  * copies the model directory you return into `/opt/ml/model` (→ `model.tar.gz`, which
    becomes the Train Model node's `model` output);
  * scrapes the `metrics` dict you return → the node's `metrics` output → lineage.

So nothing here talks to AWS or SageMaker. The exact same `train()` runs:
  * locally, called by `scripts/run_loop_local.py` (real GPU fine-tune, no faking);
  * in your account, inside the image Conduit builds for the Train Model node;
  * once PER ROUND of the self-training loop (the `flow.loop` re-feeds the grown labelled
    set into the `train` channel each round).

SageMaker delivers every hyperparameter as a STRING, so we coerce.

## On-disk data format (see ../../README.md)
A "split" is a directory holding `index.csv` — a manifest with header `id,image_path,label`
— plus the referenced image files (PNG). `image_path` is relative to the split dir.
`label` is the integer CIFAR-10 class (0-9) for the labelled/validation splits; it is empty
for the unlabeled pool. This is the SIMPLEST real format: a CSV index + real PNGs, readable
without torch. `train()` reads the `train` channel (training) and `validation` channel (eval).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = len(CIFAR10_CLASSES)
INDEX_NAME = "index.csv"


def _coerce(value: Any, cast, default):
    """Best-effort cast (SageMaker hyperparameters all arrive as strings)."""
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def find_index(split_dir: Path) -> Path:
    """Locate the manifest CSV inside a mounted split channel."""
    candidate = split_dir / INDEX_NAME
    if candidate.exists():
        return candidate
    matches = sorted(split_dir.rglob(INDEX_NAME))
    if not matches:
        raise FileNotFoundError(f"no {INDEX_NAME} found under {split_dir}")
    return matches[0]


def read_index(split_dir: Path, *, require_label: bool) -> list[dict[str, Any]]:
    """Read a split's manifest into rows {id, image_path (absolute), label|None}.

    `require_label=True` drops rows without a label (a labelled/test split must be
    fully labelled to train/eval on); `require_label=False` keeps label as None
    (the unlabeled pool). Pure stdlib csv — no torch needed to inspect a split.
    """
    index_path = find_index(split_dir)
    root = index_path.parent
    rows: list[dict[str, Any]] = []
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_raw = (row.get("label") or "").strip()
            has_label = label_raw != ""
            if require_label and not has_label:
                continue
            rows.append({
                "id": str(row["id"]),
                "image_path": str((root / row["image_path"]).resolve()),
                "label": int(label_raw) if has_label else None,
            })
    return rows


def _arch_weights(arch: str):
    """Return (model_ctor, default_weights) for the requested torchvision backbone."""
    import torchvision

    arch = (arch or "resnet18").lower()
    table = {
        "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V1),
    }
    if arch not in table:
        raise ValueError(f"unsupported arch {arch!r}; choose one of {sorted(table)}")
    return table[arch]


def train(hyperparameters: dict[str, Any], channels: dict[str, str]) -> dict[str, Any]:
    """Fine-tune a torchvision ResNet on the labelled CIFAR-10 split; report REAL top-1.

    Args:
        hyperparameters: keys `epochs`, `batch`, `lr`, `arch` (default resnet18). Values
            may be strings (SageMaker hands every HP over as a string).
        channels: maps channel name → local path. `channels["train"]` is the current
            labelled split dir (the loop grows this each round); `channels["validation"]` is a
            held-out, fully-labelled split for evaluation. (These are the Train Model node's
            `trainingData`→channel `train` and `validationData`→channel `validation` ports.)

    Returns:
        {"model": <model dir>, "metrics": {"accuracy": float}} — `model` is the directory
        Conduit copies into /opt/ml/model (it holds `model.pt` + `classes.json`);
        `metrics.accuracy` is the REAL held-out top-1 accuracy the loop's stop condition reads.
    """
    epochs = _coerce(hyperparameters.get("epochs", 3), int, 3)
    batch = _coerce(hyperparameters.get("batch", 64), int, 64)
    lr = _coerce(hyperparameters.get("lr", 0.001), float, 0.001)
    arch = str(hyperparameters.get("arch", "resnet18"))
    device_hp = hyperparameters.get("device")  # None → auto

    labeled_dir = Path(channels.get("train") or next(iter(channels.values()))).resolve()
    test_dir = Path(channels["validation"]).resolve()

    model_dir = Path(
        os.environ.get("CONDUIT_MODEL_DIR")
        or hyperparameters.get("model_dir")
        or "runs/model"
    ).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train] arch={arch} epochs={epochs} batch={batch} lr={lr} "
        f"labeled={labeled_dir} test={test_dir} model_dir={model_dir}",
        flush=True,
    )

    # Heavy imports kept inside the function so importing this module (tests, the wrapper
    # discovering `train`) stays cheap and torch-free.
    import json

    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    device = torch.device(
        device_hp if device_hp else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ImageNet-normalised 32→224 upsample so the pretrained ResNet stem sees a sane scale.
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class _SplitDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            image = Image.open(row["image_path"]).convert("RGB")
            return tfm(image), int(row["label"])

    train_rows = read_index(labeled_dir, require_label=True)
    test_rows = read_index(test_dir, require_label=True)
    if not train_rows:
        raise ValueError(f"no labelled rows under {labeled_dir}")
    if not test_rows:
        raise ValueError(f"no labelled rows under {test_dir}")
    print(f"[train] {len(train_rows)} labelled / {len(test_rows)} test", flush=True)

    train_loader = DataLoader(_SplitDataset(train_rows), batch_size=batch, shuffle=True)
    test_loader = DataLoader(_SplitDataset(test_rows), batch_size=batch, shuffle=False)

    ctor, weights = _arch_weights(arch)
    model = ctor(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # adapt the head to 10 classes
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        print(f"[train] epoch {epoch + 1}/{epochs} loss={running / max(len(train_loader), 1):.4f}", flush=True)

    # Held-out top-1 accuracy (REAL eval — no replay).
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1).cpu()
            correct += int((preds == labels.cpu()).sum().item())
            total += int(labels.size(0))
    accuracy = correct / total if total else 0.0

    torch.save({"state_dict": model.state_dict(), "arch": arch}, model_dir / "model.pt")
    (model_dir / "classes.json").write_text(json.dumps(list(CIFAR10_CLASSES)), encoding="utf-8")

    print(f"accuracy: {accuracy:.5f}", flush=True)
    print(f"[train] model saved to {model_dir}", flush=True)

    return {"model": str(model_dir), "metrics": {"accuracy": accuracy}}


def _main() -> int:
    """Run a single round's training locally for a quick smoke."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune one round locally.")
    parser.add_argument("--labeled", required=True, help="labelled split dir → `train` channel (holds index.csv)")
    parser.add_argument("--test", required=True, help="held-out test split dir → `validation` channel (holds index.csv)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    args = parser.parse_args()

    out = train(
        {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "arch": args.arch, "device": args.device},
        {"train": args.labeled, "validation": args.test},
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
