"""Conduit Run Code step: pseudo-label the unlabeled pool with REAL inference.

Drop this on the canvas as a Run Code node (Container/GPU or CPU — it runs the trained
torchvision ResNet, so a GPU helps but isn't required). It is the SECOND step of the loop
body:

    flow.loop[ train -> pseudo_label -> select_confident -> merge ]
                        ^^^^^^^^^^^^

It loads the model `train` just produced, runs REAL forward passes over every example in
the current unlabeled pool, and for each emits the argmax class + the softmax max-prob as
`confidence`. NO label replay, NO canned scores — the confidence is a genuine softmax over
the model's logits. The heavy data (model dir, pool images) travels as files/dirs (s3-ref);
the small per-example preds list returns as json.

Input ports
    model : model-artifact / s3-ref — the dir `train` returned (holds model.pt + classes.json).
    pool  : s3-ref (dir) — the unlabeled pool split dir (holds index.csv + images).
    batch : (optional) inference batch size; defaults to 128. May be a string.

Output ports
    preds : json[] — one {id, label, confidence} per pooled example. `label` is the argmax
            CIFAR-10 class (0-9); `confidence` is the softmax max-probability (0..1).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make the sibling package + the train/ module importable when Conduit runs this as the entrypoint.
_RUN_CODE_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _RUN_CODE_DIR.parent / "train"
for _p in (str(_RUN_CODE_DIR), str(_TRAIN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_dir(value: Any) -> Path:
    """Accept a plain path string or an s3-ref-shaped {path|dir|local}: locate the local dir."""
    if isinstance(value, str):
        return Path(value).resolve()
    if isinstance(value, dict):
        for key in ("path", "dir", "local", "localPath"):
            if value.get(key):
                return Path(str(value[key])).resolve()
    raise ValueError(f"cannot resolve a local directory from {value!r}")


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    model_dir = _resolve_dir(inputs.get("model"))
    pool_dir = _resolve_dir(inputs.get("pool"))
    infer_batch = int(inputs.get("batch") or 128)

    # Heavy imports kept inside main() so importing this file stays cheap + torch-free for tests.
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    # Reuse the EXACT training-side index reader + class list so inference matches training.
    from finetune import NUM_CLASSES, _arch_weights, read_index

    rows = read_index(pool_dir, require_label=False)
    if not rows:
        return {"preds": []}

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class _PoolDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            row = self.rows[i]
            image = Image.open(row["image_path"]).convert("RGB")
            return tfm(image), row["id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_dir / "model.pt", map_location=device)
    arch = str(ckpt.get("arch", "resnet18"))
    ctor, _ = _arch_weights(arch)
    model = ctor(weights=None)
    from torch import nn

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    loader = DataLoader(_PoolDataset(rows), batch_size=infer_batch, shuffle=False)

    preds: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, ids in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, label = probs.max(dim=1)
            for example_id, lbl, c in zip(ids, label.cpu().tolist(), conf.cpu().tolist()):
                preds.append({"id": str(example_id), "label": int(lbl), "confidence": float(c)})

    return {"preds": preds}
