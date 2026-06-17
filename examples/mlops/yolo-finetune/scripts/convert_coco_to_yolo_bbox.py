"""Convert the saved COCO person+car subset to YOLO detection labels.

The COCO JSON written by scripts/download_coco_subset.py remains the source
ground-truth record. This script writes the per-split label files and the
dataset YAML that a real YOLO detection training command expects.

Class remap: COCO person -> YOLO 0, COCO car -> YOLO 1. Only person/car
boxes are emitted; iscrowd==1 boxes are skipped. Each label line is
`cls cx cy w h`, normalized to [0,1] by the image width/height and clamped.

Images that end up with no kept boxes still get an EMPTY .txt label file:
ultralytics treats an empty label file as a valid background image, so this
keeps the image-to-label mapping complete without inventing boxes.

Run from examples/mlops/yolo-finetune:

    python scripts/convert_coco_to_yolo_bbox.py

Outputs:
    data/dataset/labels/train/<stem>.txt
    data/dataset/labels/val/<stem>.txt
    data/dataset/data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ANNOTATIONS = "data/raw/annotations/instances_coco_subset.json"
DEFAULT_DATASET = "data/dataset"

CLASSES = ["person", "car"]
# COCO category id -> YOLO class index.
COCO_TO_YOLO = {1: 0, 3: 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create YOLO detection labels from the demo COCO subset JSON.")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    args = parser.parse_args()

    example_dir = Path(__file__).resolve().parents[1]
    annotations_path = example_dir / args.annotations
    dataset_dir = example_dir / args.dataset_dir
    convert(annotations_path=annotations_path, dataset_dir=dataset_dir)
    print(f"Wrote YOLO detection labels under {(dataset_dir / 'labels').relative_to(example_dir)}")
    print(f"Wrote dataset YAML to {(dataset_dir / 'data.yaml').relative_to(example_dir)}")


def convert(*, annotations_path: Path, dataset_dir: Path) -> None:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    splits: dict[str, list[int]] = data.get("splits", {})

    # image_id -> [yolo rows]; start every selected image with an empty list so
    # background images (no kept boxes) still get an empty label file.
    rows_by_image: dict[int, list[str]] = {image_id: [] for image_id in images_by_id}
    for annotation in data.get("annotations", []):
        if int(annotation.get("iscrowd", 0)) == 1:
            continue
        cls = COCO_TO_YOLO.get(int(annotation.get("category_id", -1)))
        if cls is None:
            continue
        image = images_by_id.get(int(annotation.get("image_id", -1)))
        if not image:
            continue
        rows_by_image[int(image["id"])].append(to_yolo_bbox_row(cls, annotation, image))

    for split, image_ids in splits.items():
        labels_dir = dataset_dir / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            image = images_by_id[int(image_id)]
            rows = rows_by_image.get(int(image_id), [])
            label_path = labels_dir / f"{Path(str(image['file_name'])).stem}.txt"
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    yaml_path = dataset_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(text(), encoding="utf-8")


def to_yolo_bbox_row(cls: int, annotation: dict[str, Any], image: dict[str, Any]) -> str:
    width = float(image["width"])
    height = float(image["height"])
    x, y, w, h = [float(value) for value in annotation["bbox"]]
    values = [
        float(cls),
        clamp((x + w / 2) / width),
        clamp((y + h / 2) / height),
        clamp(w / width),
        clamp(h / height),
    ]
    return " ".join(format_value(value) for value in values)


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def text() -> str:
    return (
        "# YOLO detection dataset config. Paths are relative to THIS file = the `dataset`\n"
        "# channel root (SageMaker mounts the channel at /opt/ml/input/data/dataset inside\n"
        "# the training container).\n"
        "#\n"
        "# This static copy lets you read the shape without downloading. Running\n"
        "# scripts/download_coco_subset.py + scripts/convert_coco_to_yolo_bbox.py overwrites\n"
        "# it and fills images/ + labels/.\n"
        "#\n"
        "# path: '' means 'the directory holding THIS yaml' (ultralytics resolves an empty\n"
        "# relative path to the yaml's parent). Do NOT use '.', which ultralytics resolves\n"
        "# against the process CWD, not the yaml location.\n"
        "path: ''\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: [{', '.join(CLASSES)}]\n"
    )


if __name__ == "__main__":
    main()
