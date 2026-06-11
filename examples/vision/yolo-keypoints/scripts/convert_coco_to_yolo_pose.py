"""Convert the demo COCO keypoint annotations to YOLO pose labels.

The COCO JSON remains the source ground-truth record. This script writes the
label files and dataset YAML a real YOLO pose training or evaluation command
expects.

Run from examples/vision/yolo-keypoints:

    python scripts/convert_coco_to_yolo_pose.py

Outputs:
    data/raw/labels/*.txt
    data/yolo-pose.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ANNOTATIONS = "data/raw/annotations/person_keypoints_coco_sample.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create YOLO pose labels from the demo COCO keypoint JSON.")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--labels-dir", default="data/raw/labels")
    parser.add_argument("--yaml", default="data/yolo-pose.yaml")
    args = parser.parse_args()

    example_dir = Path(__file__).resolve().parents[1]
    annotations_path = example_dir / args.annotations
    labels_dir = example_dir / args.labels_dir
    yaml_path = example_dir / args.yaml
    convert(annotations_path=annotations_path, labels_dir=labels_dir, yaml_path=yaml_path)
    print(f"Wrote YOLO pose labels to {labels_dir.relative_to(example_dir)}")
    print(f"Wrote dataset YAML to {yaml_path.relative_to(example_dir)}")


def convert(*, annotations_path: Path, labels_dir: Path, yaml_path: Path) -> None:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    labels_by_file: dict[str, list[str]] = {str(image["file_name"]): [] for image in images_by_id.values()}

    for annotation in data.get("annotations", []):
        if int(annotation.get("category_id", 0)) != 1:
            continue
        image = images_by_id.get(int(annotation.get("image_id", -1)))
        if not image:
            continue
        labels_by_file[str(image["file_name"])].append(to_yolo_pose_row(annotation, image))

    labels_dir.mkdir(parents=True, exist_ok=True)
    for file_name, rows in labels_by_file.items():
        label_path = labels_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(text(), encoding="utf-8")


def to_yolo_pose_row(annotation: dict[str, Any], image: dict[str, Any]) -> str:
    width = float(image["width"])
    height = float(image["height"])
    x, y, w, h = [float(value) for value in annotation["bbox"]]
    bbox = [
        clamp((x + w / 2) / width),
        clamp((y + h / 2) / height),
        clamp(w / width),
        clamp(h / height),
    ]
    keypoints = []
    raw = list(annotation.get("keypoints", []))
    if len(raw) != 51:
        raise ValueError(f"annotation {annotation.get('id')} has {len(raw)} keypoint values, expected 51")
    for index in range(0, len(raw), 3):
        px, py, visible = float(raw[index]), float(raw[index + 1]), int(raw[index + 2])
        keypoints.extend([clamp(px / width), clamp(py / height), float(visible)])
    values = [0.0, *bbox, *keypoints]
    return " ".join(format_value(value) for value in values)


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def text() -> str:
    return """# YOLO pose dataset file for the Conduit keypoint demo.\n# Paths are relative to this YAML file.\npath: .\ntrain: data/raw/images\nval: data/raw/images\n\nkpt_shape: [17, 3]\nflip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n\nnames:\n  0: person\n"""


if __name__ == "__main__":
    main()
