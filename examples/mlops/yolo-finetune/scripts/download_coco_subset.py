"""Download a deterministic real COCO 2017 person+car detection subset.

This script fetches the COCO 2017 validation annotations and copies a
deterministic set of images that contain at least one non-crowd person or
car bounding box into this example's dataset layout, split ~80/20 into
train/val. The downloaded images are real photographs from COCO — there is
NO synthetic data here.

Run from examples/mlops/yolo-finetune:

    python scripts/download_coco_subset.py --limit 100

Outputs:
    data/dataset/images/train/*.jpg
    data/dataset/images/val/*.jpg
    data/raw/annotations/instances_coco_subset.json   (offline source for convert)

After downloading, build the YOLO labels + data.yaml:

    python scripts/convert_coco_to_yolo_bbox.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{file_name}"
ANNOTATION_MEMBER = "annotations/instances_val2017.json"

# Target COCO categories. The ids are confirmed against the JSON's categories
# array at runtime by matching on name, so a category-id change upstream still
# resolves correctly.
TARGET_CATEGORY_NAMES = ("person", "car")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small real COCO 2017 person+car detection subset.")
    parser.add_argument("--limit", type=int, default=100, help="total number of images to download (train + val)")
    parser.add_argument("--force", action="store_true", help="overwrite existing downloaded images")
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    example_dir = Path(__file__).resolve().parents[1]
    dataset_dir = example_dir / "data" / "dataset"
    annotations_dir = example_dir / "data" / "raw" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="conduit-coco-") as tmp:
        tmpdir = Path(tmp)
        zip_path = tmpdir / "annotations_trainval2017.zip"
        print(f"Downloading COCO annotations: {ANNOTATIONS_ZIP_URL}")
        download(ANNOTATIONS_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(ANNOTATION_MEMBER) as handle:
                annotations = json.load(handle)

    subset = select_detection_subset(annotations, limit=args.limit)

    splits = {"train": subset["splits"]["train"], "val": subset["splits"]["val"]}
    images_by_id = {int(image["id"]): image for image in subset["images"]}
    for split, image_ids in splits.items():
        split_dir = dataset_dir / "images" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            file_name = str(images_by_id[image_id]["file_name"])
            dest = split_dir / file_name
            if dest.exists() and not args.force:
                print(f"exists {split}/{file_name}")
                continue
            url = IMAGE_URL_TEMPLATE.format(file_name=file_name)
            print(f"Downloading {split}/{file_name} <- {url}")
            download(url, dest)

    out_path = annotations_dir / "instances_coco_subset.json"
    out_path.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {out_path.relative_to(example_dir)} with "
        f"{len(subset['splits']['train'])} train + {len(subset['splits']['val'])} val images"
    )
    print("Next: python scripts/convert_coco_to_yolo_bbox.py")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Conduit-Demo/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)


def select_detection_subset(annotations: dict[str, Any], *, limit: int) -> dict[str, Any]:
    """Pick a deterministic set of images with a non-crowd person/car box.

    Candidate image ids are sorted ascending so re-runs choose the same images.
    The first ~80% become train, the rest val (val = round(limit * 0.2)).
    """
    target_ids = resolve_target_category_ids(annotations.get("categories", []))

    images_by_id = {int(image["id"]): image for image in annotations.get("images", [])}

    # Group target (person/car, non-crowd) annotations by image.
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations.get("annotations", []):
        image_id = int(annotation.get("image_id", -1))
        if image_id not in images_by_id:
            continue
        if int(annotation.get("category_id", -1)) not in target_ids:
            continue
        if int(annotation.get("iscrowd", 0)) == 1:
            continue
        annotations_by_image.setdefault(image_id, []).append(annotation)

    # Deterministic candidate ordering: image ids with >=1 kept box, ascending.
    candidate_ids = sorted(annotations_by_image.keys())[:limit]

    val_count = round(limit * 0.2)
    train_count = limit - val_count
    train_ids = candidate_ids[:train_count]
    val_ids = candidate_ids[train_count : train_count + val_count]
    selected_ids = train_ids + val_ids

    selected_annotations: list[dict[str, Any]] = []
    for image_id in selected_ids:
        selected_annotations.extend(annotations_by_image[image_id])

    return {
        "info": {
            "description": (
                f"COCO 2017 val person+car detection subset for Conduit YOLO fine-tune demo "
                f"({len(selected_ids)} images)"
            ),
            "source": "http://images.cocodataset.org/",
            "format": "coco-instances-ground-truth",
        },
        "splits": {"train": train_ids, "val": val_ids},
        "images": [clean_image(images_by_id[image_id]) for image_id in selected_ids],
        "annotations": [clean_annotation(item) for item in selected_annotations],
        "categories": [
            clean_category(item)
            for item in annotations.get("categories", [])
            if int(item.get("id", -1)) in target_ids
        ],
    }


def resolve_target_category_ids(categories: list[dict[str, Any]]) -> set[int]:
    by_name = {str(category["name"]): int(category["id"]) for category in categories}
    ids: set[int] = set()
    for name in TARGET_CATEGORY_NAMES:
        if name not in by_name:
            raise SystemExit(f"category {name!r} not found in COCO categories")
        ids.add(by_name[name])
    return ids


def clean_image(image: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(image["id"]),
        "file_name": str(image["file_name"]),
        "width": int(image["width"]),
        "height": int(image["height"]),
    }


def clean_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(annotation["id"]),
        "image_id": int(annotation["image_id"]),
        "category_id": int(annotation["category_id"]),
        "bbox": annotation["bbox"],
        "area": annotation.get("area"),
        "iscrowd": int(annotation.get("iscrowd", 0)),
    }


def clean_category(category: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(category["id"]),
        "name": str(category["name"]),
        "supercategory": str(category.get("supercategory", "")),
    }


if __name__ == "__main__":
    main()
