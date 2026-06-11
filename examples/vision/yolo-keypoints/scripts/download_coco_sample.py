"""Download a deterministic 100-image COCO keypoint sample for this demo.

This script fetches COCO 2017 validation annotations and copies the first
N images with person keypoint annotations into this example's data layout.
The downloaded images are intentionally not required for the repository's
smoke tests; they are for users who want a larger, real-image Run-only demo.

Run from examples/vision/yolo-keypoints:

    python scripts/download_coco_sample.py --limit 100

Outputs:
    data/raw/images/*.jpg
    data/raw/annotations/person_keypoints_coco_sample.json
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
ANNOTATION_MEMBER = "annotations/person_keypoints_val2017.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small COCO 2017 person-keypoints sample.")
    parser.add_argument("--limit", type=int, default=100, help="number of images to download")
    parser.add_argument("--force", action="store_true", help="overwrite existing downloaded images")
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    example_dir = Path(__file__).resolve().parents[1]
    images_dir = example_dir / "data" / "raw" / "images"
    annotations_dir = example_dir / "data" / "raw" / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="conduit-coco-") as tmp:
        tmpdir = Path(tmp)
        zip_path = tmpdir / "annotations_trainval2017.zip"
        print(f"Downloading COCO annotations: {ANNOTATIONS_ZIP_URL}")
        download(ANNOTATIONS_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(ANNOTATION_MEMBER) as handle:
                annotations = json.load(handle)

    sample = select_keypoint_sample(annotations, limit=args.limit)
    for image in sample["images"]:
        file_name = str(image["file_name"])
        dest = images_dir / file_name
        if dest.exists() and not args.force:
            print(f"exists {file_name}")
            continue
        url = IMAGE_URL_TEMPLATE.format(file_name=file_name)
        print(f"Downloading {url}")
        download(url, dest)

    out_path = annotations_dir / "person_keypoints_coco_sample.json"
    out_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(example_dir)} with {len(sample['images'])} images")
    from convert_coco_to_yolo_pose import convert
    convert(
        annotations_path=out_path,
        labels_dir=example_dir / "data/raw/labels",
        yaml_path=example_dir / "data/yolo-pose.yaml",
    )
    print("Regenerated data/raw/labels/*.txt and data/yolo-pose.yaml")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Conduit-Demo/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)


def select_keypoint_sample(annotations: dict[str, Any], *, limit: int) -> dict[str, Any]:
    images_by_id = {int(image["id"]): image for image in annotations.get("images", [])}
    selected_image_ids: list[int] = []
    selected_annotations: list[dict[str, Any]] = []
    seen: set[int] = set()
    for annotation in annotations.get("annotations", []):
        image_id = int(annotation.get("image_id", -1))
        if image_id not in images_by_id:
            continue
        if int(annotation.get("num_keypoints", 0)) <= 0:
            continue
        if image_id not in seen:
            if len(selected_image_ids) >= limit:
                break
            seen.add(image_id)
            selected_image_ids.append(image_id)
        if image_id in seen:
            selected_annotations.append(annotation)

    selected_ids = set(selected_image_ids)
    return {
        "info": {
            "description": f"COCO 2017 val person-keypoints sample for Conduit YOLO demo ({len(selected_image_ids)} images)",
            "source": "http://images.cocodataset.org/",
            "format": "coco-keypoints-ground-truth",
        },
        "images": [clean_image(images_by_id[image_id]) for image_id in selected_image_ids],
        "annotations": [
            clean_annotation(item)
            for item in selected_annotations
            if int(item.get("image_id", -1)) in selected_ids
        ],
        "categories": [clean_category(item) for item in annotations.get("categories", [])],
    }


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
        "num_keypoints": int(annotation["num_keypoints"]),
        "keypoints": annotation["keypoints"],
    }


def clean_category(category: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(category["id"]),
        "name": str(category["name"]),
        "supercategory": str(category.get("supercategory", "person")),
        "keypoints": category.get("keypoints", []),
        "skeleton": category.get("skeleton", []),
    }



if __name__ == "__main__":
    main()
