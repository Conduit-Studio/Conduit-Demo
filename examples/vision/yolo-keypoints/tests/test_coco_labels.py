"""Validate the COCO sample and derived YOLO pose labels.

Run from examples/vision/yolo-keypoints:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1]


class CocoYoloLabelTest(unittest.TestCase):
    def test_coco_sample_has_matching_yolo_pose_labels(self) -> None:
        annotations_path = EXAMPLE_DIR / "data/raw/annotations/person_keypoints_coco_sample.json"
        labels_dir = EXAMPLE_DIR / "data/raw/labels"
        data = json.loads(annotations_path.read_text(encoding="utf-8"))
        images = data["images"]
        annotations = data["annotations"]

        self.assertEqual(len(images), 100)
        self.assertEqual(len(annotations), 188)
        self.assertTrue((EXAMPLE_DIR / "data/yolo-pose.yaml").exists())

        label_paths = [labels_dir / f"{Path(image['file_name']).stem}.txt" for image in images]
        missing = [path.name for path in label_paths if not path.exists()]
        self.assertEqual(missing, [])

        label_rows = [
            row
            for path in label_paths
            for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()
        ]
        self.assertEqual(len(label_rows), len(annotations))
        for row in label_rows[:10]:
            values = row.split()
            self.assertEqual(values[0], "0")
            self.assertEqual(len(values), 56)  # class + bbox(4) + 17 keypoints * xyv


if __name__ == "__main__":
    unittest.main()
