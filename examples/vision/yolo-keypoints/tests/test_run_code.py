"""Smoke tests for the YOLO keypoint run-only Conduit demo.

Run from examples/vision/yolo-keypoints:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


preprocess_image = load_module(EXAMPLE_DIR / "run_code" / "preprocess_image.py", "demo_preprocess_image")
run_keypoints = load_module(EXAMPLE_DIR / "run_code" / "run_keypoints.py", "demo_run_keypoints")
render_results = load_module(EXAMPLE_DIR / "run_code" / "render_results.py", "demo_render_results")


class YoloKeypointsRunCodeTest(unittest.TestCase):
    def test_demo_image_runs_through_preprocess_keypoints_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            common = {
                "bucket": "demo-bucket",
                "_local_s3_root": str(EXAMPLE_DIR),
                "_local_output_root": tmp,
            }
            preprocess = preprocess_image.main({
                **common,
                "image": {
                    "bucket": "demo-bucket",
                    "key": "data/prepared/images/person-walk-001.png",
                },
                "output_prefix": "data/processed/images/",
            })
            predictions = run_keypoints.main({
                **common,
                "image": preprocess["preprocessed"],
                "annotations": {
                    "bucket": "demo-bucket",
                    "key": "data/raw/annotations/person_keypoints_demo.json",
                },
                "output_prefix": "data/predictions/",
            })
            rendered = render_results.main({
                **common,
                "image": preprocess["preprocessed"],
                "predictions": predictions["predictions"],
                "output_prefix": "data/outputs/overlays/",
            })

            self.assertEqual(preprocess["image_id"], "person-walk-001")
            self.assertEqual(predictions["prediction"]["label"], "person")
            self.assertGreaterEqual(predictions["prediction"]["confidence"], 0.8)
            self.assertTrue((Path(tmp) / rendered["overlay"]["key"]).exists())
            self.assertTrue((Path(tmp) / rendered["labels"]["key"]).exists())


if __name__ == "__main__":
    unittest.main()
