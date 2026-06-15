"""Smoke tests for the YOLO keypoint run-only Conduit demo.

These exercise the file-port contract Conduit uses: every Run Code entry takes
local file paths for its file ports (Conduit's wrapper downloads the objects and
hands the code a path) and returns local file paths for its output file ports
(the wrapper uploads them). JSON ports are inline values.

Run from examples/vision/yolo-keypoints:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess_image = load_module(EXAMPLE_DIR / "run_code" / "preprocess_image.py", "demo_preprocess_image")
run_keypoints = load_module(EXAMPLE_DIR / "run_code" / "run_keypoints.py", "demo_run_keypoints")
render_results = load_module(EXAMPLE_DIR / "run_code" / "render_results.py", "demo_render_results")

ANNOTATIONS_PATH = EXAMPLE_DIR / "data/raw/annotations/person_keypoints_demo.json"


def _write_png(path: Path, size: tuple[int, int] = (200, 150)) -> Path:
    image = Image.new("RGB", size, (120, 130, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return path


class PreprocessImageTest(unittest.TestCase):
    def test_returns_processed_file_path_and_threaded_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_png(Path(tmp) / "input.png")
            result = preprocess_image.main({
                "image": str(src),
                "image_name": "person-walk-001.png",
                "target_size": 160,
            })

            self.assertEqual(result["image_id"], "person-walk-001")
            out = Path(result["preprocessed"])
            self.assertTrue(out.exists())
            with Image.open(out) as image:
                self.assertEqual(image.size, (160, 160))
            self.assertIn("output", result["metadata"])


class RunKeypointsTest(unittest.TestCase):
    def test_returns_prediction_file_and_inline_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            annotations = Path(tmp) / "annotations.json"
            annotations.write_text(ANNOTATIONS_PATH.read_text(encoding="utf-8"))

            result = run_keypoints.main({
                "annotations": str(annotations),
                "image_id": "person-walk-001",
            })

            self.assertEqual(result["image_id"], "person-walk-001")
            self.assertEqual(result["prediction"]["label"], "person")
            self.assertGreaterEqual(result["prediction"]["confidence"], 0.8)
            self.assertEqual(len(result["prediction"]["keypoints"]), 17)

            out = Path(result["predictions"])
            self.assertTrue(out.exists())
            on_disk = json.loads(out.read_text())
            self.assertEqual(on_disk["image_id"], "person-walk-001")
            self.assertEqual(on_disk, result["prediction"])


class RenderResultsTest(unittest.TestCase):
    def _prediction(self) -> dict:
        annotations = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        return run_keypoints.estimate_keypoints("person-walk-001", annotations)

    def test_accepts_inline_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _write_png(Path(tmp) / "person-walk-001-processed.png", (160, 160))
            result = render_results.main({
                "image": str(image),
                "image_id": "person-walk-001",
                "prediction": self._prediction(),
            })

            overlay = Path(result["overlay"])
            labels = Path(result["labels"])
            self.assertTrue(overlay.exists())
            self.assertTrue(labels.exists())
            with Image.open(overlay) as image_out:
                self.assertEqual(image_out.format, "PNG")
            payload = json.loads(labels.read_text())
            self.assertIn("summary", payload)
            self.assertIn("prediction", payload)
            self.assertEqual(result["summary"]["label"], "person")
            self.assertGreaterEqual(result["summary"]["keypoint_count"], 1)

    def test_accepts_prediction_file_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _write_png(Path(tmp) / "person-walk-001-processed.png", (160, 160))
            prediction_path = Path(tmp) / "person-walk-001-keypoints.json"
            prediction_path.write_text(json.dumps(self._prediction()))

            result = render_results.main({
                "image": str(image),
                "image_id": "person-walk-001",
                "predictions": str(prediction_path),
            })

            self.assertTrue(Path(result["overlay"]).exists())
            self.assertTrue(Path(result["labels"]).exists())

    def test_missing_prediction_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _write_png(Path(tmp) / "person-walk-001-processed.png", (160, 160))
            with self.assertRaises(ValueError):
                render_results.main({"image": str(image), "image_id": "person-walk-001"})


if __name__ == "__main__":
    unittest.main()
