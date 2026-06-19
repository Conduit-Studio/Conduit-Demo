"""CI-fast unit tests for train/finetune.py's pure helpers.

These never import ultralytics or torch (it's lazy-imported inside `train()`),
so they run anywhere with no GPU. The real training path is exercised by
scripts/run_sweep_local.py against a real dataset + GPU.

    python -m unittest examples/mlops/yolo-finetune/tests/test_finetune.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Allow `python -m unittest tests/test_finetune.py` from the example root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from finetune import _coerce, _resolve_weights, _run_name, find_data_yaml  # noqa: E402


class TestCoerce(unittest.TestCase):
    def test_coerces_string_hyperparameters(self):
        # SageMaker delivers every hyperparameter as a string.
        self.assertEqual(_coerce("640", int, 640), 640)
        self.assertAlmostEqual(_coerce("0.005", float, 0.01), 0.005)

    def test_passthrough_native_types(self):
        self.assertEqual(_coerce(80, int, 50), 80)

    def test_none_falls_back_to_default(self):
        self.assertEqual(_coerce(None, int, 50), 50)

    def test_unparseable_falls_back_to_default(self):
        self.assertEqual(_coerce("not-a-number", float, 0.01), 0.01)


class TestRunName(unittest.TestCase):
    def test_prefers_trial_name(self):
        self.assertEqual(_run_name({"name": "s-1280", "weights": "yolov8s.pt"}), "s-1280")

    def test_derives_from_weights_and_imgsz_when_no_name(self):
        self.assertEqual(_run_name({"weights": "yolov8n.pt", "imgsz": 640}), "yolov8n-640")


class TestResolveWeights(unittest.TestCase):
    def test_bare_name_when_no_models_channel(self):
        # No `models` channel wired → return the name so ultralytics downloads it (unchanged path).
        self.assertEqual(_resolve_weights("yolov8n.pt", {"dataset": "/data"}), "yolov8n.pt")

    def test_bare_name_when_file_absent_in_channel(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_resolve_weights("yolov8n.pt", {"models": d}), "yolov8n.pt")

    def test_uses_local_file_when_present_in_channel(self):
        with tempfile.TemporaryDirectory() as d:
            staged = Path(d) / "yolov8s.pt"
            staged.write_bytes(b"weights")
            self.assertEqual(_resolve_weights("yolov8s.pt", {"models": d}), str(staged))


class TestFindDataYaml(unittest.TestCase):
    def test_finds_top_level_data_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "data.yaml").write_text("nc: 2\n")
            self.assertEqual(find_data_yaml(d), d / "data.yaml")

    def test_finds_nested_yaml_when_no_top_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "nested").mkdir()
            (d / "nested" / "custom.yaml").write_text("nc: 2\n")
            self.assertEqual(find_data_yaml(d), d / "nested" / "custom.yaml")

    def test_raises_when_no_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_data_yaml(Path(tmp))


if __name__ == "__main__":
    unittest.main()
