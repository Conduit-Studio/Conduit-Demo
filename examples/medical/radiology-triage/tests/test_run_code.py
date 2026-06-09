"""Smoke test for the public radiology triage demo.

Run from examples/medical/radiology-triage:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


read_study = load_module(EXAMPLE_DIR / "run_code" / "read_study.py", "demo_read_study")
infer_urgency = load_module(EXAMPLE_DIR / "run_code" / "infer_urgency.py", "demo_infer_urgency")


class RadiologyTriageRunCodeTest(unittest.TestCase):
    def test_demo_manifest_runs_through_read_and_infer_steps(self) -> None:
        features_result = read_study.main({
            "manifest": {
                "bucket": "demo-bucket",
                "key": "data/studies/incoming/study-0002.json",
            },
            "_local_s3_root": str(EXAMPLE_DIR),
        })

        urgency_result = infer_urgency.main({
            "features": features_result["features"],
            "model": {
                "bucket": "demo-bucket",
                "key": "model/pneumothorax_rf.joblib",
            },
            "_local_s3_root": str(EXAMPLE_DIR),
        })

        urgency = urgency_result["urgency"]
        self.assertEqual(features_result["study_id"], "ST-SIIM-0002")
        self.assertTrue(features_result["series_complete"])
        self.assertEqual(features_result["stats"]["slice_count"], 1)
        self.assertEqual(urgency["study_id"], "ST-SIIM-0002")
        self.assertIn(urgency["class"], {"routine", "critical"})
        self.assertGreaterEqual(urgency["confidence"], 0.0)
        self.assertLessEqual(urgency["confidence"], 1.0)
        self.assertEqual(urgency["model"]["version"], "rf-baseline-v1")


if __name__ == "__main__":
    unittest.main()
