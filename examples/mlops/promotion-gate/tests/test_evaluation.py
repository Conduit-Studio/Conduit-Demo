"""Unit tests for the promotion-gate comparison logic.

    python -m unittest examples/mlops/promotion-gate/tests/test_evaluation.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run_code"))

from promotion_gate.evaluation import evaluate  # noqa: E402


class TestEvaluate(unittest.TestCase):
    def test_beats_when_better(self):
        out = evaluate({"accuracy": 0.96}, {"accuracy": 0.93})
        self.assertTrue(out["beats"])
        self.assertEqual(out["report"]["delta"], 0.03)

    def test_does_not_beat_when_worse(self):
        out = evaluate({"accuracy": 0.91}, {"accuracy": 0.93})
        self.assertFalse(out["beats"])
        self.assertEqual(out["report"]["delta"], -0.02)

    def test_tie_promotes_at_zero_min_delta(self):
        self.assertTrue(evaluate({"accuracy": 0.93}, {"accuracy": 0.93})["beats"])

    def test_min_delta_requires_real_improvement(self):
        # +0.005 improvement does not clear a 0.01 minimum
        self.assertFalse(evaluate({"accuracy": 0.935}, {"accuracy": 0.93}, min_delta=0.01)["beats"])
        self.assertTrue(evaluate({"accuracy": 0.945}, {"accuracy": 0.93}, min_delta=0.01)["beats"])

    def test_alternate_metric_name(self):
        out = evaluate({"mAP50-95": 0.71}, {"mAP50-95": 0.68}, metric="mAP50-95")
        self.assertTrue(out["beats"])
        self.assertEqual(out["report"]["metric"], "mAP50-95")

    def test_missing_metric_raises(self):
        with self.assertRaises(ValueError):
            evaluate({"f1": 0.9}, {"accuracy": 0.8}, metric="accuracy")

    def test_model_package_arn_carried_into_report(self):
        out = evaluate({"accuracy": 0.96}, {"accuracy": 0.93}, model_package_arn="arn:x")
        self.assertEqual(out["report"]["modelPackageArn"], "arn:x")


if __name__ == "__main__":
    unittest.main()
