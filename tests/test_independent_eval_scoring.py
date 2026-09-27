import json
import tempfile
import unittest
from pathlib import Path

from scripts.score_independent_eval import (
    TASKS, protocol_values, summarize, validate_predictions,
)


class IndependentEvalScoringTests(unittest.TestCase):
    def test_fixed_thresholds_are_all_required(self):
        rows = [{"id": f"{task}_{i}", "task": task}
                for task in TASKS for i in range(100)]
        slices = {"synthetic": [r["id"] for r in rows]}
        baseline = {}
        candidate = {}
        for task in TASKS:
            old_n = 80 if task in ("existence", "attribute") else 65
            for i in range(100):
                identity = f"{task}_{i}"
                baseline[identity] = {"correct": i < old_n, "parsed": True,
                                      "f1": float(i < old_n)}
                candidate[identity] = {"correct": i < old_n + 5, "parsed": True,
                                       "f1": float(i < old_n + 5)}
        bootstrap = {"macro_gain_ci": [0.001, 0.10],
                     "per_task_gain_ci": {task: [0.001, 0.10] for task in TASKS}}
        criteria = {"macro_gain_min": .05, "ci_lower_gt": 0,
                    "per_task_regression_max": .03,
                    "candidate_floors": {"existence": .85, "counting": .70,
                                         "attribute": .85, "listing": .70}}
        result = summarize(rows, slices, baseline, candidate, bootstrap, criteria)
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["macro"]["gain"], .05)

        bootstrap["macro_gain_ci"] = [0.0, 0.10]
        result = summarize(rows, slices, baseline, candidate, bootstrap, criteria)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["acceptance_checks"]["macro_gain_ci_lower_gt"])

        bootstrap["macro_gain_ci"] = [0.001, 0.10]
        for i in range(76, 80):
            candidate[f"existence_{i}"]["correct"] = False
        result = summarize(rows, slices, baseline, candidate, bootstrap, criteria)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["acceptance_checks"]["candidate_floors"])

    def test_prediction_ids_must_be_complete_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps({"id": "a", "prediction": "是"}) + "\n",
                            encoding="utf-8")
            self.assertEqual(validate_predictions(path, {"a"}), {"a": "是"})
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                validate_predictions(path, {"a", "b"})
            path.write_text((json.dumps({"id": "a", "prediction": "是"}) + "\n") * 2,
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                validate_predictions(path, {"a"})

    def test_bootstrap_settings_are_frozen(self):
        protocol = {"macro_gain_min": .05, "ci_lower_gt": 0,
                    "per_task_regression_max": .03,
                    "candidate_floors": dict.fromkeys(TASKS, .7),
                    "paired_bootstrap": {"iterations": 10000, "seed": 20260926,
                                         "resamples_per_task": 100,
                                         "stratify_by_task": True}}
        self.assertEqual(protocol_values(protocol)["macro_gain_min"], .05)
        protocol["paired_bootstrap"]["iterations"] = 100
        with self.assertRaisesRegex(ValueError, "paired_bootstrap"):
            protocol_values(protocol)


if __name__ == "__main__":
    unittest.main()
