"""Regression tests for image-only controls; no model weights or GPU needed."""
import copy
import unittest
from types import SimpleNamespace

import torch

from scripts.evaluate import mismatched_visual_records
from training.trainer import Trainer


class VisualControlTests(unittest.TestCase):
    def make_trainer(self, records):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = SimpleNamespace(batch_size=4)
        trainer.device = torch.device("cpu")
        trainer.model = SimpleNamespace(
            eval=lambda: None, train=lambda: None,
            projector=torch.nn.Linear(1, 1),
        )
        trainer.feature_cache = {
            r["image"]: torch.tensor([float(r["answer"])]) for r in records
        }
        self.prompts = []
        self.visual_inputs = []

        def make_batch(chunk):
            self.prompts.extend(r["question"] for r in chunk)
            return {
                "input_ids": torch.zeros(len(chunk), 1, dtype=torch.long),
                "attention_mask": torch.ones(len(chunk), 1),
                "answer_starts": [1] * len(chunk),
            }, trainer._features_for(chunk)

        def generate_text(**kwargs):
            values = kwargs["visual_features"].flatten().tolist()
            self.visual_inputs.extend(values)
            return [str(int(x)) for x in values]

        trainer.make_batch = make_batch
        trainer.model.generate_text = generate_text
        return trainer

    @staticmethod
    def records(n):
        return [{"id": str(i), "image": str(i % 3), "task": "counting",
                 "question": f"question-{i}", "answer": str(i % 3)}
                for i in range(n)]

    def test_control_changes_only_visual_input_and_keeps_all_205_samples(self):
        records = self.records(205)
        before = copy.deepcopy(records)
        visual = mismatched_visual_records(records)
        self.assertTrue(all(r["image"] != v["image"]
                            for r, v in zip(records, visual)))
        trainer = self.make_trainer(records)
        original = trainer.evaluate_generation(records)
        self.prompts.clear()
        self.visual_inputs.clear()
        control = trainer.evaluate_generation(records, visual_records=visual)
        self.assertEqual(original["exact"], 1.0)
        self.assertEqual(control["exact"], 0.0)
        self.assertEqual(original["n"], 205)
        self.assertEqual(control["n"], 205)
        self.assertEqual(original["content"]["correct"], 1.0)
        self.assertEqual(control["content"]["correct"], 0.0)
        self.assertEqual(control["content"]["n"], 205)
        self.assertEqual(self.prompts, [r["question"] for r in records])
        self.assertEqual(self.visual_inputs, [float(r["answer"]) for r in visual])
        self.assertEqual([d["reference"] for d in control["details"]],
                         [r["answer"] for r in records[:200]])
        self.assertEqual(records, before)

    def test_max_samples_slices_both_inputs_together(self):
        records = self.records(9)
        visual = mismatched_visual_records(records)
        trainer = self.make_trainer(records)
        result = trainer.evaluate_generation(records, max_samples=5,
                                             visual_records=visual)
        self.assertEqual(result["n"], 5)
        self.assertEqual(self.prompts, [r["question"] for r in records[:5]])
        self.assertEqual(self.visual_inputs, [float(r["answer"]) for r in visual[:5]])

    def test_full_details_keep_every_prediction(self):
        records = self.records(205)
        result = self.make_trainer(records).evaluate_generation(records, detail_limit=None)
        self.assertEqual(len(result["details"]), 205)
        self.assertEqual(result["details"][-1]["id"], records[-1]["id"])
        self.assertTrue(result["details"][-1]["content"]["correct"])

    def test_rejects_unpaired_lengths(self):
        records = self.records(3)
        with self.assertRaises(ValueError):
            self.make_trainer(records).evaluate_generation(records, visual_records=[])

    def test_rejects_empty_or_single_distinct_image(self):
        for records in ([], self.records(1), [self.records(1)[0]] * 3):
            with self.subTest(records=records), self.assertRaises(ValueError):
                mismatched_visual_records(records)


if __name__ == "__main__":
    unittest.main()
