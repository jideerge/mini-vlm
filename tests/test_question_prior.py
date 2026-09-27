"""Question-only prior must learn exclusively from training answers."""
import unittest
from scripts.evaluate_question_prior import QuestionPrior, most_common, paired_comparison


class QuestionPriorTests(unittest.TestCase):
    def test_frequency_and_order_independent_tie(self):
        train = [
            {"task": "counting", "question": "几只猫？", "answer": "两只"},
            {"task": "counting", "question": "几只猫？", "answer": "一只"},
            {"task": "counting", "question": "几只狗？", "answer": "三只"},
        ]
        prior = QuestionPrior(train)
        self.assertEqual(prior.predict({"task": "counting", "question": "几只猫？"}),
                         ("一只", "question", 1))
        self.assertEqual(QuestionPrior(list(reversed(train))).predict(
            {"task": "counting", "question": "几只猫？"}), ("一只", "question", 1))
        self.assertEqual(prior.predict({"task": "counting", "question": "几只鸟？"}),
                         ("一只", "task_fallback", 1))

    def test_test_label_cannot_change_prediction(self):
        prior = QuestionPrior([{"task": "existence", "question": "有猫吗？",
                               "answer": "有", "image": "train.jpg"}])
        a = {"task": "existence", "question": "有猫吗？", "answer": "有", "image": "a.jpg"}
        b = {**a, "answer": "没有", "image": "b.jpg"}
        self.assertEqual(prior.predict(a), prior.predict(b))

    def test_subset_can_reuse_original_split(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.evaluate_question_prior import split_records
        rows = [{"id": "train", "image": "a.jpg"}, {"id": "test", "image": "b.jpg"}]
        splits = {"ids": {"train": ["train", "filtered"], "val": [], "test": ["test"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps(splits), encoding="utf-8")
            selected = split_records(rows, path)
            self.assertEqual([r["id"] for r in selected["train"]], ["train"])
            self.assertEqual([r["id"] for r in selected["test"]], ["test"])

    def test_paired_comparison_requires_matching_ids_and_prompts(self):
        row = {"id": "a", "task": "existence", "question": "有猫吗？",
               "reference": "有", "content": {"correct": False}}
        model = {"split": "test", "details": [{**row, "content": {"correct": True}}]}
        paired = paired_comparison([row], model)["overall"]
        self.assertEqual(paired["model_only_correct"], 1)
        self.assertEqual(paired["model_minus_prior"], 1)
        with self.assertRaises(ValueError):
            paired_comparison([row], {**model, "details": [{**model["details"][0],
                                                          "question": "different"}]})
        with self.assertRaises(ValueError):
            paired_comparison([row], {**model, "details": []})


if __name__ == "__main__":
    unittest.main()
