"""No-inference tests for the sealed holdout prediction exporter."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import export_independent_predictions as export


class IndependentExportTests(unittest.TestCase):
    def test_project_paths_cannot_fall_back_or_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "data" / "independent_eval_v1" / "images"
            images.mkdir(parents=True)
            picture = images / "sample.jpg"
            picture.write_bytes(b"not a model input")
            with patch.object(export, "ROOT", root):
                self.assertEqual(export.project_path(
                    "data/independent_eval_v1/images/sample.jpg", within=images), picture)
                for raw in (
                    "../outside.jpg", "data/../outside.jpg", "C:/outside.jpg",
                    "data\\independent_eval_v1\\images\\sample.jpg", "/outside.jpg",
                ):
                    with self.subTest(raw=raw), self.assertRaises((ValueError, FileNotFoundError)):
                        export.project_path(raw, within=images)

    def test_freeze_requires_selected_checkpoint_and_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = root / "outputs" / "b16_four_task_pilot"
            pilot.mkdir(parents=True)
            checkpoint = pilot / "checkpoint_selected.pt"
            checkpoint.write_bytes(b"frozen weights")
            selected = checkpoint.relative_to(root).as_posix()
            freeze_path = pilot / "candidate_freeze.json"
            results_path = pilot / "results.json"
            freeze_path.write_text(json.dumps({
                "checkpoint": selected,
                "checkpoint_sha256": export.sha256(checkpoint),
                "protocol_sha256": "protocol digest", "selected_on": "old validation",
            }), encoding="utf-8")
            results_path.write_text(json.dumps({
                "selected_checkpoint": selected, "selected_step": 256,
                "protocol_sha256": "protocol digest",
                "independent_holdout_used": False,
                "eligible_steps": [256],
                "validation": {"256": {"checkpoint_sha256": export.sha256(checkpoint)}},
            }), encoding="utf-8")
            with (patch.object(export, "ROOT", root),
                  patch.object(export, "PILOT_DIR", pilot),
                  patch.object(export, "FREEZE_PATH", freeze_path),
                  patch.object(export, "RESULTS_PATH", results_path)):
                _, actual, result = export.load_freeze("protocol digest")
                self.assertEqual(actual, checkpoint)
                self.assertEqual(result["selected_step"], 256)
                checkpoint.write_bytes(b"changed weights")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    export.load_freeze("protocol digest")

    def test_publish_is_atomic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "independent_eval_v1"
            with patch.object(export, "OUTPUT_ROOT", output):
                result = export.publish("baseline", [{"id": "old-validation-id", "prediction": "猫"}],
                                        {"role": "baseline"})
                self.assertTrue(result.is_file())
                provenance = json.loads((result.parent / "provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["predictions_sha256"], export.sha256(result))
                with self.assertRaises(FileExistsError):
                    export.publish("baseline", [], {})
                with patch.object(export.os, "rename", side_effect=OSError("simulated commit failure")):
                    with self.assertRaises(OSError):
                        export.publish("candidate", [{"id": "old-validation-id", "prediction": "猫"}], {})
                self.assertFalse((output / "candidate").exists())
                self.assertEqual(list(output.glob(".candidate.partial-*")), [])

    def test_empty_answer_prompt_has_only_eos_after_answer_start(self):
        import torch
        from PIL import Image
        from transformers import AutoTokenizer
        from datasets.collator import MultipleImageCollator
        from datasets.dataset import JsonlVLDataset

        tokenizer = AutoTokenizer.from_pretrained(
            export.ROOT / "checkpoints/pretrained/qwen2.5-0.5b-instruct", local_files_only=True
        )
        class DummyProcessor:
            def __call__(self, *, images, return_tensors):
                return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}

        collator = MultipleImageCollator(processor=DummyProcessor(), tokenizer=tokenizer,
                                         num_image_token=196)
        old_style_question = JsonlVLDataset.build_question("图中一共有几个猫？")
        ids, labels, answer_start = collator.build_text_and_labels(old_style_question, "")
        ids, labels, answer_start = collator.expand(ids, labels, answer_start)
        self.assertEqual(ids[answer_start:], [tokenizer.eos_token_id])
        self.assertEqual(labels[answer_start:], [tokenizer.eos_token_id])
        self.assertEqual(ids.count(collator.image_token_id), 196)
        batch = collator([{"image": Image.new("RGB", (30, 20)),
                           "question": old_style_question, "answer": ""}])
        valid = int(batch["attention_mask"][0].sum())
        start = batch["answer_starts"][0]
        self.assertEqual(batch["input_ids"][0, start:valid].tolist(),
                         [tokenizer.eos_token_id])


if __name__ == "__main__":
    unittest.main()
