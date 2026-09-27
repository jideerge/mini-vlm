#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate horizontal-flip consistency on the clean spatial test subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageOps

from datasets.collator import MultipleImageCollator
from datasets.image_paths import resolve_image_path
from models.multimodal_model import MiniVLM, config_from_yaml
from scripts.evaluate import load_records, split_records
from training.trainer import Trainer, TrainingConfig


def swap_direction(value: str) -> str:
    """Exchange exactly one left/right label without changing the question."""
    if value.count("左边") + value.count("右边") != 1:
        raise ValueError(f"expected one direction in {value!r}")
    return value.replace("左边", "{LEFT}").replace("右边", "左边").replace("{LEFT}", "右边")


def mirrored_record(record: dict) -> dict:
    answer = swap_direction(record["answer"])
    meta = dict(record["meta"])
    meta["answer"] = swap_direction(meta["answer"])
    if "left" in meta and "right" in meta:
        meta["left"], meta["right"] = meta["right"], meta["left"]
    with Image.open(resolve_image_path(record["image"])) as source:
        width = source.width
    for key in ("x_subject", "x_other"):
        if key in meta:
            meta[key] = width - meta[key]
    return {**record, "image": "mirror::" + record["image"], "answer": answer, "meta": meta}


def load_one(record: dict) -> Image.Image:
    path = record["image"]
    mirror = path.startswith("mirror::")
    if mirror:
        path = path[len("mirror::"):]
    with Image.open(resolve_image_path(path)) as source:
        image = source.convert("RGB")
    return ImageOps.mirror(image) if mirror else image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/b16_spatial_flip/eval_spatial_flip.json"))
    ap.add_argument("--gen-batch", type=int, default=4)
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    data = ROOT / "data/processed/spatial_clean_9333.jsonl"
    parts = split_records(load_records(data), ROOT / "data/processed/splits.json")
    records = [r for r in parts["test"] if r["task"] == "spatial"]
    if args.limit is not None:
        records = records[:args.limit]
    flipped = [mirrored_record(r) for r in records]

    ckpt = ROOT / "outputs/b16_spatial_clean_cloud/checkpoint_best.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "lora" in payload or "trainable_extra" in payload:
        raise ValueError("expected a projector-only checkpoint")
    if payload.get("global_step") != 840:
        raise ValueError(f"unexpected checkpoint step: {payload.get('global_step')}")

    cfg = config_from_yaml(ROOT / "configs/model_clip_b16_qwen05.yaml")
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniVLM(cfg)
    model.projector.load_state_dict(payload["projector"], strict=True)
    collator = MultipleImageCollator(
        processor=model.processor, tokenizer=model.tokenizer,
        max_length=cfg.max_seq_len, num_image_token=model.num_image_token,
    )
    tcfg = TrainingConfig(
        device=cfg.device, batch_size=args.gen_batch, cache_features_on_gpu=False,
        output_dir=str(args.out.parent),
    )
    trainer = Trainer(model, collator, [], [], load_one, tcfg)
    trainer.precompute_features(records)
    original = trainer.evaluate_generation(
        records, max_new_tokens=24, micro_batch=args.gen_batch, detail_limit=None,
    )
    trainer.precompute_features(flipped)
    feature_rms = [
        (trainer.feature_cache[record["image"]].float()
         - trainer.feature_cache[flipped_record["image"]].float())
        .square().mean().sqrt().item()
        for record, flipped_record in zip(records, flipped)
    ]
    if not all(value > 1e-4 for value in feature_rms):
        raise AssertionError("mirrored images did not produce distinct visual features")
    mirror = trainer.evaluate_generation(
        flipped, max_new_tokens=24, micro_batch=args.gen_batch, detail_limit=None,
    )

    cloud = json.loads(
        (ROOT / "outputs/b16_spatial_clean_cloud_eval/eval_test_per_task.json").read_text(encoding="utf-8")
    )
    cloud_spatial = {r["id"]: r for r in cloud["details"] if r["task"] == "spatial"}
    pairs = []
    for orig, flip in zip(original["details"], mirror["details"]):
        assert orig["id"] == flip["id"]
        old_pred = orig["content"]["predicted"]
        new_pred = flip["content"]["predicted"]
        pairs.append({
            "id": orig["id"], "expected": orig["content"]["expected"],
            "original_predicted": old_pred, "original_correct": orig["content"]["correct"],
            "mirrored_expected": flip["content"]["expected"],
            "mirrored_predicted": new_pred, "mirrored_correct": flip["content"]["correct"],
            "prediction_flipped": old_pred in ("左边", "右边") and new_pred == swap_direction(old_pred),
            "cloud_replay_match": orig["generated"] == cloud_spatial[orig["id"]]["generated"],
        })

    n = len(pairs)
    result = {
        "method": "horizontal_mirror_with_left_right_label_swap",
        "checkpoint_sha256": sha256(ckpt), "checkpoint_step": payload["global_step"],
        "device": cfg.device, "n": n,
        "original_correct": sum(p["original_correct"] for p in pairs),
        "mirrored_correct": sum(p["mirrored_correct"] for p in pairs),
        "prediction_flipped": sum(p["prediction_flipped"] for p in pairs),
        "both_correct": sum(p["original_correct"] and p["mirrored_correct"] for p in pairs),
        "cloud_replay_matches": sum(p["cloud_replay_match"] for p in pairs),
        "feature_rms_min": min(feature_rms),
        "feature_rms_median": statistics.median(feature_rms),
        "gpu_peak_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                        if cfg.device == "cuda" else None),
        "pairs": pairs,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: v for k, v in result.items() if k != "pairs"})
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

