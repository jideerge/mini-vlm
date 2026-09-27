#!/usr/bin/env python
"""Fail fast on a moved B/16 training bundle before GPU time is spent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
B16_SHA256 = "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"
DATA_SHA256 = "b58e78fec62465709164e36a0f497209317b0ebce3027a4c767d6ebf75a083bf"
EXPECTED_SPLITS = {"train": 8383, "val": 461, "test": 489}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the portable B/16 training inputs")
    parser.add_argument("--skip-gpu", action="store_true", help="Allow an archive-only check on a non-GPU machine")
    parser.add_argument("--source-paths", action="store_true", help="Check original source images before bundling")
    args = parser.parse_args()
    image_root = Path(os.environ.get("MINIVLM_IMAGE_ROOT", ROOT / "images"))
    if not args.source_paths:
        assert image_root.is_dir(), f"missing flat image directory: {image_root}"
    data = ROOT / "data/processed/spatial_clean_9333.jsonl"
    assert sha256(data) == DATA_SHA256, "clean dataset hash mismatch"
    records = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 9333
    by_id = {row["id"]: row for row in records}
    assert len(by_id) == len(records)
    splits = json.loads((ROOT / "data/processed/splits.json").read_text(encoding="utf-8"))["ids"]
    filtered = {part: [sample_id for sample_id in splits[part] if sample_id in by_id]
                for part in EXPECTED_SPLITS}
    assert {part: len(ids) for part, ids in filtered.items()} == EXPECTED_SPLITS
    assert len(set().union(*map(set, filtered.values()))) == len(records), "missing/overlapping split IDs"
    assert sum(by_id[i]["task"] == "spatial" for i in filtered["test"]) == 76
    paths = {row["image"] for row in records}
    names = [PureWindowsPath(path).name for path in paths]
    assert len(set(names)) == len(names), "image basenames collide in the flat export"
    missing = ([path for path in paths if not Path(path).is_file()]
               if args.source_paths else
               [name for name in names if not (image_root / name).is_file()])
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    clip = ROOT / "checkpoints/pretrained/clip-vit-base-patch16"
    qwen = ROOT / "checkpoints/pretrained/qwen2.5-0.5b-instruct"
    for model_dir, required in ((clip, ["config.json", "preprocessor_config.json", "pytorch_model.bin",
                                       "tokenizer_config.json", "vocab.json", "merges.txt"]),
                                (qwen, ["config.json", "model.safetensors", "tokenizer.json",
                                        "tokenizer_config.json"])):
        for name in required:
            file = model_dir / name
            assert file.is_file() and file.stat().st_size > 0, f"missing model asset: {file}"
    assert sha256(clip / "pytorch_model.bin") == B16_SHA256, "B/16 official weight hash mismatch"
    config = json.loads((clip / "config.json").read_text(encoding="utf-8"))["vision_config"]
    assert (config["image_size"], config["patch_size"], config["hidden_size"]) == (224, 16, 768)
    import yaml
    experiment = yaml.safe_load((ROOT / "configs/model_clip_b16_qwen05.yaml").read_text(encoding="utf-8"))
    assert experiment["data"]["num_image_token"] == experiment["vision"]["num_tokens"] == 196
    assert experiment["training"]["strategy"] == "projector_only"
    import torch
    print("versions:", {name: version(name) for name in ("torch", "transformers", "Pillow", "PyYAML")})
    if not args.skip_gpu:
        assert torch.cuda.is_available(), "CUDA GPU unavailable"
        props = torch.cuda.get_device_properties(0)
        print("GPU:", props.name, "VRAM GiB:", round(props.total_memory / 2**30, 2))
    print("verified:", len(records), "records;", len(paths), "images; splits", EXPECTED_SPLITS,
          "; B/16 weights and 196-token config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
