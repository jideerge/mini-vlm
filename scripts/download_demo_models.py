"""Fetch pinned upstream weights for the public/local B/16 demo.

Only official CLIP and Qwen files are downloaded. The trained projector is the
small checkpoint shipped in ``demo/assets/checkpoint_best.pt``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    {
        "repo": "openai/clip-vit-base-patch16",
        "revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "directory": ROOT / "checkpoints/pretrained/clip-vit-base-patch16",
        "files": (
            "config.json", "preprocessor_config.json", "pytorch_model.bin",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt",
        ),
        "weight": "pytorch_model.bin",
        "sha256": "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0",
    },
    {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "directory": ROOT / "checkpoints/pretrained/qwen2.5-0.5b-instruct",
        "files": (
            "config.json", "generation_config.json", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        ),
        "weight": "model.safetensors",
        "sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_pretrained(*, verify_only: bool = False) -> None:
    for spec in SPECS:
        target = spec["directory"]
        missing = [name for name in spec["files"] if not (target / name).is_file()]
        weight = target / spec["weight"]
        if weight.is_file() and sha256(weight) != spec["sha256"]:
            raise RuntimeError(f"Upstream weight hash mismatch: {weight}")
        if missing:
            if verify_only:
                raise FileNotFoundError(f"Missing files for {spec['repo']}: {missing}")
            from huggingface_hub import snapshot_download

            print(f"Downloading {spec['repo']} at {spec['revision']} ...", flush=True)
            snapshot_download(
                repo_id=spec["repo"], revision=spec["revision"],
                local_dir=target, allow_patterns=list(spec["files"]),
            )
        still_missing = [name for name in spec["files"] if not (target / name).is_file()]
        if still_missing or sha256(weight) != spec["sha256"]:
            raise RuntimeError(f"Pinned upstream model verification failed: {spec['repo']}")
        print(f"Verified {spec['repo']} ({spec['revision']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Check local files and hashes without downloading")
    args = parser.parse_args()
    ensure_pretrained(verify_only=args.verify_only)


if __name__ == "__main__":
    main()
