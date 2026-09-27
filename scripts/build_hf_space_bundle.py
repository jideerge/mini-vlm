"""Build an explicit, minimal Hugging Face Space upload directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from demo.inference import CHECKPOINT_SHA256, _sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "dist" / "hf_space"
FILES = {
    "deploy/hf_space/SPACE_README.md": "README.md",
    "deploy/hf_space/requirements.txt": "requirements.txt",
    "deploy/hf_space/app.py": "app.py",
    "demo/inference.py": "demo/inference.py",
    "demo/__init__.py": "demo/__init__.py",
    "demo/assets/checkpoint_best.pt": "demo/assets/checkpoint_best.pt",
    "scripts/download_demo_models.py": "scripts/download_demo_models.py",
    "configs/model_clip_b16_qwen05.yaml": "configs/model_clip_b16_qwen05.yaml",
    "datasets/__init__.py": "datasets/__init__.py",
    "datasets/build_voc_vqa.py": "datasets/build_voc_vqa.py",
    "datasets/voc_source.py": "datasets/voc_source.py",
    "datasets/collator.py": "datasets/collator.py",
    "datasets/dataset.py": "datasets/dataset.py",
    "datasets/image_paths.py": "datasets/image_paths.py",
    "models/multimodal_model.py": "models/multimodal_model.py",
    "models/projector.py": "models/projector.py",
    "models/vision_encoder.py": "models/vision_encoder.py",
    "training/losses.py": "training/losses.py",
}


def build(destination: Path = DEFAULT_DEST) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Bundle already exists; choose an empty path: {destination}")
    checkpoint = ROOT / "demo/assets/checkpoint_best.pt"
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Bundled Projector checkpoint SHA-256 mismatch")
    if destination == ROOT or ROOT in destination.parents and destination == ROOT / "data":
        raise ValueError("Unsafe bundle destination")
    for source_rel, target_rel in FILES.items():
        source = ROOT / source_rel
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    bundle = build(args.destination)
    print(f"Created {bundle} with {len(FILES)} allowlisted files")


if __name__ == "__main__":
    main()
