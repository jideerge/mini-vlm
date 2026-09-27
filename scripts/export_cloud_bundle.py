#!/usr/bin/env python
"""Build the minimal, self-contained B/16 cloud training input archive."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/cloud_b16_spatial_clean_input.zip"
MODEL_DIRS = (
    "checkpoints/pretrained/clip-vit-base-patch16",
    "checkpoints/pretrained/qwen2.5-0.5b-instruct",
)
DATA_FILES = (
    "data/processed/spatial_clean_9333.jsonl",
    "data/processed/spatial_clean_smoke_1k.jsonl",
    "data/processed/splits.json",
)
SOURCE_DIRS = ("models", "datasets", "training", "evaluation", "scripts", "tests")
OTHER_FILES = (
    "configs/model_clip_b16_qwen05.yaml",
    "requirements-cloud.txt",
    "scripts/run_cloud_b16.sh",
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {OUTPUT}")
    rows = [json.loads(line) for line in (ROOT / DATA_FILES[0]).read_text(encoding="utf-8").splitlines() if line]
    images = {PureWindowsPath(row["image"]).name: Path(row["image"]) for row in rows}
    assert len(images) == len({row["image"] for row in rows}) == 6400, "image basename collision"
    assert all(path.is_file() for path in images.values()), "source image missing"
    included: dict[str, Path] = {}
    def add(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        if relative in included and included[relative] != path:
            raise ValueError(f"duplicate archive path: {relative}")
        included[relative] = path
    for folder in SOURCE_DIRS:
        for path in sorted((ROOT / folder).glob("*.py")):
            add(path)
    for name in (*DATA_FILES, *OTHER_FILES):
        add(ROOT / name)
    for folder in MODEL_DIRS:
        for path in sorted((ROOT / folder).iterdir()):
            if path.is_file() and not path.name.startswith("."):
                add(path)
    manifest = {
        "experiment": "B/16 projector-only on spatial_clean_9333",
        "source_model": "openai/clip-vit-base-patch16",
        "source_revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "records": len(rows),
        "images": len(images),
        "image_bytes": sum(path.stat().st_size for path in images.values()),
        "data_sha256": file_hash(ROOT / DATA_FILES[0]),
        "weights_sha256": file_hash(ROOT / MODEL_DIRS[0] / "pytorch_model.bin"),
        "nonimage_paths": sorted(included),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print("archive:", OUTPUT, "images:", len(images), "other files:", len(included), flush=True)
    with zipfile.ZipFile(OUTPUT, "w", allowZip64=True, compression=zipfile.ZIP_STORED) as zf:
        for name, path in sorted(included.items()):
            zf.write(path, name)
        zf.writestr("report/figures/", b"")
        for count, (name, path) in enumerate(sorted(images.items()), 1):
            zf.write(path, f"images/{name}")
            if count % 1000 == 0:
                print("packed images:", count, flush=True)
        zf.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(OUTPUT) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        assert len(zf.namelist()) == len(included) + len(images) + 2
    digest = file_hash(OUTPUT)
    # Write bytes to preserve LF on Windows for Linux sha256sum -c.
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_bytes(
        (digest + "  " + OUTPUT.name + "\n").encode("ascii")
    )
    print("verified ZIP CRC, bytes:", OUTPUT.stat().st_size, "SHA-256:", digest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
