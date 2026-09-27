"""Resolve local or portable image paths without changing the dataset JSONL."""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def resolve_image_path(image: str, image_root: str | Path | None = None) -> Path:
    original = Path(image)
    if original.is_file():
        return original
    root = image_root or os.environ.get("MINIVLM_IMAGE_ROOT")
    if root:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"图片根目录不存在：{root}")
        # Accept ordinary relative paths first; for saved Windows absolute paths
        # on Linux, use the basename in a flat exported image directory.
        if not original.is_absolute():
            relative = root / original
            if relative.is_file():
                return relative
        candidate = root / PureWindowsPath(image).name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"图片不存在：{candidate}（原始路径：{image}）")
    raise FileNotFoundError(
        f"图片不存在：{image}；迁移环境时请设置 MINIVLM_IMAGE_ROOT 为平铺 JPEG 目录"
    )
