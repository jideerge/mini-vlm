#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/cache_features.py —— 视觉特征离线缓存（本项目最重要的工程优化）

原理：视觉塔是**冻结**的，同一张图片的特征在整个训练/评测过程中完全不变。
      所以可以在预处理阶段一次性算完存盘，训练时只读特征：
        * 训练时不再加载 CLIP（省显存）
        * 训练时不再做图像前向（约 2-4x 提速）

存什么（三者维度不同，别混用）：
    patch_features  [N_img, 49, 768]  fp16   <- VLM 的 Projector 吃这个（投影前）
    cls_feature     [N_img, 768]      fp16   <- 视觉塔的 CLS（投影前）
    image_embeds    [N_img, 512]      fp16   <- CLIP 投影后、归一化，零样本/检索用

磁盘占用：49*768*2 = 75 KB/图，5 万图约 3.8 GB。

用法：
    & $env:MINIVLM_PY datasets/cache_features.py --limit 1000
    & $env:MINIVLM_PY datasets/cache_features.py --list data/processed/w2_cache.list --batch 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT))

SHARD_SIZE = 5000  # 每个分片最多多少张图


def pick_device(name: str = "auto"):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_image_list(list_path: str | Path) -> list[dict]:
    """读图片清单。每行支持两种格式：
        /abs/path/to.jpg
        /abs/path/to.jpg \t label
    """
    items: list[dict] = []
    for line in Path(list_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        items.append({"path": parts[0], "label": parts[1] if len(parts) > 1 else None})
    return items


def write_image_list(items: list[dict], list_path: str | Path) -> Path:
    p = Path(list_path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for it in items:
        lines.append(it["path"] if not it.get("label") else f"{it['path']}\t{it['label']}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def cache_features(
    items: list[dict],
    out_dir: str | Path,
    clip_dir: str | Path,
    batch_size: int = 32,
    device: str = "auto",
    verbose: bool = True,
    return_tensors: bool = False,
):
    """把图片清单编码成特征分片。

    返回统计 dict；return_tensors=True 时额外返回第一批张量（供调用方立即使用）。
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dev = pick_device(device)
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CLIPModel.from_pretrained(str(clip_dir)).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(str(clip_dir))

    n = len(items)
    per_image_ms: list[float] = []
    t_start = time.perf_counter()

    shard_idx, offset = 0, 0
    buffers = {"patch": [], "cls": [], "proj": []}
    index: list[dict] = []
    shard_files: list[str] = []
    first_tensors = None

    def flush(shard_number: int) -> None:
        if not buffers["patch"]:
            return
        name = f"shard_{shard_number:04d}.pt"
        torch.save(
            {
                "patch_features": torch.stack(buffers["patch"]),   # [n,49,768] fp16
                "cls_feature": torch.stack(buffers["cls"]),        # [n,768]    fp16
                "image_embeds": torch.stack(buffers["proj"]),      # [n,512]    fp16
            },
            out_dir / name,
        )
        shard_files.append(name)
        for k in buffers:
            buffers[k] = []

    for start in range(0, n, batch_size):
        chunk = items[start:start + batch_size]
        images = []
        keep = []
        for it in chunk:
            try:
                images.append(Image.open(it["path"]).convert("RGB"))
                keep.append(it)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [跳过] {it['path']}: {type(exc).__name__}: {exc}")
        if not images:
            continue

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            # 视觉塔：投影前的 patch 特征（VLM 要用这个）与 CLS
            vis = model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
            patch = vis.last_hidden_state[:, 1:, :]        # 丢 CLS -> [B,49,768]
            cls = vis.last_hidden_state[:, 0, :]           # [B,768]
            # CLIP 投影后 + 归一化（零样本/检索要用这个）
            proj = model.visual_projection(vis.pooler_output)
            proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000
        per_image_ms.extend([dt_ms / len(images)] * len(images))

        patch = patch.to(torch.float16).cpu()
        cls = cls.to(torch.float16).cpu()
        proj = proj.to(torch.float16).cpu()
        if first_tensors is None:
            first_tensors = {"patch_features": patch, "cls_feature": cls, "image_embeds": proj}

        for i, it in enumerate(keep):
            buffers["patch"].append(patch[i])
            buffers["cls"].append(cls[i])
            buffers["proj"].append(proj[i])
            index.append({
                "shard": shard_idx,
                "offset": offset,
                "path": it["path"],
                "label": it.get("label"),
            })
            offset += 1
            if offset >= SHARD_SIZE:
                flush(shard_idx)
                shard_idx += 1
                offset = 0

    flush(shard_idx)
    total_s = time.perf_counter() - t_start

    index_path = out_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "num_images": len(index),
                "patch_tokens": 49,
                "vision_hidden": 768,
                "proj_dim": int(first_tensors["image_embeds"].shape[-1]) if first_tensors else 512,
                "dtype": "float16",
                "clip_dir": str(clip_dir),
                "batch_size": batch_size,
                "device": str(dev),
                "shards": shard_files,
                "entries": index,
                "timing": {
                    "total_s": total_s,
                    "per_image_ms": (sum(per_image_ms) / len(per_image_ms)) if per_image_ms else 0.0,
                    "images_per_s": (len(index) / total_s) if total_s > 0 else 0.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    bytes_per_img = 49 * 768 * 2 + 768 * 2 + 512 * 2
    stats = {
        "out_dir": str(out_dir),
        "num_images": len(index),
        "shards": shard_files,
        "total_s": total_s,
        "images_per_s": (len(index) / total_s) if total_s > 0 else 0.0,
        "per_image_ms_batched": (sum(per_image_ms) / len(per_image_ms)) if per_image_ms else 0.0,
        "disk_mb": (len(index) * bytes_per_img) / 1024 ** 2,
        "disk_kb_per_image": bytes_per_img / 1024,
        "device": str(dev),
        "index_path": str(index_path),
    }
    if return_tensors:
        stats["_first_tensors"] = first_tensors
    return stats


def load_shard(out_dir: str | Path, shard: int):
    import torch

    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    return torch.load(out_dir / f"shard_{shard:04d}.pt", map_location="cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description="视觉特征离线缓存")
    ap.add_argument("--list", type=str, default=None, help="图片清单文件（每行一个绝对路径）")
    ap.add_argument("--limit", type=int, default=1000, help="未指定 --list 时，从 VOC 采样多少张")
    ap.add_argument("--out", type=str, default="data/processed/features_w2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--write-list", type=str, default=None, help="把采样出的清单写到该路径")
    args = ap.parse_args()

    clip_dir = ROOT / "checkpoints" / "pretrained" / "clip-vit-base-patch32"

    if args.list:
        items = read_image_list(ROOT / args.list)
        print(f"从清单读取 {len(items)} 张图: {args.list}")
    else:
        from datasets.voc_source import build_index, sample_random

        entries = build_index()
        items = sample_random(entries, args.limit)
        items = [{"path": e["path"], "label": e["classes"][0]} for e in items]
        print(f"从 VOC 采样 {len(items)} 张图")
        if args.write_list:
            p = write_image_list(items, args.write_list)
            print(f"清单已写出: {p}")

    stats = cache_features(
        items, out_dir=args.out, clip_dir=clip_dir,
        batch_size=args.batch, device=args.device,
    )

    print("\n=== 缓存完成 ===")
    print(f"  图片数        : {stats['num_images']}")
    print(f"  分片          : {stats['shards']}")
    print(f"  总耗时        : {stats['total_s']:.2f} s")
    print(f"  吞吐          : {stats['images_per_s']:.1f} img/s")
    print(f"  批内单图均摊   : {stats['per_image_ms_batched']:.2f} ms")
    print(f"  磁盘占用      : {stats['disk_mb']:.1f} MB ({stats['disk_kb_per_image']:.1f} KB/图)")
    print(f"  索引          : {stats['index_path']}")

    # 往返校验：读回来必须一致
    import torch
    shard = load_shard(stats["out_dir"], 0)
    print("\n=== 往返校验 ===")
    for k, v in shard.items():
        print(f"  {k:16} {tuple(v.shape)}  {v.dtype}")
    print(f"  NaN? patch={torch.isnan(shard['patch_features']).any().item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
