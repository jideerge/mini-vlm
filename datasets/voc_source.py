#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/voc_source.py —— PASCAL VOC 2007 图片来源（Week 2 实验用）

为什么用 VOC 而不是下载 COCO：
  * 本机 7.YOLOv1 项目下已有完整 VOC2007（约 9963 张真实照片 + XML 标注），零下载成本
  * 有真实类别标注 -> 可以诚实地量化"视觉编码器到底有没有效"
  * 图片尺寸多样（500x375 / 500x333 / ...）-> 正好用来做 resize/crop 敏感性实验

注意：
  * 该目录 ImageSets/ 是空的，所以类别只能从 Annotations/*.xml 里解析
  * VOC 图片是**外部资产**，不做复制、不做软链，用索引文件引用绝对路径
    （索引缓存在 data/processed/voc_index.json，首次解析后不再重复读 XML）
  * 路径可通过环境变量 MINIVLM_VOC_ROOT 覆盖
"""
from __future__ import annotations

import json
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VOC_ROOT = Path(
    os.environ.get(
        "MINIVLM_VOC_ROOT",
        r"D:\学习\深度学习\dl_project\7.YOLOv1\data\20260804-140453",
    )
)

# VOC 2007 的 20 个类别
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# 给 CLIP 用的自然语言模板（单个模板，避免 20xN 次文本编码开销）
VOC_PROMPTS = {
    "aeroplane": "a photo of an aeroplane",
    "bicycle": "a photo of a bicycle",
    "bird": "a photo of a bird",
    "boat": "a photo of a boat",
    "bottle": "a photo of a bottle",
    "bus": "a photo of a bus",
    "car": "a photo of a car",
    "cat": "a photo of a cat",
    "chair": "a photo of a chair",
    "cow": "a photo of a cow",
    "diningtable": "a photo of a dining table",
    "dog": "a photo of a dog",
    "horse": "a photo of a horse",
    "motorbike": "a photo of a motorbike",
    "person": "a photo of a person",
    "pottedplant": "a photo of a potted plant",
    "sheep": "a photo of a sheep",
    "sofa": "a photo of a sofa",
    "train": "a photo of a train",
    "tvmonitor": "a photo of a tv monitor",
}


def _parse_annotation(xml_path: Path) -> dict | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            continue
        objects.append({
            "name": name,
            "difficult": obj.findtext("difficult") == "1",
            "truncated": obj.findtext("truncated") == "1",
        })
    if not objects:
        return None
    bbox = root.find("bndbox")
    size = root.find("size")
    return {
        "objects": objects,
        "width": int(size.findtext("width")) if size is not None else None,
        "height": int(size.findtext("height")) if size is not None else None,
        "has_box": bbox is not None,
    }


def build_index(
    voc_root: Path | str = DEFAULT_VOC_ROOT,
    cache_path: Path | str | None = None,
    splits: tuple[str, ...] = ("VOCtrainval_06-Nov-2007", "VOCtest_06-Nov-2007"),
    force: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """扫描 VOC，返回每张图的 {path, split, stem, width, height, classes, single_class}。

    结果缓存到 cache_path（默认 data/processed/voc_index.json）。
    """
    voc_root = Path(voc_root)
    cache_path = Path(cache_path) if cache_path else ROOT / "data" / "processed" / "voc_index.json"

    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("voc_root") == str(voc_root) and cached.get("entries"):
                if verbose:
                    print(f"  索引命中缓存: {cache_path}  ({len(cached['entries'])} 条)")
                return cached["entries"]
        except (json.JSONDecodeError, KeyError):
            pass

    entries: list[dict] = []
    for split in splits:
        base = voc_root / split / "VOCdevkit" / "VOC2007"
        ann_dir, img_dir = base / "Annotations", base / "JPEGImages"
        if not ann_dir.exists():
            if verbose:
                print(f"  [跳过] 不存在: {ann_dir}")
            continue
        for xml_path in sorted(ann_dir.glob("*.xml")):
            img_path = img_dir / f"{xml_path.stem}.jpg"
            if not img_path.exists():
                continue
            info = _parse_annotation(xml_path)
            if info is None:
                continue
            # 简单起见：只取"非 difficult"的物体类别
            names = [o["name"] for o in info["objects"] if not o["difficult"]]
            if not names:
                continue
            uniq = sorted(set(names))
            entries.append({
                "path": str(img_path),
                "split": split,
                "stem": xml_path.stem,
                "width": info["width"],
                "height": info["height"],
                "classes": uniq,
                "n_objects": len(names),
                "single_class": len(uniq) == 1,
            })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"voc_root": str(voc_root), "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    if verbose:
        print(f"  索引已建立: {len(entries)} 条 -> {cache_path}")
    return entries


def class_distribution(entries: list[dict]) -> Counter:
    c: Counter = Counter()
    for e in entries:
        for name in e["classes"]:
            c[name] += 1
    return c


def sample_balanced(
    entries: list[dict],
    per_class: int = 5,
    classes: list[str] | None = None,
    single_class_only: bool = True,
    seed: int = 42,
    min_objects: int = 1,
    max_objects: int = 3,
) -> list[dict]:
    """按类别均衡采样。用于零样本分类与 kNN 验证。

    single_class_only=True 时只选"整张图只有一个类别"的图片，
    这样"图片属于哪个类别"这个问题本身才有唯一答案（否则指标无法解释）。
    """
    rng = random.Random(seed)
    classes = classes or VOC_CLASSES
    pool: dict[str, list[dict]] = {c: [] for c in classes}
    for e in entries:
        if single_class_only and not e["single_class"]:
            continue
        if not (min_objects <= e["n_objects"] <= max_objects):
            continue
        cls = e["classes"][0]
        if cls in pool:
            pool[cls].append(e)

    picked: list[dict] = []
    for c in classes:
        candidates = sorted(pool[c], key=lambda x: x["stem"])  # 先排序保证可复现
        rng.shuffle(candidates)
        picked.extend(candidates[:per_class])
    return picked


def sample_random(entries: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pool = sorted(entries, key=lambda x: x["stem"])
    rng.shuffle(pool)
    return pool[:n]


def summarize(entries: list[dict]) -> str:
    dist = class_distribution(entries)
    lines = [f"图片数: {len(entries)}", "", "类别分布:"]
    for name, cnt in dist.most_common():
        lines.append(f"  {name:<14} {cnt}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("VOC root:", DEFAULT_VOC_ROOT)
    entries = build_index(force=True)
    print(summarize(entries))
    print("\n单类别图片数:", sum(1 for e in entries if e["single_class"]))
    print("均衡采样 20 类 x 5 张 ->", len(sample_balanced(entries, per_class=5)))
