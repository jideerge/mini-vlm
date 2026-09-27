#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/make_splits.py —— 生成固定的 train/val/test 划分

⚠️ **必须按 image_id 划分，不能按样本划分。**
同一张图会贡献多条问答（counting / existence / listing / ...），
如果按样本随机划分，同一张图的问题会同时出现在训练集和验证集里 ——
模型只要"记住这张图"，验证指标就会虚高。这是最典型的数据泄漏。

产物：data/processed/splits.json
    {
      "seed": 42,
      "by": "image_id",
      "images": {"train": [...], "val": [...], "test": [...]},
      "ids":    {"train": [...], "val": [...], "test": [...]},
      "stats":  {...}
    }

用法：
    & $env:MINIVLM_PY datasets/make_splits.py --data data/processed/debug_1k.jsonl
    & $env:MINIVLM_PY datasets/make_splits.py --data data/processed/baseline_10k.jsonl \
        --out data/processed/splits.json --ratios 0.9 0.05 0.05
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_splits(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
) -> dict:
    """按 image_id 划分。返回 splits 字典。"""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"三个比例之和必须为 1，收到 {ratios}")

    images = sorted({r["image"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_imgs = set(images[:n_train])
    val_imgs = set(images[n_train:n_train + n_val])
    test_imgs = set(images[n_train + n_val:])

    def collect(imgs: set[str]) -> list[dict]:
        return [r for r in records if r["image"] in imgs]

    train, val, test = collect(train_imgs), collect(val_imgs), collect(test_imgs)

    # 泄漏自检
    leak = (train_imgs & val_imgs) | (train_imgs & test_imgs) | (val_imgs & test_imgs)
    if leak:
        raise RuntimeError(f"发现图片跨 split 泄漏：{list(leak)[:5]}")

    def task_dist(recs: list[dict]) -> dict[str, int]:
        return dict(Counter(r.get("task", "?") for r in recs))

    splits = {
        "seed": seed,
        "by": "image_id",
        "ratios": list(ratios),
        "images": {
            "train": sorted(train_imgs),
            "val": sorted(val_imgs),
            "test": sorted(test_imgs),
        },
        "ids": {
            "train": [r["id"] for r in train],
            "val": [r["id"] for r in val],
            "test": [r["id"] for r in test],
        },
        "stats": {
            "n_images_total": n,
            "n_images": {"train": len(train_imgs), "val": len(val_imgs),
                         "test": len(test_imgs)},
            "n_samples": {"train": len(train), "val": len(val), "test": len(test)},
            "task_dist": {
                "train": task_dist(train), "val": task_dist(val),
                "test": task_dist(test),
            },
            "leak_check": "passed",
        },
    }
    return splits


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 train/val/test 划分（按 image_id）")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/processed/splits.json")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.9, 0.05, 0.05),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    records = load_jsonl(data_path)
    print(f"读取 {len(records)} 条样本 <- {data_path}")

    splits = make_splits(records, tuple(args.ratios), args.seed)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    s = splits["stats"]
    print(f"\n已写出 {out}")
    print(f"  图片数：train={s['n_images']['train']}  val={s['n_images']['val']}  "
          f"test={s['n_images']['test']}（合计 {s['n_images_total']}）")
    print(f"  样本数：train={s['n_samples']['train']}  val={s['n_samples']['val']}  "
          f"test={s['n_samples']['test']}")
    print(f"  泄漏自检：{s['leak_check']}")
    for part in ("train", "val", "test"):
        dist = s["task_dist"][part]
        total = sum(dist.values()) or 1
        pretty = "  ".join(f"{k}={v}({v/total:.0%})" for k, v in sorted(dist.items()))
        print(f"  {part:5s} 任务分布: {pretty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
