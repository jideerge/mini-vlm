#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/dataset.py —— Mini-VLM 的 Dataset

一条样本在 JSONL 里长这样：

    {"id": "...", "image": "D:\\\\...\\\\000005.jpg", "task": "counting",
     "question": "图中一共有几个椅子？", "answer": "图中有五个椅子。", "meta": {...}}

Dataset 负责把它变成 collator 能吃的形式：图片 + 带占位符的问题 + 答案。

两个来源都支持：
  * 直接读图片（`from_jsonl`）
  * 用 Week 2 缓存的视觉特征（`from_feature_cache`）—— 训练时不加载 CLIP，省显存 + 提速
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from datasets.image_paths import resolve_image_path
from torch.utils.data import Dataset

IMAGE_PLACEHOLDER = "<|image_pad|>"


class JsonlVLDataset(Dataset):
    """从 JSONL 读取的多模态指令数据集。

    build_question 会把 <|image_pad|> 拼到问题前面：
        "<|image_pad|>\\n图中一共有几个椅子？"
    注意**只写一个占位符**，由 collator 展开成 49 个（见 collator 文档）。
    """

    def __init__(
        self,
        records: list[dict],
        image_root: str | Path | None = None,
        system_prompt: str | None = None,
        image_cache: bool = True,
        max_records: int | None = None,
    ) -> None:
        self.records = records[:max_records] if max_records else records
        self.image_root = Path(image_root) if image_root else None
        self.system_prompt = system_prompt
        self.image_cache = image_cache
        self._cache: dict[int, Image.Image] = {}

    # ------------------------------------------------------------------
    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        image_root: str | Path | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> "JsonlVLDataset":
        p = Path(path)
        records: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
        return cls(records, image_root=image_root, max_records=limit, **kwargs)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def resolve_image_path(self, rel: str) -> Path:
        return resolve_image_path(rel, self.image_root)

    def load_image(self, idx: int) -> Image.Image:
        if self.image_cache and idx in self._cache:
            return self._cache[idx]
        rec = self.records[idx]
        img = Image.open(self.resolve_image_path(rec["image"])).convert("RGB")
        if self.image_cache:
            self._cache[idx] = img
        return img

    @staticmethod
    def build_question(question: str, with_image: bool = True) -> str:
        if not with_image:
            return question
        return f"{IMAGE_PLACEHOLDER}\n{question}"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        return {
            "index": idx,
            "id": rec.get("id", str(idx)),
            "task": rec.get("task"),
            "image": self.load_image(idx),
            "question": self.build_question(rec["question"]),
            "answer": rec["answer"],
            "system": self.system_prompt,
            "raw_question": rec["question"],
        }

    # ------------------------------------------------------------------
    def split_by_task(self, tasks: list[str], per_task: int | None = None) -> list[int]:
        """按任务均衡挑选索引——用于过拟合测试（避免 100 条全是同一类）。"""
        out: list[int] = []
        for t in tasks:
            idxs = [i for i, r in enumerate(self.records) if r.get("task") == t]
            out.extend(idxs[:per_task] if per_task else idxs)
        return sorted(out)


class SubsetIndices(Dataset):
    """按索引取子集（保持 JsonlVLDataset 的行为）。"""

    def __init__(self, base: JsonlVLDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.base[self.indices[i]]


# ----------------------------------------------------------------------
# 特征缓存读取（Week 2 的产物）
# ----------------------------------------------------------------------
class FeatureCacheReader:
    """读取 data/processed/features_w2 的分片缓存。

    缓存内容：
        patch_features [N,49,768] fp16  <- VLM 的 Projector 吃这个（投影前）
        cls_feature    [N,768]    fp16
        image_embeds   [N,512]    fp16  <- 零样本/检索用（投影后）

    索引 index.json 里每条记录有 {shard, offset, path, label}。
    """

    def __init__(self, cache_dir: str | Path) -> None:
        import torch

        self.dir = Path(cache_dir)
        meta = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.entries: list[dict] = meta["entries"]
        self.by_path: dict[str, int] = {e["path"]: i for i, e in enumerate(self.entries)}
        self._shards: dict[int, dict] = {}
        self._torch = torch

    def _shard(self, idx: int) -> dict:
        s = self.entries[idx]["shard"]
        if s not in self._shards:
            self._shards[s] = self._torch.load(
                self.dir / f"shard_{s:04d}.pt", map_location="cpu"
            )
        return self._shards[s]

    def get_patch_features(self, image_path: str):
        idx = self.by_path[image_path]
        e = self.entries[idx]
        return self._shard(idx)["patch_features"][e["offset"]]

    def get_image_embed(self, image_path: str):
        idx = self.by_path[image_path]
        e = self.entries[idx]
        return self._shard(idx)["image_embeds"][e["offset"]]

    def __contains__(self, image_path: str) -> bool:
        return image_path in self.by_path

    def __len__(self) -> int:
        return len(self.entries)
