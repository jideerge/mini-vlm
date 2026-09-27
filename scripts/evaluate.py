#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/evaluate.py —— 按任务评测（含每个任务自己的多数类基线）

为什么必须按任务拆开：
  聚合指标会严重误导。本项目里 spatial 占 20% 的样本，
  而它"永远猜多数边"就有 51% 的正确率 ——
  只看总体精确匹配，会把"比瞎猜还差"的模型看成"还行"。

输出三样东西：
  1. 每个任务在**训练集**上的多数类基线（这才是该任务真正的及格线）
  2. 模型在指定 split 上每个任务的精确 / 前缀匹配
  3. 是否超过该任务自己的基线（✅ / ❌），以及一个"视觉特征错位"的对照实验

用法：
    & $env:MINIVLM_PY scripts/evaluate.py --out outputs/baseline_10k --split test
    & $env:MINIVLM_PY scripts/evaluate.py --out outputs/baseline_10k --split val --max-samples 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from datasets.collator import MultipleImageCollator  # noqa: E402
from datasets.image_paths import resolve_image_path  # noqa: E402
from models.multimodal_model import MiniVLM, config_from_yaml  # noqa: E402
from training.trainer import Trainer, TrainingConfig  # noqa: E402

TASKS = ["counting", "existence", "listing", "attribute", "spatial"]


def mismatched_visual_records(records: list[dict]) -> list[dict]:
    """将每张图片映射到下一张不同图片，同图多问时也不保留原配对。"""
    by_image = {r["image"]: r for r in records}
    images = list(by_image)
    if len(images) < 2:
        raise ValueError("视觉错位对照至少需要两张不同图片")
    replacement = {p: by_image[images[(i + 1) % len(images)]]
                   for i, p in enumerate(images)}
    return [replacement[r["image"]] for r in records]


def force_utf8() -> None:
    for n in ("stdout", "stderr"):
        s = getattr(sys, n, None)
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def load_records(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def split_records(records: list[dict], splits_path: Path) -> dict[str, list[dict]]:
    sp = json.loads(splits_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    return {part: [by_id[i] for i in sp["ids"][part] if i in by_id]
            for part in ("train", "val", "test")}


def task_baselines(train_recs: list[dict]) -> dict[str, float]:
    """每个任务在训练集上的多数类基线。

    spatial 特殊：答案模板几乎唯一，区别只在「左边 / 右边」，
    所以它的基线是"绝大多数边"的比例，而不是"最高频整句"。
    """
    out: dict[str, float] = {}
    for t in TASKS:
        sub = [r for r in train_recs if r.get("task") == t]
        if not sub:
            continue
        if t == "spatial":
            left = sum(1 for r in sub if r["answer"].endswith("左边。"))
            right = len(sub) - left
            out[t] = max(left, right) / len(sub)
        else:
            c = Counter(r["answer"] for r in sub)
            out[t] = c.most_common(1)[0][1] / len(sub)
    return out


def main() -> int:  # noqa: C901
    force_utf8()
    ap = argparse.ArgumentParser(description="按任务评测")
    ap.add_argument("--config", type=str, default="configs/model_clip_b32_qwen05.yaml")
    ap.add_argument("--data", type=str, default="data/processed/baseline_10k.jsonl")
    ap.add_argument("--splits", type=str, default="data/processed/splits.json")
    ap.add_argument("--out", type=str, default="outputs/baseline_10k")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--gen-batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        ap.error("--max-samples 必须为正整数")

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")

    records = load_records(ROOT / args.data)
    parts = split_records(records, ROOT / args.splits)
    train_recs = parts["train"]
    eval_recs = parts[args.split]
    # 只截取一次，原图和错图评测完全相同的问题与参考答案。
    if args.max_samples is not None:
        eval_recs = eval_recs[:args.max_samples]
    visual_recs = mismatched_visual_records(eval_recs)
    print(f"数据 {args.data}：train={len(train_recs)} val={len(parts['val'])} "
          f"test={len(parts['test'])}")
    print(f"评测 split = {args.split}（{len(eval_recs)} 条）")

    cfg = config_from_yaml(ROOT / args.config)
    cfg.device = device

    # ⚠️ 策略必须与训练时一致，否则 LoRA 权重会被静默丢弃。
    # 优先从 checkpoint 的 config 字段读回真正用过的 strategy，
    # 而不是相信当前配置文件里的值（实测踩过这个坑：
    #  用 projector_only 的配置加载 LoRA checkpoint → 评测从 50% 掉到 0.8%）。
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(args.out) / "checkpoint_best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if ckpt_path.exists():
        try:
            _meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _cfg = _meta.get("config") or {}
            has_lora = bool(_meta.get("lora"))
            _lr = _cfg.get("lora_lr")
            if has_lora:
                cfg.strategy = "qlora" if cfg.strategy == "qlora" else "lora"
                print(f"[checkpoint] 检测到 LoRA 权重，自动切换 strategy -> {cfg.strategy}")
                if _lr:
                    cfg.lora_lr = float(_lr)
            elif _cfg.get("lora_lr") is None:
                cfg.strategy = "projector_only"
            del _meta
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 读取 checkpoint 元信息失败：{type(exc).__name__}: {exc}")
    print(f"  评测策略 strategy = {cfg.strategy}")

    model = MiniVLM(cfg)
    collator = MultipleImageCollator(
        processor=model.processor, tokenizer=model.tokenizer,
        max_length=cfg.max_seq_len, num_image_token=model.num_image_token)

    from PIL import Image

    def load_one(rec: dict):
        return Image.open(resolve_image_path(rec["image"])).convert("RGB")

    tcfg = TrainingConfig(device=device, batch_size=4, cache_features_on_gpu=False,
                          output_dir=args.out)
    trainer = Trainer(model, collator, train_recs, parts["val"], load_one, tcfg)

    ckpt = Path(args.ckpt) if args.ckpt else trainer.out_dir / "checkpoint_best.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)
    else:
        print(f"[warn] 找不到 {ckpt}，将用随机初始化的 Projector 评测（仅作对照）")
    trainer.precompute_features(eval_recs)

    base = task_baselines(train_recs)
    print("\n每个任务在训练集上的多数类基线（这才是该任务的及格线）：")
    for t, b in base.items():
        print(f"  {t:<12} {b:>6.1%}")

    gen = trainer.evaluate_generation(
        eval_recs,
        max_new_tokens=args.max_new_tokens, micro_batch=args.gen_batch, detail_limit=None)

    print(f"\n{'任务':<12}{'样本':>6}{'精确':>9}{'前缀':>9}{'多数类基线':>11}{'超过基线':>10}")
    print("-" * 60)
    rows = []
    n_beat = 0
    for t, d in gen["per_task"].items():
        b = base.get(t, float("nan"))
        beat = d["exact"] > b
        n_beat += int(beat)
        rows.append((t, d["n"], d["exact"], d["prefix"], b, beat))
        print(f"{t:<12}{d['n']:>6}{d['exact']:>9.1%}{d['prefix']:>9.1%}"
              f"{b:>11.1%}{'超' if beat else '未超':>10}")

    print(f"\n总体：精确 {gen['exact']:.1%}  前缀 {gen['prefix']:.1%}"
          f"（{len(rows)} 个任务中 {n_beat} 个超过各自基线）")
    print("注意：总体指标不能与单个任务的基线比较 —— spatial 单独就有 51% 的多数类基线。")

    print("\n内容评分（保守规则；无法解析也计入准确率分母）：")
    for task, scores in gen["content_per_task"].items():
        print(f"  {task:<12} 内容正确={scores['correct']:.1%} 可解析={scores['parsed']:.1%} "
              f"内容对但非整句匹配={scores['correct_nonexact']:.1%}")

    # ---- 对照：把视觉特征与问题错位（视觉信息不再匹配）----
    print("\n对照实验：仅替换为下一张不同图片的特征（问题与答案保持不变）")
    gen_shift = trainer.evaluate_generation(
        eval_recs, visual_records=visual_recs,
        max_new_tokens=args.max_new_tokens, micro_batch=args.gen_batch, detail_limit=None)
    assert gen["n"] == gen_shift["n"] == len(eval_recs)
    print(f"  错位后：精确 {gen_shift['exact']:.1%}  前缀 {gen_shift['prefix']:.1%}")
    delta = gen["exact"] - gen_shift["exact"]
    print(f"  同样本原图减错图：{delta:+.1%}（单次对照差值，不代表统计显著性）")

    out = {
        "data": args.data, "split": args.split, "n": gen["n"],
        "exact": gen["exact"], "prefix": gen["prefix"],
        "per_task": gen["per_task"], "task_baselines": base,
        "content_metrics_version": 2,
        "content": gen["content"], "content_per_task": gen["content_per_task"],
        "n_tasks_beating_baseline": n_beat, "n_tasks": len(rows),
        "shifted_features_n": gen_shift["n"],
        "shifted_features": {"exact": gen_shift["exact"],
                             "prefix": gen_shift["prefix"],
                             "per_task": gen_shift["per_task"],
                             "content": gen_shift["content"],
                             "content_per_task": gen_shift["content_per_task"],
                             "details": gen_shift["details"]},
        "visual_control": {
            "version": 2, "method": "next_distinct_image",
            "exact_delta": delta,
            "pairs": [{"id": r["id"], "image": r["image"],
                       "control_image": v["image"]}
                      for r, v in zip(eval_recs, visual_recs)],
        },
        "details": gen["details"],
    }
    out_path = trainer.out_dir / f"eval_{args.split}_per_task.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
