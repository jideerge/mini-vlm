#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train.py —— Mini-VLM 训练入口

一条命令跑完：读配置 -> 读数据与划分 -> 建模型 -> 训练 -> 验证 -> 生成评测 -> 出图

用法：
    # 1k 数据本地 Baseline（Week 5 的第一版）
    & $env:MINIVLM_PY scripts/train.py --data data/processed/debug_1k.jsonl \
        --out outputs/baseline_1k --steps 400

    # 只做评测（加载已有 checkpoint）
    & $env:MINIVLM_PY scripts/train.py --out outputs/baseline_1k --eval-only

    # 换 Projector 结构做消融（Week 7）
    & $env:MINIVLM_PY scripts/train.py --depth 0 --out outputs/ablation_linear
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from datasets.collator import MultipleImageCollator  # noqa: E402
from datasets.image_paths import resolve_image_path  # noqa: E402
from models.multimodal_model import MiniVLM, config_from_yaml  # noqa: E402
from training.trainer import Trainer, TrainingConfig, plot_history  # noqa: E402


def force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def load_records(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_one(rec: dict) -> Image.Image:
    return Image.open(resolve_image_path(rec["image"])).convert("RGB")


def split_records(records: list[dict], splits_path: Path) -> tuple[list, list, list]:
    if not splits_path.exists():
        raise SystemExit(
            f"找不到划分文件 {splits_path}\n"
            f"请先运行：& $env:MINIVLM_PY datasets/make_splits.py --data <数据.jsonl>"
        )
    sp = json.loads(splits_path.read_text(encoding="utf-8"))
    ids = sp["ids"]
    by_id = {r["id"]: r for r in records}
    parts = []
    for part in ("train", "val", "test"):
        parts.append([by_id[i] for i in ids[part] if i in by_id])
    return parts[0], parts[1], parts[2]


def main() -> int:  # noqa: C901
    force_utf8()
    ap = argparse.ArgumentParser(description="Mini-VLM 训练")
    ap.add_argument("--config", type=str, default="configs/model_clip_b32_qwen05.yaml")
    ap.add_argument("--data", type=str, default="data/processed/debug_1k.jsonl")
    ap.add_argument("--splits", type=str, default="data/processed/splits.json")
    ap.add_argument("--out", type=str, default="outputs/baseline_1k")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depth", type=int, default=None, help="Projector 层数 0/2/3")
    ap.add_argument("--strategy", type=str, default=None,
                    choices=["projector_only", "lora", "qlora"],
                    help="覆盖配置里的训练策略（Experiment A/B/C）")
    ap.add_argument("--lora-lr", type=float, default=None,
                    help="LoRA 适配器的学习率（默认 2e-4；绝不能与 projector lr 混用）")
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--grad-ckpt", action="store_true", help="启用梯度检查点（省显存）")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-gpu-cache", action="store_true", help="特征缓存放内存而非显存")
    ap.add_argument("--eval-only", action="store_true", help="只加载 checkpoint 做评测")
    ap.add_argument("--dry-run", action="store_true", help="只检查真实 batch、参数和前向，不训练或保存权重")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--gen-samples", type=int, default=40)
    args = ap.parse_args()

    # Seed before constructing the randomly initialized Projector.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Mini-VLM 训练")
    print("=" * 70)
    print(f"  device={device}  data={args.data}  out={args.out}")

    # ---------------- 数据 ----------------
    data_path = ROOT / args.data
    records = load_records(data_path)
    train_recs, val_recs, test_recs = split_records(records, ROOT / args.splits)
    print(f"  样本：train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")
    if not train_recs:
        raise SystemExit("训练集为空 —— 检查 splits.json 与数据文件是否匹配")

    # ---------------- 模型 ----------------
    cfg = config_from_yaml(ROOT / args.config)
    cfg.device = device
    if args.depth is not None:
        cfg.projector_depth = args.depth
    if args.strategy is not None:
        cfg.strategy = args.strategy
    if args.lora_r is not None:
        cfg.lora_r = args.lora_r
    t0 = time.perf_counter()
    model = MiniVLM(cfg)
    print(f"  模型加载 {time.perf_counter() - t0:.1f} s")
    print("  " + model.parameter_summary().replace("\n", "\n  "))

    if args.grad_ckpt:
        if hasattr(model.llm, "gradient_checkpointing_enable"):
            model.llm.gradient_checkpointing_enable()
            print("  已启用梯度检查点")
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()

    # LoRA 策略下 lora_lr 必须有值，否则会与 projector 共用 lr（必然训练失败）
    lora_lr = args.lora_lr
    if cfg.strategy in ("lora", "qlora") and lora_lr is None:
        lora_lr = 2e-4
    if cfg.strategy in ("lora", "qlora"):
        print(f"  策略={cfg.strategy}  projector_lr={args.lr}  lora_lr={lora_lr}"
              f"  r={cfg.lora_r}")

    collator = MultipleImageCollator(
        processor=model.processor, tokenizer=model.tokenizer,
        max_length=cfg.max_seq_len, num_image_token=model.num_image_token,
    )

    tcfg = TrainingConfig(
        data_path=args.data, splits_path=args.splits, output_dir=args.out,
        projector_lr=args.lr, lora_lr=lora_lr, max_steps=args.steps,
        batch_size=args.batch,
        grad_accum_steps=args.accum, warmup_steps=args.warmup,
        eval_every=args.eval_every, log_every=args.log_every, seed=args.seed,
        device=device, max_length=cfg.max_seq_len,
        cache_features_on_gpu=not args.no_gpu_cache,
    )
    trainer = Trainer(model, collator, train_recs, val_recs, load_one, tcfg)

    if args.dry_run:
        sample = train_recs[:min(2, len(train_recs))]
        trainer.precompute_features(sample)
        batch, features = trainer.make_batch(sample)
        model.eval()
        with torch.no_grad():
            result = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"], visual_features=features,
            )
        if not torch.isfinite(result["loss"]):
            raise RuntimeError("dry-run 前向损失不是有限值")
        print(f"  dry-run input_ids={tuple(batch['input_ids'].shape)} "
              f"visual_features={tuple(features.shape)} "
              f"attention_mask={tuple(batch['attention_mask'].shape)}")
        print(f"  dry-run trainable={model.trainable_parameters:,} "
              f"loss={float(result['loss'].item()):.4f}")
        print("  dry-run 通过；未运行优化器步骤或保存 checkpoint")
        return 0

    # ---------------- 训练 ----------------
    if not args.eval_only:
        t0 = time.perf_counter()
        trainer.train()
        print(f"\n  （含评测在内的总墙钟 {time.perf_counter() - t0:.1f} s）")
        fig = plot_history(trainer.history, trainer.out_dir / "training_curve.png",
                           title=f"Mini-VLM Baseline（{Path(args.data).name}）")
        if fig:
            print(f"  训练曲线 -> {fig}")
            dest = ROOT / "report" / "figures" / f"train_curve_{Path(args.out).name}.png"
            shutil.copy(fig, dest)
            print(f"  已复制到 {dest}")

    # ---------------- 评测 ----------------
    print("\n" + "=" * 70)
    print("  评测")
    print("=" * 70)
    ckpt = Path(args.ckpt) if args.ckpt else trainer.out_dir / "checkpoint_best.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)
    else:
        print(f"  [warn] 找不到 {ckpt}，用当前权重评测")

    # ⚠️ 评测同样需要视觉特征：训练路径里是 train() 内部预编码的，
    # 而 --eval-only 不经过 train()，必须在这里显式补上，
    # 否则 _features_for 会 KeyError。
    trainer.precompute_features(val_recs + test_recs)

    val_metrics = trainer.validate()
    print(f"  val: loss={val_metrics['val_loss']:.4f}  "
          f"tok_acc={val_metrics['val_tok_acc']:.3f}  "
          f"({val_metrics['val_tokens']} tokens)")

    gen = trainer.evaluate_generation(val_recs, max_samples=args.gen_samples)
    if gen["n"]:
        print(f"  生成评测（{gen['n']} 条）：精确匹配 {gen['exact']:.1%}  "
              f"前缀匹配 {gen['prefix']:.1%}")
        # 按任务拆开看 —— 聚合指标会掩盖"某个任务特别差"
        pt = gen.get("per_task") or {}
        if pt:
            print(f"\n  {'任务':<12}{'样本':>6}{'精确':>9}{'前缀':>9}")
            print("  " + "-" * 38)
            for t, d in pt.items():
                print(f"  {t:<12}{d['n']:>6}{d['exact']:>9.1%}{d['prefix']:>9.1%}")
        print("\n  样例：")
        for d in gen["details"][:8]:
            mark = "OK  " if d["exact"] else ("PRE " if d["prefix"] else "MISS")
            print(f"    [{mark}] ({d['task']}) {d['question']}")
            print(f"           参考: {d['reference']}")
            print(f"           生成: {d['generated']}")

    counter = Counter(r["answer"] for r in train_recs)
    top_ans, top_cnt = counter.most_common(1)[0]
    majority_acc = top_cnt / len(train_recs)
    print(f"\n  多数类基线（训练集最高频答案 {top_ans!r}）：{majority_acc:.1%}")
    print(f"  模型答案 token 准确率 {val_metrics['val_tok_acc']:.1%}  "
          f"vs 多数类基线 {majority_acc:.1%}")
    print(f"  生成精确匹配 {gen['exact']:.1%}  vs 多数类基线 {majority_acc:.1%}")

    result = {
        "data": args.data, "out": args.out,
        "n_train": len(train_recs), "n_val": len(val_recs), "n_test": len(test_recs),
        "val": val_metrics,
        "generation": {k: v for k, v in gen.items() if k != "details"},
        "generation_details": gen.get("details", []),
        "majority_baseline": majority_acc,
        "run": {"steps": args.steps, "batch": args.batch, "accum": args.accum,
                "lr": args.lr, "depth": cfg.projector_depth, "device": device},
        "trainable": model.trainable_parameters, "total": model.total_parameters,
    }
    out_json = trainer.out_dir / "eval_result.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  评测结果 -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
