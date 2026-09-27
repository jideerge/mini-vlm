#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training/trainer.py —— 训练循环（自己写，不套 HuggingFace Trainer）

计划书的原则：**训练 pipeline 属于"我们亲手实现"的部分**，
所以优化、调度、梯度累积、验证、checkpoint 与日志都在这里自己写。

三个设计要点：
  1. **视觉特征预编码**：视觉塔冻结 => 同一张图特征恒定。
     训练前一次性算完并缓存，之后每个 epoch 不再跑 CLIP。
     实测 1000 张约 8 秒，而训练要跑几百步 —— 这笔投资必然划算。
  2. **用验证 loss 选 checkpoint**，不是看训练 loss。
  3. **优化器分组**：Projector 用 `projector_lr`（1e-3 量级），
     LoRA 用 `lora_lr`（2e-4 量级）—— 混用必然训练失败（Week 1 定下的纪律）。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from datasets.dataset import JsonlVLDataset
from training.losses import build_param_groups, cosine_schedule_with_warmup


@dataclass
class TrainingConfig:
    # 数据
    data_path: str = "data/processed/debug_1k.jsonl"
    splits_path: str = "data/processed/splits.json"
    output_dir: str = "outputs/baseline_1k"

    # 优化
    projector_lr: float = 1.0e-3
    lora_lr: float | None = None
    weight_decay: float = 0.0
    max_steps: int = 400
    batch_size: int = 4
    grad_accum_steps: int = 4
    warmup_steps: int = 40
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.05
    seed: int = 42

    # 评估与保存
    eval_every: int = 40
    log_every: int = 20
    save_best: bool = True
    save_last: bool = True
    device: str = "cuda"
    max_length: int = 768

    # 特征缓存
    cache_features_on_gpu: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class Trainer:
    """极简但完整的训练器。"""

    def __init__(
        self,
        model: Any,
        collator: Any,
        train_records: list[dict],
        val_records: list[dict],
        load_image: Callable[[dict], Any],
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.collator = collator
        self.cfg = config
        self.device = torch.device(config.device)
        self.load_image = load_image

        self.train_records = train_records
        self.val_records = val_records
        self.out_dir = Path(config.output_dir)
        if not self.out_dir.is_absolute():
            self.out_dir = Path(__file__).resolve().parents[1] / self.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        groups = build_param_groups(
            model,
            projector_lr=config.projector_lr,
            lora_lr=config.lora_lr,
            weight_decay=config.weight_decay,
        )
        if not groups:
            raise RuntimeError("没有任何可训练参数 —— 检查冻结设置")
        self.param_groups_info = [
            (g["name"], sum(p.numel() for p in g["params"]), g["lr"]) for g in groups
        ]
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            cosine_schedule_with_warmup(
                config.warmup_steps, config.max_steps, config.min_lr_ratio
            ),
        )

        self.history: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_step = -1
        self.global_step = 0
        self.feature_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def precompute_features(self, records: list[dict], batch: int = 8,
                            verbose: bool = True) -> None:
        """缓存视觉塔输出（**投影前** [N, Dv]）。

        ⚠️ 必须缓存投影前的特征：Projector 要留在每次前向的计算图内，
        否则梯度会断（notes/week3_projector_notes.md 坑 2）。
        """
        from datasets.collator import pad_to_square

        todo = [r for r in records if r["image"] not in self.feature_cache]
        if not todo:
            return
        t0 = time.perf_counter()
        store_gpu = self.cfg.cache_features_on_gpu and self.device.type == "cuda"
        self.model.eval()
        with torch.no_grad():
            for s in range(0, len(todo), batch):
                chunk = todo[s:s + batch]
                if self.model.vision is None:
                    raise RuntimeError("未加载视觉塔，无法编码图片")
                pv = self.model.processor(
                    images=[pad_to_square(self.load_image(r)) for r in chunk],
                    return_tensors="pt",
                )["pixel_values"].to(self.device)
                f = self.model.vision(pv).detach()
                f = f.to(torch.float16) if store_gpu else f.cpu()
                for j, r in enumerate(chunk):
                    self.feature_cache[r["image"]] = f[j]
        if verbose:
            print(f"  特征预编码：新增 {len(todo)} 张，累计 {len(self.feature_cache)} 张，"
                  f"耗时 {time.perf_counter() - t0:.1f} s"
                  f"（{'fp16 显存' if store_gpu else '内存'}）")

    def _features_for(self, records: list[dict]) -> torch.Tensor:
        dtype = next(self.model.projector.parameters()).dtype
        return torch.stack(
            [self.feature_cache[r["image"]].to(self.device, dtype=dtype) for r in records],
            dim=0,
        )

    def make_batch(self, records: list[dict]) -> tuple[dict, torch.Tensor]:
        """组装一个 batch。

        ⚠️ 问题里的 <|image_pad|> 占位符必须在这里补上：
        records 里存的是**原始问题**（"图中一共有几个椅子？"），
        而 collator 期望的是**带一个占位符的问题**（"<|image_pad|>\\n图中..."）。
        第一版直接把原始问题传给 collator，导致文本里 0 个占位符、
        视觉侧 49 个 token，前向时才报"占位符 0 个，视觉 token 49 个，不匹配"。
        """
        samples = [
            {
                "image": self.load_image(r),
                "question": JsonlVLDataset.build_question(r["question"]),
                "answer": r["answer"],
            }
            for r in records
        ]
        return self.collator(samples), self._features_for(records)

    # ------------------------------------------------------------------
    def validate(self) -> dict[str, float]:
        if not self.val_records:
            return {"val_loss": float("nan"), "val_tok_acc": float("nan"),
                    "val_tokens": 0}
        self.model.eval()
        tot_loss, n_batches, tok_ok, tok_all = 0.0, 0, 0, 0
        bs = self.cfg.batch_size
        with torch.no_grad():
            for s in range(0, len(self.val_records), bs):
                b, vf = self.make_batch(self.val_records[s:s + bs])
                out = self.model(
                    input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    labels=b["labels"], visual_features=vf,
                )
                tot_loss += float(out["loss"].item())
                n_batches += 1
                tok_ok += out["n_correct"]
                tok_all += out["n_answer_tokens"]
        self.model.train()
        return {
            "val_loss": tot_loss / max(n_batches, 1),
            "val_tok_acc": tok_ok / max(tok_all, 1),
            "val_tokens": tok_all,
        }

    @torch.no_grad()
    def evaluate_generation(self, records: list[dict], max_new_tokens: int = 24,
                            micro_batch: int = 4,
                            max_samples: int | None = None,
                            visual_records: list[dict] | None = None,
                            detail_limit: int | None = 200) -> dict:
        """自由生成评测（无 teacher forcing）：精确匹配率与前缀匹配率。"""
        from training.losses import normalize_answer_text
        from evaluation.metrics import score_content, summarize_content

        # 对照只替换视觉输入；文本和评分参考始终来自 records。
        if visual_records is not None and len(visual_records) != len(records):
            raise ValueError("visual_records 必须与 records 等长且逐项对应")
        recs = records[:max_samples] if max_samples else records
        visual_recs = (visual_records[:len(recs)]
                       if visual_records is not None else None)
        if not recs:
            return {"n": 0, "exact": float("nan"), "prefix": float("nan"), "details": []}
        self.model.eval()
        exact = prefix = 0
        content_scores = []
        content_by_task = {}
        by_task: dict[str, dict[str, int]] = {}
        details: list[dict] = []
        bs = self.cfg.batch_size
        for s in range(0, len(recs), bs):
            chunk = recs[s:s + bs]
            b, vf = self.make_batch(chunk)
            if visual_recs is not None:
                vf = self._features_for(visual_recs[s:s + bs])
            texts = self.model.generate_text(
                input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                visual_features=vf, max_new_tokens=max_new_tokens,
                micro_batch=micro_batch,
                answer_start=torch.tensor(b["answer_starts"], device=self.device),
            )
            for r, g in zip(chunk, texts):
                na = normalize_answer_text(r["answer"].lstrip())
                ng = normalize_answer_text(g.lstrip())
                is_exact = na == ng
                is_prefix = is_exact or (len(na) > 0 and ng.startswith(na))
                exact += int(is_exact)
                prefix += int(is_prefix)
                # 按任务累计：聚合指标会掩盖"某个任务特别差"这种情况
                # （实测 spatial 的多数类基线就有 48.9%）
                t = r.get("task", "?")
                content = score_content(r, g)
                content_scores.append(content)
                content_by_task.setdefault(t, []).append(content)
                d = by_task.setdefault(t, {"n": 0, "exact": 0, "prefix": 0})
                d["n"] += 1
                d["exact"] += int(is_exact)
                d["prefix"] += int(is_prefix)
                if detail_limit is None or len(details) < detail_limit:
                    details.append({
                        "id": r.get("id"), "image": r.get("image"),
                        "meta": r.get("meta", {}), "content": content,
                        "task": t, "question": r["question"],
                        "reference": r["answer"], "generated": g,
                        "exact": is_exact, "prefix": is_prefix,
                    })
        self.model.train()
        per_task = {
            t: {"n": d["n"], "exact": d["exact"] / d["n"],
                "prefix": d["prefix"] / d["n"]}
            for t, d in sorted(by_task.items())
        }
        return {"n": len(recs), "exact": exact / len(recs),
                "prefix": prefix / len(recs), "per_task": per_task,
                "content": summarize_content(content_scores),
                "content_per_task": {t: summarize_content(v) for t, v in content_by_task.items()},
                "details": details}

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        payload: dict[str, Any] = {
            "projector": self.model.projector.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg),
        }
        # LoRA 适配器：单独存一份（键名保持模型里的原始形式，便于原样 load）
        lora_sd = {n: p.detach().cpu() for n, p in self.model.named_parameters()
                   if "lora_" in n and p.requires_grad}
        if lora_sd:
            payload["lora"] = lora_sd
        # 其它可训练参数（理论上不该有；留着以防将来加模块）
        other = {
            n: p.detach().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("projector.") and "lora_" not in n
        }
        if other:
            payload["trainable_extra"] = other
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> None:
        """加载 checkpoint。

        ⚠️ `strict=True`（默认）会在"checkpoint 里有参数、但当前模型里不存在"时报错。
        这不是洁癖 —— 实测踩过：用 `strategy=projector_only` 的配置去加载一个 LoRA
        checkpoint，LoRA 权重会被**静默丢弃**，评测结果从 50% 掉到 0.8%，
        而脚本一声不吭、还打印"已加载"。静默的部分加载比直接报错危险得多。
        """
        p = Path(path)
        if not p.is_absolute():
            p = self.out_dir / p
        ckpt = torch.load(p, map_location=self.device)
        self.model.projector.load_state_dict(ckpt["projector"])

        own = dict(self.model.named_parameters())
        loaded, missing = 0, []
        for key in ("lora", "trainable_extra"):
            for name, tensor in (ckpt.get(key) or {}).items():
                if name in own:
                    own[name].data.copy_(tensor.to(self.device))
                    loaded += 1
                else:
                    missing.append(name)
        if missing and strict:
            raise RuntimeError(
                f"checkpoint 里有 {len(missing)} 个参数在当前模型中不存在，"
                f"拒绝静默加载。\n"
                f"  前 3 个：{missing[:3]}\n"
                f"  当前模型 strategy = {self.model.strategy}\n"
                f"  常见原因：用 projector_only 的配置去加载 LoRA checkpoint。\n"
                f"  正确做法：让配置里的 strategy 与训练时一致，"
                f"或从 checkpoint 的 `config` 字段读回 strategy。\n"
                f"  如确认要忽略，传 strict=False。"
            )
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        msg = (f"[checkpoint] 已加载 {p}（step {self.global_step}，"
               f"best_val_loss {self.best_val_loss:.4f}")
        msg += f"，{loaded} 个额外张量）" if loaded else "）"
        print(msg)
        if missing:
            print(f"  [warn] 忽略了 {len(missing)} 个不存在的参数")

    # ------------------------------------------------------------------
    def train(self, log_fn: Callable[[str], None] = print) -> list[dict]:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        log_fn("")
        log_fn("=" * 70)
        log_fn("  开始训练")
        log_fn("=" * 70)
        log_fn(f"  训练样本 {len(self.train_records)}   验证样本 {len(self.val_records)}")
        log_fn(f"  batch={cfg.batch_size} x accum={cfg.grad_accum_steps} "
               f"-> 等效 {cfg.batch_size * cfg.grad_accum_steps}")
        log_fn(f"  max_steps={cfg.max_steps}  warmup={cfg.warmup_steps}  "
               f"lr={cfg.projector_lr}")
        log_fn(f"  参数组: {self.param_groups_info}")

        self.precompute_features(self.train_records + self.val_records)
        (self.out_dir / "train_config.json").write_text(cfg.to_json(), encoding="utf-8")

        self.model.train()
        if self.model.vision is not None:
            self.model.vision.eval()
        self.model.llm.eval()

        n = len(self.train_records)
        micro_per_step = cfg.grad_accum_steps
        t_start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        micro_idx = 0
        epoch = 0
        while self.global_step < cfg.max_steps:
            order = torch.randperm(n).tolist()
            epoch += 1
            self.optimizer.zero_grad(set_to_none=True)
            for k in range(0, len(order), cfg.batch_size):
                if self.global_step >= cfg.max_steps:
                    break
                chunk = [self.train_records[i] for i in order[k:k + cfg.batch_size]]
                b, vf = self.make_batch(chunk)
                out = self.model(
                    input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    labels=b["labels"], visual_features=vf,
                )
                (out["loss"] / micro_per_step).backward()
                micro_idx += 1
                if micro_idx % micro_per_step != 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                rec = {
                    "step": self.global_step,
                    "epoch": epoch,
                    "loss": float(out["loss"].item()),
                    "tok_acc": out["n_correct"] / max(out["n_answer_tokens"], 1),
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                }
                if self.global_step % cfg.eval_every == 0:
                    v = self.validate()
                    rec.update(v)
                    if self.cfg.save_best and v["val_loss"] < self.best_val_loss:
                        self.best_val_loss = v["val_loss"]
                        self.best_step = self.global_step
                        self.save_checkpoint(self.out_dir / "checkpoint_best.pt")
                    self.history.append(rec)
                    log_fn(f"  step {self.global_step:4d}  loss={rec['loss']:.4f}  "
                           f"tok_acc={rec['tok_acc']:.3f}  "
                           f"val_loss={v['val_loss']:.4f}  "
                           f"val_tok_acc={v['val_tok_acc']:.3f}  "
                           f"lr={rec['lr']:.2e}")
                elif self.global_step % cfg.log_every == 0:
                    self.history.append(rec)
                    log_fn(f"  step {self.global_step:4d}  loss={rec['loss']:.4f}  "
                           f"tok_acc={rec['tok_acc']:.3f}  "
                           f"gn={rec['grad_norm']:.3f}  lr={rec['lr']:.2e}")

        train_s = time.perf_counter() - t_start
        peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                   if self.device.type == "cuda" else 0.0)
        if cfg.save_last:
            self.save_checkpoint(self.out_dir / "checkpoint_last.pt")

        final_val = self.validate()
        summary = {
            "max_steps": cfg.max_steps,
            "epochs": epoch,
            "train_s": train_s,
            "ms_per_step": train_s / max(self.global_step, 1) * 1000,
            "peak_mb": peak_mb,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step,
            "final_val": final_val,
            "trainable": self.model.trainable_parameters,
            "total": self.model.total_parameters,
            "param_groups": self.param_groups_info,
            "n_train": len(self.train_records),
            "n_val": len(self.val_records),
        }
        (self.out_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.out_dir / "history.json").write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")

        log_fn("")
        log_fn(f"  训练完成：{self.global_step} 步 / {epoch} 个 epoch，"
               f"{train_s:.1f} s（{summary['ms_per_step']:.0f} ms/step）")
        log_fn(f"  显存峰值 {peak_mb:.0f} MB")
        log_fn(f"  最优 val_loss = {self.best_val_loss:.4f}（step {self.best_step}）")
        log_fn(f"  最终 val: loss={final_val['val_loss']:.4f} "
               f"tok_acc={final_val['val_tok_acc']:.3f}")
        return self.history


# ----------------------------------------------------------------------
def plot_history(history: list[dict], out_path: str | Path,
                 title: str = "Mini-VLM 训练曲线") -> str | None:
    """把 loss / token 准确率 / lr 画成图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception:  # noqa: BLE001
        return None
    if not history:
        return None

    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False

    steps = [h["step"] for h in history]
    losses = [h.get("loss") for h in history]
    toks = [h.get("tok_acc") for h in history]
    vsteps = [h["step"] for h in history if "val_loss" in h]
    vlosses = [h["val_loss"] for h in history if "val_loss" in h]
    vtoks = [h.get("val_tok_acc") for h in history if "val_loss" in h]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, losses, "-", alpha=0.7, label="train")
    if vsteps:
        axes[0].plot(vsteps, vlosses, "o-", label="val")
    axes[0].set_yscale("log")
    axes[0].set_title("loss（对数纵轴）")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, toks, "-", alpha=0.7, label="train")
    if vsteps:
        axes[1].plot(vsteps, vtoks, "o-", label="val")
    axes[1].set_title("答案 token 准确率")
    axes[1].set_xlabel("step")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, [h.get("lr") for h in history], "-", color="tab:green")
    axes[2].set_title("学习率（warmup + 余弦）")
    axes[2].set_xlabel("step")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)
