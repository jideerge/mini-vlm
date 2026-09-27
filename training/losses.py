#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training/losses.py —— 损失、掩码与优化器分组

本项目的损失就是标准的自回归交叉熵，但有两个必须做对的地方：

  1. **只在 answer 区间计算**：labels 里 question / 图像占位符 / pad 全是 -100。
     忘记掩码的话，模型会去学"预测问题"——loss 看着在降，实际什么都没学会。
  2. **shift 对齐**：位置 t 的 logits 预测位置 t+1 的 token，
     即 `logits[:, :-1, :]` 对齐 `labels[:, 1:]`。少一位或多一位都是隐蔽 bug。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """全部位置的掩码交叉熵（保留 shift 语义，便于对照检查）。"""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
    )


def masked_lm_loss_and_acc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, int, int]:
    """返回 (loss, 答案 token 数, 预测正确的 token 数)。

    只对答案位置取 logits 再算 loss：显存占用从 O(全序列 × 词表)
    降到 O(答案长度 × 词表)，在 6GB 显存上差别很明显。
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_labels = shift_labels.reshape(-1)
    keep = flat_labels != ignore_index
    n_tok = int(keep.sum().item())
    if n_tok == 0:
        return logits.sum() * 0.0, 0, 0
    sel_logits = shift_logits.reshape(-1, shift_logits.size(-1))[keep].float()
    sel_labels = flat_labels[keep]
    loss = F.cross_entropy(sel_logits, sel_labels)
    n_correct = int((sel_logits.argmax(dim=-1) == sel_labels).sum().item())
    return loss, n_tok, n_correct


def answer_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """(正确 token 数, 总答案 token 数)，不算 loss（评估时更省显存）。"""
    with torch.no_grad():
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        flat = shift_labels.reshape(-1)
        keep = flat != IGNORE_INDEX
        if not bool(keep.any()):
            return 0, 0
        sel = shift_logits.reshape(-1, shift_logits.size(-1))[keep].float()
        tgt = flat[keep]
        return int((sel.argmax(dim=-1) == tgt).sum().item()), int(keep.sum().item())


def normalize_answer_text(s: str) -> str:
    """用于答案比对：去掉空白与中英文标点，避免标点差异被判为错误。"""
    import re

    return re.sub(r"[\s，。、？！：；,.\?!:;]", "", s)


def sequence_exact_match(
    generated_ids: torch.Tensor,
    target_texts: list[str],
    tokenizer: Any,
) -> list[bool]:
    """生成结果与参考答案的精确匹配（归一化后比较）。"""
    out = []
    for i, tgt in enumerate(target_texts):
        gen = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        out.append(normalize_answer_text(gen) == normalize_answer_text(tgt))
    return out


def build_param_groups(
    model: Any,
    projector_lr: float,
    lora_lr: float | None = None,
    weight_decay: float = 0.0,
) -> list[dict]:
    """按模块分组设置学习率。

    Week 1 就定下的纪律：Projector 随机初始化，lr 在 1e-3 量级；
    LoRA 微调预训练权重，lr 在 2e-4 量级。**两者混用必然训练失败。**
    """
    groups: list[dict] = []
    used: set[int] = set()

    proj_params = [p for p in model.projector.parameters() if p.requires_grad]
    if proj_params:
        used.update(id(p) for p in proj_params)
        groups.append({
            "params": proj_params,
            "lr": projector_lr,
            "weight_decay": weight_decay,
            "name": "projector",
        })

    if lora_lr is not None:
        lora_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "lora_" in n and id(p) not in used
        ]
        if lora_params:
            used.update(id(p) for p in lora_params)
            groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": 0.0,
                "name": "lora",
            })

    other = [
        p for _, p in model.named_parameters()
        if p.requires_grad and id(p) not in used
    ]
    if other:
        groups.append({
            "params": other,
            "lr": lora_lr if lora_lr is not None else projector_lr,
            "weight_decay": weight_decay,
            "name": "other",
        })
    return groups


def cosine_schedule_with_warmup(
    warmup_steps: int,
    total_steps: int,
    min_ratio: float = 0.05,
):
    """warmup + 余弦退火，返回 lr_lambda（配合 torch.optim.lr_scheduler.LambdaLR）。"""
    import math

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn
