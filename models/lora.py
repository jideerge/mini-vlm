#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/lora.py —— LoRA / QLoRA 注入

LoRA 的核心：把权重更新分解成两个低秩矩阵

    W' = W + BA        W 冻结，只训练 A、B

  * A: [r, in_features]，B: [out_features, r]，r << min(in, out)
  * 初始化：A 用 Kaiming，B 全零 —— 所以**训练开始时 LoRA 的输出恒为 0**，
    模型行为与原始预训练模型完全一致（很重要，否则会破坏预训练能力）

为什么本项目要用 LoRA：
    Experiment A（Projector-only）已证明"只翻译、不改模型"能到 32% 精确匹配，
    但 spatial / existence 几乎没学会。假设瓶颈在**冻结 LLM 的表达能力** ——
    Projector 只能把视觉特征翻译成 LLM 的语言，却改不了 LLM 自己的行为习惯
    （例如"答完不停"）。LoRA 让 LLM 本身也能被微调。

⚠️ 项目纪律（Week 1 定下）：
    **Projector 的 lr（1e-3 量级）与 LoRA 的 lr（2e-4 量级）必须分开设置。**
    混用必然训练失败。

Windows 提示：QLoRA 需要 bitsandbytes，Windows 原生环境不稳定。
`use_qlora=False`（纯 bf16 LoRA）不依赖它，是本机默认路径。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # QLoRA（需 bitsandbytes）
    use_qlora: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


def build_peft_config(cfg: LoRAConfig):
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


def apply_lora(model: nn.Module, cfg: LoRAConfig, verbose: bool = True) -> nn.Module:
    """给（因果）语言模型注入 LoRA 适配器，返回包装后的 PeftModel。

    注入后**基础权重会被冻结**，只有 lora_A / lora_B 可训练；
    `bias='none'` 表示连 bias 也不训练。
    """
    from peft import get_peft_model

    peft_model = get_peft_model(model, build_peft_config(cfg))
    if verbose:
        n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in peft_model.parameters())
        print(f"  LoRA 已注入: r={cfg.r} alpha={cfg.alpha} dropout={cfg.dropout}")
        print(f"    target_modules = {list(cfg.target_modules)}")
        print(f"    可训练 {n_train:,} / {n_total:,}（{n_train / n_total:.2%}）")
    return peft_model


def lora_report(peft_model: nn.Module) -> dict[str, Any]:
    """统计 LoRA 注入了哪些模块、参数量多少 —— 训练前自检用。"""
    names = [n for n, _ in peft_model.named_parameters() if "lora_" in n]
    n_params = sum(p.numel() for n, p in peft_model.named_parameters()
                   if "lora_" in n and p.requires_grad)
    by_module: dict[str, int] = {}
    for n in names:
        key = n.split("lora_")[0].rstrip(".").split(".")[-1]
        by_module[key] = by_module.get(key, 0) + 1
    return {
        "num_lora_tensors": len(names),
        "num_lora_params": n_params,
        "by_module": by_module,
        "example_names": names[:4],
    }


def named_trainable(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def assert_only_projector_and_lora(model: nn.Module, projector_attr: str = "projector") -> list[str]:
    """自检：可训练参数必须只来自 projector 与 lora_*。

    返回"不属于这两类"的参数名（空列表 = 通过）。
    这是防止"以为只训 LoRA，实际把整个 LLM 也训了"这类事故的关键检查。
    """
    bad = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(f"{projector_attr}.") or "lora_" in n:
            continue
        bad.append(n)
    return bad


def prepare_qlora(model_name_or_path: str, cfg: LoRAConfig):
    """QLoRA：4-bit 量化加载基础模型（需 bitsandbytes）。

    Windows 原生环境通常装不上 bitsandbytes，这里给出明确提示，
    而不是让训练跑到一半才失败。
    """
    if not cfg.use_qlora:
        raise ValueError("prepare_qlora 只在 use_qlora=True 时调用")
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "QLoRA 需要 bitsandbytes，但导入失败：\n"
            f"  {type(exc).__name__}: {exc}\n"
            "  解决方式（任选）：\n"
            "   1) 用 WSL2 + Linux 环境（推荐，坑最少）\n"
            "   2) 改用纯 bf16 LoRA（use_qlora=False，Windows 上的默认路径）\n"
            "   3) 把 QLoRA 实验放到云端 Linux 实例上跑"
        ) from exc

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path, quantization_config=bnb, device_map={"": 0}
    )
