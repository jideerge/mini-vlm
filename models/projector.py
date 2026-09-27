#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/projector.py —— 视觉-语言连接器（本项目唯一"自己写"的核心模块）

这是 LLaVA 式 VLM 的关键：把冻结视觉塔的特征映射到冻结 LLM 的嵌入空间。

    CLIP 输出  [B, 49, 768]
        -> Projector
    LLM 嵌入   [B, 49, 896]

三种结构（Week 3 的消融实验对象，参数 depth 控制）：

    depth=0  Linear       768 -> 896
    depth=2  2-layer MLP  768 -> 2048 -> GELU -> 896      (LLaVA-1.0 用这个)
    depth=3  3-layer MLP  768 -> 2048 -> GELU -> 2048 -> GELU -> 896

参数量（768/2048/896，请用 count_parameters 实测核对，别照抄）：
    Linear      : 768*896 + 896            =    689,024   (0.69M)
    2-layer MLP : 768*2048+2048 + 2048*896+896 = 3,410,944  (3.41M)
    3-layer MLP : 3,410,944 + 2048*2048+2048   = 7,606,272  (7.61M)
"""
from __future__ import annotations

import torch.nn as nn


class Projector(nn.Module):
    """把视觉特征投到 LLM 嵌入维度。

    输入: [B, N, in_dim]
    输出: [B, N, out_dim]
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 896,
        hidden: int = 2048,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth

        if depth == 0:
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        elif depth == 2:
            layers = [
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            ]
        elif depth == 3:
            layers = [
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            ]
        else:
            raise ValueError(f"depth 只支持 0 / 2 / 3，收到 {depth}")

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    ps = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)


if __name__ == "__main__":
    # 直接运行本文件即可得到 Week 3 要填的参数量表
    import torch

    print(f"{'Projector':<14}{'参数量':>14}{'输出形状':>22}")
    print("-" * 52)
    for label, depth in (("Linear", 0), ("2-layer MLP", 2), ("3-layer MLP", 3)):
        p = Projector(in_dim=768, out_dim=896, hidden=2048, depth=depth)
        n = count_parameters(p)
        with torch.no_grad():
            out = p(torch.randn(2, 49, 768))
        print(f"{label:<14}{n:>14,}{str(tuple(out.shape)):>22}")
