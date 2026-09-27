#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
short_time_check.py —— Week 2 视觉塔冒烟测试（10 秒级，不需要 GPU 大显存）

用途：最小代价确认「CLIP 能离线加载 + VisionEncoder 输出形状正确」。
      任何改动 vision_encoder.py 之后都应该跑一次。

运行（在仓库根）：
    & $env:MINIVLM_PY short_time_check.py
"""
import sys
from pathlib import Path

import torch

# 路径按文件位置推导，不依赖当前工作目录
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.vision_encoder import VisionEncoder  # noqa: E402

CLIP_DIR = ROOT / "checkpoints" / "pretrained" / "clip-vit-base-patch32"


def main() -> int:
    print("repo root :", ROOT)
    print("clip dir  :", CLIP_DIR, "| exists =", CLIP_DIR.exists())

    enc = VisionEncoder(str(CLIP_DIR))
    pixel_values = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = enc(pixel_values)

    print("output shape :", tuple(feats.shape))
    print("num_tokens   :", enc.num_tokens)
    print("hidden_size  :", enc.hidden_size)
    print("dtype        :", feats.dtype)

    pass_shape = tuple(feats.shape) == (2, 49, 768)
    print("形状断言      :", "PASS" if pass_shape else "FAIL (期望 (2, 49, 768))")
    if not pass_shape:
        print("  -> 检查 drop_cls 逻辑；CLIP last_hidden_state 含 CLS，需切片 out[:,1:,:]")
    return 0 if pass_shape else 1


if __name__ == "__main__":
    raise SystemExit(main())
