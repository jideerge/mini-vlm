#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/check_env.py —— 环境自检（本地开工前 / 上云前都跑一次）

两种用法：
    python scripts/check_env.py            # 快速自检（不导入 torch）
    python scripts/check_env.py --deep     # 深度自检（导入 torch/transformers，验证 GPU）

注意：本脚本自己会设置 HF 离线变量（幂等），
      所以即使没有先 dot-source scripts/env.ps1，也能保证离线状态。
      若需要在联网状态下重新下载模型，先设 $env:MINIVLM_ALLOW_NET = "1"。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import site
import sys
import warnings
from pathlib import Path

# ------------------------------------------------------------------
# 0. 自行固定 HF 离线变量（幂等；可在 shell 中覆盖）
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = Path(os.environ.get("MINIVLM_PRETRAINED", ROOT / "checkpoints" / "pretrained"))

allow_net = os.environ.get("MINIVLM_ALLOW_NET") == "1"
if not allow_net:
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

OFFLINE_VARS = ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")


def head(title: str) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ------------------------------------------------------------------
# 1. 基础信息
# ------------------------------------------------------------------
def report_basic() -> None:
    head("1. 基础环境")
    print(f"  python      : {sys.version.split()[0]}   ({sys.executable})")
    print(f"  platform    : {platform.platform()}")
    print(f"  工作目录     : {os.getcwd()}")
    print(f"  仓库根目录   : {ROOT}")
    print(f"  离线变量     : {'全部关闭（MINIVLM_ALLOW_NET=1，允许联网）' if allow_net else '已开启'}")

    for var in OFFLINE_VARS:
        print(f"    {var:<26} = {os.environ.get(var)}")

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        print(f"  VIRTUAL_ENV : {venv}")
    else:
        warnings.warn("未检测到 VIRTUAL_ENV —— 你可能用错了 python（venv 在 dl_project\\venv）")

    if site.getusersitepackages() and Path(site.getusersitepackages()).exists():
        print(f"  user site   : {site.getusersitepackages()}")


# ------------------------------------------------------------------
# 2. 包版本
# ------------------------------------------------------------------
PKGS = ("torch", "torchvision", "transformers", "peft", "accelerate",
        "huggingface_hub", "tokenizers", "safetensors", "numpy",
        "datasets", "gradio", "bitsandbytes", "tensorboard")


def report_packages() -> dict[str, str]:
    head("2. 关键依赖")
    import importlib.metadata as md

    found: dict[str, str] = {}
    for name in PKGS:
        try:
            ver = md.version(name)
            found[name] = ver
            print(f"  [OK]      {name:<18} {ver}")
        except md.PackageNotFoundError:
            print(f"  [MISSING] {name:<18} 需要时再装（pip install {name}）")
    return found


# ------------------------------------------------------------------
# 3. GPU（仅 --deep）
# ------------------------------------------------------------------
def report_gpu() -> None:
    head("3. GPU / CUDA")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] 无法导入 torch: {type(exc).__name__}: {exc}")
        return

    print(f"  torch                : {torch.__version__}")
    print(f"  torch.version.cuda   : {torch.version.cuda}")
    print(f"  cuda.is_available()  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  [!] CUDA 不可用：检查显卡驱动 / 是否装成 CPU 版 torch")
        return

    prop = torch.cuda.get_device_properties(0)
    print(f"  gpu                  : {torch.cuda.get_device_name(0)}")
    print(f"  vram                 : {prop.total_memory / 1024 ** 3:.2f} GB")
    print(f"  compute capability   : {prop.major}.{prop.minor}")
    print(f"  bf16 support         : {torch.cuda.is_bf16_supported()}")
    # 6GB 显存的现实约束提示
    if prop.total_memory / 1024 ** 3 < 8:
        print("  [!] 显存 < 8GB：0.5B + bf16 LoRA 可跑；1.5B 需 QLoRA；3B 必须上云")


# ------------------------------------------------------------------
# 4. 本地模型资产
# ------------------------------------------------------------------
REQUIRED_FILES = {
    "clip-vit-base-patch32": ["config.json", "preprocessor_config.json"],
    "qwen2.5-0.5b-instruct": ["config.json", "tokenizer.json", "tokenizer_config.json"],
}
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")


def report_assets() -> None:
    head("4. 本地模型资产")
    print(f"  目录: {PRETRAINED}")
    if not PRETRAINED.exists():
        print("  [FAIL] 目录不存在")
        return

    for name, files in REQUIRED_FILES.items():
        d = PRETRAINED / name
        if not d.exists():
            print(f"  [FAIL] {name} 不存在")
            continue
        missing = [f for f in files if not (d / f).exists()]
        weights = [w for w in WEIGHT_NAMES if (d / w).exists()]
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024 ** 2
        status = "OK  " if not missing and weights else "FAIL"
        print(f"  [{status}] {name:<26} {size_mb:8.0f} MB  权重={weights or '无'}")
        if missing:
            print(f"         缺失文件: {missing}")
        if not weights:
            print("         缺少权重文件（只有 config 不算下载完整）")


# ------------------------------------------------------------------
# 5. 离线真实性验证（仅 --deep）：确认真的不发网络请求
# ------------------------------------------------------------------
def report_offline_truth() -> None:
    head("5. 离线真实性验证")
    try:
        from huggingface_hub import constants
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] 无法导入 huggingface_hub: {exc}")
        return

    print(f"  constants.HF_HOME        = {constants.HF_HOME}")
    print(f"  constants.HF_HUB_CACHE   = {constants.HF_HUB_CACHE}")
    print(f"  constants.HF_HUB_OFFLINE = {constants.HF_HUB_OFFLINE}")
    if not allow_net and not constants.HF_HUB_OFFLINE:
        print("  [!] 变量已设但 huggingface_hub 认为仍在线 —— 检查是否在 import 之前就设好了")
    else:
        print("  [OK] 离线标志已生效")

    # 真正试一次联网调用，验证它会被拦截（只做一次，失败属于预期）
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("openai/clip-vit-base-patch32", "config.json")
        print("  [!] 离线模式下竟然下载成功 —— HF_HUB_OFFLINE 未生效")
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if allow_net:
            print(f"  [OK] 允许联网模式，调用返回: {name}")
        else:
            print(f"  [OK] 离线拦截生效（抛出 {name}）—— 这正是期望行为")


def main() -> int:
    ap = argparse.ArgumentParser(description="Mini-VLM 环境自检")
    ap.add_argument("--deep", action="store_true", help="导入 torch/transformers 做深度检查")
    ap.add_argument("--json", metavar="PATH", help="把结果额外写成 JSON（如 report/env.json）")
    args = ap.parse_args()

    report_basic()
    pkgs = report_packages()
    report_assets()
    if args.deep:
        report_gpu()
        report_offline_truth()
    else:
        print("\n  (未使用 --deep，跳过 GPU 与离线真实性检查)")

    head("结论")
    has_torch = "torch" in pkgs
    print(f"  python 3.12 venv : {'OK' if sys.version_info[:2] == (3, 12) else '检查（当前 %s）' % sys.version.split()[0]}")
    print(f"  torch            : {'OK' if has_torch else '未安装'}")
    print(f"  离线变量         : {'已开启' if not allow_net else '已关闭（允许联网）'}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "packages": pkgs,
            "offline": {v: os.environ.get(v) for v in OFFLINE_VARS},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
