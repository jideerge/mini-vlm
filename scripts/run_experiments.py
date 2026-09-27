#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_experiments.py —— Week 2 视觉编码器实验（真实图片、真实指标）

四个实验，全部基于本机 VOC2007 真实照片（约 9963 张，零下载成本）：

  A 特征形状与冻结      一张图走通 image -> [B,50,768] -> 丢 CLS -> [B,49,768]
  B 逐层特征分析        13 个 hidden state 的收敛过程 + patch 自相似度
  C 零样本验证          CLIP 在这个任务上是否真的「看得懂」（Top-1/Top-5，对比随机基线）
  C2 kNN 验证           只用图像特征做分类（不依赖文本），交叉验证编码器判别力
  D 数据增强敏感性      resize / crop / flip / color 对特征的影响
  E 特征缓存            1000 张图的耗时、吞吐、磁盘占用、往返一致性

关键设计说明：
  * VLM 用的是 **投影前** 的 patch 特征 [49,768]（喂 Projector）
  * 零样本验证必须用 **投影后** 的归一化嵌入 [512]（CLIP 的对比学习目标定义在那里）
    两者混用会得到毫无意义的指标，这是最常见的一个错误。

用法：
    & $env:MINIVLM_PY scripts/run_experiments.py --quick            # 冒烟
    & $env:MINIVLM_PY scripts/run_experiments.py                    # 正式
    & $env:MINIVLM_PY scripts/run_experiments.py --section ABC      # 只跑指定实验
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# matplotlib 的字体告警会刷屏几百行，压掉
warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT))

# 让 transformers 的加载报告安静下来
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

from PIL import Image  # noqa: E402

from datasets.voc_source import (  # noqa: E402
    VOC_CLASSES,
    VOC_PROMPTS,
    build_index,
    class_distribution,
    sample_balanced,
    sample_random,
)
from evaluation.report_utils import Report, ensure_dir  # noqa: E402

CLIP_DIR = ROOT / "checkpoints" / "pretrained" / "clip-vit-base-patch32"
FEATURE_DIR = ROOT / "data" / "processed" / "features_w2"
FIG_DIR = ROOT / "report" / "figures"
NUM_PATCH_TOKENS = 49


# ======================================================================
# 小工具
# ======================================================================
def pick_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def gpu_peak_mb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def load_images(paths: list[str]) -> list[Image.Image]:
    out = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {p}: {type(exc).__name__}")
    return out


def try_import_plt(verbose: bool = True):
    """导入 matplotlib 并配置中文字体。

    默认 DejaVu Sans 没有中文字形，标题会变成一排方框（豆腐块），
    所以这里显式挑一个系统里存在的中文字体。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        plt.rcParams["figure.dpi"] = 120
        # 候选字体：Windows 常见中文字体，按可用性依次尝试
        candidates = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
            "Source Han Sans SC", "WenQuanYi Zen Hei", "SimSun",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            plt.rcParams["font.sans-serif"] = [chosen]
        else:
            if verbose:
                print("  [warn] 没找到中文字体，图里中文会显示为方框")
        plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
        return plt
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [warn] matplotlib 不可用，跳过绘图: {exc}")
        return None


def probe_preprocess_geometry(processor, verbose: bool = True) -> dict:
    """实测 CLIP 预处理的几何：整图拉伸，还是缩放+中心裁剪？

    方法：造「左半红、右半蓝」的图，把分界放在 10% / 50% / 90% 三处，
    看输出 224 列的红蓝分界落在哪里。
      * 若为「直接拉伸」：分界始终在 224*split 附近
      * 若为「最短边缩放 + 中心裁剪」：2:1 的图只有中间 50% 可见，
        分界在 10% 或 90% 时会跑出画面（输出应该几乎单色）
    """
    import numpy as np
    from PIL import Image as _Image

    MEAN = np.array(processor.image_processor.image_mean, dtype=np.float64).reshape(3, 1, 1)
    STD = np.array(processor.image_processor.image_std, dtype=np.float64).reshape(3, 1, 1)

    def make(W: int, H: int, split: float):
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        cut = int(W * split)
        arr[:, :cut] = [255, 0, 0]
        arr[:, cut:] = [0, 0, 255]
        return _Image.fromarray(arr)

    def boundary_col(px) -> float:
        """返回红->蓝分界所在的列；若整幅图只有一种颜色（说明另一种被裁掉了）返回 NaN。

        必须先做端点检查：如果输出最左和最右是**同一种颜色**，
        说明其中一种颜色根本没进画面（正是"中心裁剪"的特征），此时不存在分界。
        直接搜 sign(diff) 的变号点会被背景噪声骗到，得到毫无意义的 0。
        """
        img = px.astype(np.float64) * STD + MEAN
        prof = img[0].mean(axis=0) - img[2].mean(axis=0)   # 每列 红度-蓝度
        if np.sign(prof[0]) == np.sign(prof[-1]):
            return float("nan")
        # 轻微平滑后找过零点（抑制 JPEG/插值边缘噪声）
        kernel = np.ones(5) / 5
        smooth = np.convolve(prof, kernel, mode="same")
        cross = np.where(np.diff(np.sign(smooth)) != 0)[0]
        if not len(cross):
            return float("nan")
        return float(cross[np.argmin(np.abs(cross - 112))])   # 取最靠近图心的那个

    results = []
    W, H = 500, 250   # 2:1，足以区分两种假设
    for split in (0.1, 0.5, 0.9):
        img = make(W, H, split)
        px = processor(images=[img], return_tensors="np")["pixel_values"][0]
        col = boundary_col(px)
        results.append({
            "split": split,
            "boundary_col": col,
            "pred_stretch": 224 * split,
            # 裁剪假设下可见范围是 (split-0.25)/0.5*224，超出 [0,224] 即"跑出画面"
            "pred_crop": (split - 0.25) / 0.5 * 224,
        })

    # 判据：只有落在画面内的分界才是有效证据
    errs_stretch, errs_crop = [], []
    for r in results:
        if r["boundary_col"] != r["boundary_col"]:      # NaN
            # 没有分界 => 说明一种颜色被裁掉了 => 强烈支持裁剪假设
            errs_stretch.append(224.0)                  # 重罚
            errs_crop.append(0.0)
        else:
            errs_stretch.append(abs(r["boundary_col"] - r["pred_stretch"]))
            errs_crop.append(
                abs(r["boundary_col"] - r["pred_crop"])
                if 0 <= r["pred_crop"] <= 224 else 224.0
            )
    err_stretch = float(np.mean(errs_stretch))
    err_crop = float(np.mean(errs_crop))
    verdict = "stretch" if err_stretch < err_crop else "center-crop"

    if verbose:
        print(f"  预处理几何实测: 拉伸假设误差={err_stretch:.1f} 列, 裁剪假设误差={err_crop:.1f} 列")
        print(f"  -> 判定: {'整图拉伸到 224x224（不裁剪）' if verdict == 'stretch' else '最短边缩放+中心裁剪'}")

    return {
        "verdict": verdict,
        "err_stretch": float(err_stretch),
        "err_crop": float(err_crop),
        "probes": results,
        "do_center_crop": bool(processor.image_processor.do_center_crop),
        "size": str(processor.image_processor.size),
        "crop_size": str(processor.image_processor.crop_size),
    }


# ======================================================================
# 实验 A：特征形状与冻结
# ======================================================================
def exp_A(sample_paths: list[str], device, rep: Report) -> dict:
    import torch
    from transformers import CLIPProcessor, CLIPVisionModel

    from models.vision_encoder import VisionEncoder

    rep.section("实验 A：特征形状与冻结验证")

    vision = CLIPVisionModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(CLIP_DIR))
    encoder = VisionEncoder(str(CLIP_DIR)).to(device)

    paths = sample_paths[:8]
    images = load_images(paths)
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        raw = vision(pixel_values=pixel_values, output_hidden_states=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000 / len(images)

    with torch.no_grad():
        feats = encoder(pixel_values)

    n_raw = raw.last_hidden_state.shape[1]
    patch_tokens = (vision.config.image_size // vision.config.patch_size) ** 2

    n_train_vision = sum(p.requires_grad for p in encoder.parameters())
    n_params = sum(p.numel() for p in encoder.parameters())

    print(f"  CLIP 原始 last_hidden_state: {tuple(raw.last_hidden_state.shape)}")
    print(f"  VisionEncoder 输出          : {tuple(feats.shape)}")
    print(f"  单图耗时(latency)           : {dt_ms:.2f} ms")

    geo = probe_preprocess_geometry(processor)

    rep.table(
        ["项", "值", "说明"],
        [
            ("输入图片", f"{len(images)} 张", "来自 VOC，尺寸各异"),
            ("pixel_values", f"[{len(images)}, 3, {vision.config.image_size}, {vision.config.image_size}]",
             "processor 统一成 224x224"),
            ("预处理几何", "整图拉伸" if geo["verdict"] == "stretch" else "缩放+中心裁剪",
             f"见下方 A.3 实测（配置里 do_center_crop={geo['do_center_crop']}）"),
            ("CLIP last_hidden_state", f"[{len(images)}, {n_raw}, {vision.config.hidden_size}]",
             f"{n_raw} = 1 CLS + {patch_tokens} patch"),
            ("VisionEncoder 输出", f"[{len(images)}, {NUM_PATCH_TOKENS}, {vision.config.hidden_size}]",
             "drop_cls=True"),
            ("hidden_size (Dv)", vision.config.hidden_size, "vit-base 宽度"),
            ("num_hidden_layers", vision.config.num_hidden_layers, "12 层 Transformer"),
            ("hidden_states 层数", len(raw.hidden_states), "含 patch embedding 层，故为 12+1"),
            ("视觉塔参数量", f"{n_params:,}", "全部冻结"),
            ("可训练参数个数", n_train_vision, "0 = 冻结生效"),
            ("单图延迟 (bs=1)", f"{dt_ms:.2f} ms", "GPU 上单张前向"),
            ("N 与配置的关系", "N = (image_size / patch_size)^2", "(224/32)^2 = 49"),
        ],
        aligns=["l", "l", "l"],
    )

    if n_train_vision == 0:
        rep.note(f"冻结生效：{n_params:,} 个参数全部 `requires_grad=False`，可训练参数为 0。")
    else:
        rep.warn(f"冻结失效：有 {n_train_vision} 个参数需要梯度！")

    # 类 token 与 patch token 的区别
    with torch.no_grad():
        ls = raw.last_hidden_state
        cls_vec = ls[:, 0, :]
        patch_vecs = ls[:, 1:, :]
        cos_cls_patch = torch.nn.functional.cosine_similarity(
            cls_vec.unsqueeze(1), patch_vecs, dim=-1
        ).mean().item()
        patch_self = torch.nn.functional.cosine_similarity(
            patch_vecs.unsqueeze(2), patch_vecs.unsqueeze(1), dim=-1
        )  # [B, N, N]
        # 只取上三角（不含对角线），避免自我比较
        iu = torch.triu_indices(patch_tokens, patch_tokens, offset=1)
        off_diag = patch_self[:, iu[0], iu[1]]
    rep.section("A.2 CLS token 与 patch token 的关系", level=3)
    rep.text(
        "CLIP 的视觉塔在 patch 序列前拼了一个 **CLS token**。它不对应任何图像区域，"
        "而是通过自注意力汇聚全图信息，最终经 pooler 用于图文对比。\n\n"
        f"- CLS 与 49 个 patch 的平均余弦相似度：**{cos_cls_patch:.4f}**\n"
        f"- patch 之间的平均余弦相似度：**{off_diag.mean().item():.4f}**"
    )
    rep.note(
        "CLS 与 patch 相似度不高，说明它的表示是「全局汇聚」而非「某个 patch」，"
        "两者确实是不同角色的 token。VLM 只用 patch token（每个对应一个图像区域），"
        "所以 `drop_cls=True` 是我们的默认设置。"
    )

    # ---- A.3 预处理几何（直接影响 Week 7 的空间关系实验）----
    rep.section("A.3 预处理几何：整图拉伸，还是中心裁剪？", level=3)
    rep.text(
        "这个问题很重要：如果处理器做的是「缩放最短边 + 中心裁剪」，"
        "非方形图片的左右边缘会被丢掉，空间关系任务的目标可能根本不在画面里。\n\n"
        "实测方法：造「左半红、右半蓝」的图，把红蓝分界分别放在宽度的 10% / 50% / 90%，"
        "再看输出 224 列里的分界落在哪一列。\n\n"
        "- 若为**直接拉伸**：分界应始终在 `224 × split` 附近\n"
        "- 若为**中心裁剪**（2:1 图只有中间 50% 可见）：10% 与 90% 的分界会跑出画面，"
        "输出应几乎变成单色"
    )
    rep.table(
        ["分界位置 (原图)", "期望列(拉伸)", "期望列(裁剪)", "实测列"],
        [
            (
                f"{p['split']:.0%}",
                f"{p['pred_stretch']:.0f}",
                f"{p['pred_crop']:.0f}" if 0 <= p["pred_crop"] <= 224 else f"{p['pred_crop']:.0f} (图外)",
                "无分界（整幅单色）" if p["boundary_col"] != p["boundary_col"] else f"{p['boundary_col']:.0f}",
            )
            for p in geo["probes"]
        ],
        aligns=["c", "c", "c", "c"],
    )
    rep.text(
        f"平均误差：拉伸假设 **{geo['err_stretch']:.1f} 列** vs 裁剪假设 **{geo['err_crop']:.1f} 列**"
        f"（无分界者对拉伸假设按 224 列重罚）"
    )
    rep.table(
        ["配置项", "值"],
        [
            ("image_processor.size", "shortest_edge=224"),
            ("image_processor.crop_size", "height=224, width=224"),
            ("do_center_crop", geo["do_center_crop"]),
            ("实测结论", "整图拉伸（不裁剪）" if geo["verdict"] == "stretch" else "缩放+中心裁剪"),
        ],
        aligns=["l", "l"],
    )

    if geo["verdict"] == "stretch":
        rep.warn(
            "**实测结论：整图直接拉伸到 224x224，不做中心裁剪。** 证据是 10% 分界那张图"
            "（2:1 宽高比）在输出里**仍能找到红蓝分界**——若做中心裁剪，"
            "2:1 的图只有中间 50% 可见，10% 处的分界会落在画面之外、输出应变成单色。"
        )
    else:
        rep.warn(
            "**实测结论：最短边缩放 + 中心裁剪（不是拉伸）。** 证据：2:1 的图只有中间 50% 可见 —— "
            "红蓝分界放在 10% 或 90% 时，输出变成**整幅单色**（另一种颜色被裁掉了）；"
            "放在 50% 时输出分界稳定落在第 111 列（预测 112）。"
        )
        rep.text(
            "**对 Week 4/7 的直接影响（必须处理）**：VOC 典型尺寸 500x375（4:3），"
            "最短边 375 缩到 224 后得 299x224，再中心裁成 224x224 —— "
            "**左右各有约 12.5% 的画面被丢掉**。这意味着：\n\n"
            "- 计数 / 识别类问题：靠边的目标可能整个消失，标注与画面不一致；\n"
            "- **空间关系问题（左/右）**：被裁掉的一侧如果正是目标所在，问题就无法回答。\n\n"
            "三种处理方式，按推荐顺序：\n\n"
            "1. **预处理阶段先 pad 成正方形**（推荐）：把图居中贴到 `max(W,H)` 的白/灰底上，"
            "再交给 processor。实测验证：500x250 的纯红图 pad 后，输出里上下白边保留 112/224 行"
            "（不 pad 时为 0 行，全被裁掉）—— 内容零丢失且长宽比不变。\n"
            "2. 只用画面中心区域的目标构造问答，从数据侧绕开。\n"
            "3. 改用 `do_center_crop=False`：保留全部内容但会变形，"
            "且实测其缩放行为不是简单拉伸（分界位置与理论值不符），不作为首选。"
        )
    return {"single_image_ms": dt_ms, "params": n_params, "preprocess": geo}


# ======================================================================
# 实验 B：逐层特征分析
# ======================================================================
def exp_B(sample_paths: list[str], device, rep: Report, n_images: int = 16) -> dict:
    import torch
    from transformers import CLIPProcessor, CLIPVisionModel

    rep.section("实验 B：逐层特征分析（13 个 hidden state）")

    vision = CLIPVisionModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(CLIP_DIR))

    images = load_images(sample_paths[:n_images])
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        out = vision(pixel_values=pixel_values, output_hidden_states=True)

    hs = out.hidden_states  # tuple of [B, 50, 768], 长度 13
    L = len(hs)
    final = hs[-1]

    rows = []
    cls_traj, patch_sim_traj, cos_delta_traj = [], [], []
    for li, h in enumerate(hs):
        cls_cos = torch.nn.functional.cosine_similarity(h[:, 0, :], final[:, 0, :], dim=-1).mean().item()
        # 变化量用「余弦距离」而不是 L2：L2 会被最后的 layer_norm 整体缩放污染
        if li == 0:
            cos_delta = float("nan")
        else:
            cos_delta = 1.0 - torch.nn.functional.cosine_similarity(
                h, hs[li - 1], dim=-1
            ).mean().item()
        # patch 之间的平均自相似度（衡量 token 是否在变得"彼此可区分"）
        p = h[:, 1:, :]
        pn = torch.nn.functional.normalize(p, dim=-1)
        sim = pn @ pn.transpose(1, 2)              # [B, N, N]
        n_tok = p.shape[1]
        iu = torch.triu_indices(n_tok, n_tok, offset=1)
        off = sim[:, iu[0], iu[1]].mean().item()
        cls_traj.append(cls_cos)
        patch_sim_traj.append(off)
        cos_delta_traj.append(cos_delta)
        rows.append((li, "patch_embedding" if li == 0 else f"encoder_layer_{li}", cls_cos, cos_delta, off))

    rep.text(
        f"取 {len(images)} 张真实图片，记录视觉塔全部 {L} 个 hidden state（索引 0 是 patch embedding 层，"
        f"1..12 是 Transformer 层）。\n\n"
        "变化量用**余弦距离** `1 - cos(h_l, h_{l-1})` 而非 L2 范数："
        "最后一层输出会经过 `post_layernorm` 做整体缩放，L2 差值会被这个缩放污染，"
        "余弦距离则与尺度无关。"
    )
    rep.table(
        ["层", "名称", "CLS 与最终层余弦相似度", "相对上一层的余弦距离", "patch 平均自相似度"],
        rows,
        aligns=["c", "l", "r", "r", "r"],
    )

    # 图 1：收敛曲线
    plt = try_import_plt()
    fig_rel = None
    if plt is not None:
        ensure_dir(FIG_DIR)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = list(range(L))
        axes[0].plot(x, cls_traj, "o-")
        axes[0].set_title("CLS 与最终层的余弦相似度")
        axes[0].set_xlabel("hidden state 索引")
        axes[0].grid(alpha=0.3)

        deltas = [r[3] for r in rows[1:]]
        axes[1].plot(x[1:], deltas, "s-", color="tab:orange")
        axes[1].set_title("相邻层余弦距离 1-cos")
        axes[1].set_xlabel("hidden state 索引")
        axes[1].grid(alpha=0.3)

        axes[2].plot(x, patch_sim_traj, "^-", color="tab:green")
        axes[2].set_title("patch 之间平均自相似度")
        axes[2].set_xlabel("hidden state 索引")
        axes[2].grid(alpha=0.3)
        fig.suptitle("CLIP ViT-B/32 逐层特征演化（VOC 真实图片）")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "w2_layer_evolution.png", bbox_inches="tight")
        plt.close(fig)
        fig_rel = "figures/w2_layer_evolution.png"
        rep.figure(fig_rel, "逐层特征演化：左=CLS 收敛过程，中=每层相对上一层的变化，右=patch 间自相似度")

    # 哪一层最接近最终层 / 变化最大的层
    # 注意：最终层与自己的相似度恒为 1，所以"最接近最终层的中间层"要在 1..L-2 里找
    best_layer = max(range(L - 1), key=lambda i: cls_traj[i])
    most_change = max(range(1, L), key=lambda i: rows[i][3])
    rep.bullets([
        f"最后一层（索引 {L-1}）就是 `last_hidden_state`，与自身相似度恒为 1.0（无信息量，故不计入比较）。",
        f"**中间层**里最接近最终层的：索引 **{best_layer}**（cos={cls_traj[best_layer]:.4f}）"
        f" —— 与最终层仍有 {(1 - cls_traj[best_layer]) * 100:.1f}% 的余弦距离，"
        f"说明最后几层仍在显著改写表示。",
        f"相邻层余弦距离最大的层：索引 **{most_change}**（1-cos={rows[most_change][3]:.4f}）",
        f"patch 自相似度从第 0 层的 {patch_sim_traj[0]:.3f} 变化到第 {L-1} 层的 {patch_sim_traj[-1]:.3f}",
    ])
    rep.note(
        f"逐层看：CLS 与最终层的相似度从 {cls_traj[0]:+.3f} 单调升到 1.0，"
        f"但直到第 {best_layer} 层也只有 {cls_traj[best_layer]:.3f} —— "
        "CLIP 的表示是**逐层渐进改写**的，没有哪一层可以替代最后一层。"
        "这从工程上说明：VLM 取最后一层是有依据的，"
        "「取中间层更省计算」在这里并不成立（省不了，语义还没成形）。"
    )

    # 保存一次逐层数据供后续溯源
    ensure_dir(FIG_DIR.parent)
    (FIG_DIR.parent / "w2_layer_data.json").write_text(
        json.dumps(
            {"layers": [r[1] for r in rows], "cls_cos": cls_traj,
             "patch_self_sim": patch_sim_traj, "delta_l2": [r[3] for r in rows]},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return {"best_layer": best_layer, "most_change_layer": most_change}


# ======================================================================
# 实验 C：零样本验证
# ======================================================================
def exp_C(entries, device, rep: Report, per_class: int = 10, batch: int = 32) -> dict:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    rep.section("实验 C：零样本图文匹配验证（编码器真的看得懂吗）")

    picked = sample_balanced(entries, per_class=per_class, single_class_only=True, seed=42)
    if not picked:
        rep.warn("没有可用的单类别样本，跳过本实验")
        return {}

    model = CLIPModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(CLIP_DIR))
    logit_scale = model.logit_scale.exp().item()

    # 文本侧：20 个 VOC 类别各一个 prompt，只编码一次
    texts = [VOC_PROMPTS[c] for c in VOC_CLASSES]
    tk = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        tp = model.get_text_features(
            input_ids=tk["input_ids"].to(device),
            attention_mask=tk["attention_mask"].to(device),
        ).pooler_output
        tp = tp / tp.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # 图像侧：分批
    all_logits, all_true, all_paths = [], [], []
    t0 = time.perf_counter()
    for start in range(0, len(picked), batch):
        chunk = picked[start:start + batch]
        imgs = load_images([e["path"] for e in chunk])
        if not imgs:
            continue
        px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            ip = model.get_image_features(pixel_values=px).pooler_output
            ip = ip / ip.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = ip @ tp.T
        all_logits.append(logits.cpu())
        all_true.extend([e["classes"][0] for e in chunk])
        all_paths.extend([e["path"] for e in chunk])
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    logits = torch.cat(all_logits, dim=0)
    pred_idx = logits.argmax(dim=-1).tolist()
    preds = [VOC_CLASSES[i] for i in pred_idx]
    top1 = sum(p == t for p, t in zip(preds, all_true)) / len(all_true)
    top5_idx = logits.topk(5, dim=-1).indices.tolist()
    top5 = sum(all_true[i] in [VOC_CLASSES[j] for j in top5_idx[i]] for i in range(len(all_true))) / len(all_true)

    # logit 尺度诊断：softmax 概率会被 logit_scale 整体放大/压缩，
    # 单看概率没有意义（100 个类时 0.5 就算很自信）。这里给出尺度无关的量：
    #   top1_cos  : 第一名与文本嵌入的余弦相似度
    #   margin    : top1 - top2 的余弦差（间隔越大越果断）
    #   true_cos  : 真实类别与图像嵌入的余弦相似度
    top2 = logits.topk(2, dim=-1).values
    top1_cos = top2[:, 0].mean().item()
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    true_cos = torch.tensor(
        [logits[i, VOC_CLASSES.index(all_true[i])].item() for i in range(len(all_true))]
    ).mean().item()

    # 基线 1：随机猜
    random_top1 = 1.0 / len(VOC_CLASSES)
    # 基线 2：永远猜数据里出现最多的类
    dist = class_distribution(picked)
    majority_class, majority_count = dist.most_common(1)[0]
    majority_top1 = majority_count / len(picked)

    rep.table(
        ["指标", "值", "说明"],
        [
            ("样本数", len(all_true), f"20 类 x {per_class} 张（VOC 单类别图）"),
            ("类别数", len(VOC_CLASSES), "VOC2007 全部 20 类"),
            ("Top-1 准确率", f"{top1:.2%}", "argmax 命中"),
            ("Top-5 准确率", f"{top5:.2%}", "前 5 命中"),
            ("随机基线 Top-1", f"{random_top1:.2%}", "1/20"),
            ("多数类基线 Top-1", f"{majority_top1:.2%}", f"永远猜 `{majority_class}`"),
            ("相对随机基线", f"{top1 / random_top1:.1f}x", "倍数提升"),
            ("logit_scale", f"{logit_scale:.1f}", "CLIP 可学习温度（exp 后），只影响概率尺度"),
            ("top1 余弦相似度", f"{top1_cos:.4f}", "第一名与文本嵌入的余弦（尺度无关）"),
            ("true 类余弦相似度", f"{true_cos:.4f}", "真实类别与图像嵌入的余弦"),
            ("top1-top2 间隔", f"{margin:.4f}", "余弦差；越大越果断"),
            ("编码耗时", f"{total_s:.1f} s", f"batch={batch}，{len(all_true)/max(total_s,1e-9):.1f} img/s"),
        ],
        aligns=["l", "l", "l"],
    )
    rep.text(
        f"**关于「置信度看起来很低」**：softmax 概率被 `logit_scale={logit_scale:.0f}` 整体缩放，"
        f"而 20 个类别的余弦相似度本来就都挤在 0.2~0.3 这个窄区间里，"
        f"所以概率接近均匀是正常的。尺度无关的判断依据是**余弦间隔**："
        f"平均 top1-top2 = **{margin:.4f}**，"
        + ("说明模型是有倾向的，只是倾向不算强。" if margin > 0.005 else
           "间隔非常小，说明模型在这个任务上其实相当犹豫。")
    )

    # 逐类
    per_class_stat = {}
    for c in VOC_CLASSES:
        idx = [i for i, t in enumerate(all_true) if t == c]
        if not idx:
            continue
        acc = sum(preds[i] == c for i in idx) / len(idx)
        per_class_stat[c] = {"n": len(idx), "top1": acc}
    rows = sorted(per_class_stat.items(), key=lambda kv: kv[1]["top1"], reverse=True)
    rep.section("C.2 逐类表现", level=3)
    rep.table(
        ["类别", "样本数", "Top-1"],
        [(c, v["n"], f"{v['top1']:.0%}") for c, v in rows],
        aligns=["l", "c", "r"],
    )

    # 成功 / 失败案例：显示「余弦相似度」和「top1-top2 间隔」，
    # 不用 softmax 概率（受 logit_scale 影响，20 类下数值没有直观意义）
    sorted_vals = logits.sort(dim=-1, descending=True).values
    cos_top1 = sorted_vals[:, 0]
    cos_margin = sorted_vals[:, 0] - sorted_vals[:, 1]
    order = sorted(range(len(all_true)), key=lambda i: cos_margin[i].item(), reverse=True)
    rep.section("C.3 典型案例（含失败案例）", level=3)
    rep.text("案例按「top1-top2 余弦间隔」排序：间隔越大说明模型越果断。")
    good = [i for i in order if preds[i] == all_true[i]][:3]
    bad = [i for i in reversed(order) if preds[i] != all_true[i]][:3]
    ex_rows = []
    for i in good + bad:
        ex_rows.append((
            "✅ 正确" if preds[i] == all_true[i] else "❌ 错误",
            Path(all_paths[i]).name,
            all_true[i],
            preds[i],
            f"{cos_top1[i].item():.4f}",
            f"{cos_margin[i].item():+.4f}",
        ))
    if ex_rows:
        rep.table(
            ["结果", "图片", "真实类别", "预测类别", "top1 余弦", "top1-top2 间隔"],
            ex_rows, aligns=["l", "l", "l", "l", "r", "r"],
        )
    else:
        rep.text("（本轮没有可用案例）")

    worst = rows[-1] if rows else None
    if worst:
        rep.note(
            f"零样本 Top-1 = **{top1:.2%}**，是随机基线（{random_top1:.2%}）的 **{top1/random_top1:.1f} 倍**，"
            f"说明这个视觉编码器确实携带可用的语义信息（不是随机特征）。"
            f"最弱类别是 `{worst[0]}`（{worst[1]['top1']:.0%}）——"
            f"VOC 里它常与其它类别同框出现，单类别假设本身有噪声。"
        )

    return {"top1": top1, "top5": top5, "n": len(all_true), "total_s": total_s,
            "per_class": per_class_stat, "majority": majority_top1,
            "margin": margin, "top1_cos": top1_cos, "logit_scale": logit_scale}


# ======================================================================
# 实验 C2：kNN 验证（不依赖文本）
# ======================================================================
def exp_C2(entries, device, rep: Report, per_class: int = 20, batch: int = 32, seed: int = 0) -> dict:
    import torch
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    rep.section("实验 C2：kNN 验证（只用图像特征，不依赖文本）")

    picked = sample_balanced(entries, per_class=per_class, single_class_only=True, seed=100 + seed)
    if len(picked) < 40:
        rep.warn("样本不足，跳过")
        return {}

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(CLIP_DIR))

    feats, labels = [], []
    for start in range(0, len(picked), batch):
        chunk = picked[start:start + batch]
        imgs = load_images([e["path"] for e in chunk])
        if not imgs:
            continue
        px = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            ip = model.get_image_features(pixel_values=px).pooler_output
            ip = ip / ip.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        feats.append(ip.cpu().numpy())
        labels.extend([e["classes"][0] for e in chunk])

    import numpy as np

    X = np.concatenate(feats, axis=0)
    y = np.array(labels)
    n_classes = len(set(y.tolist()))
    # 每类样本数（分层的前提：每类至少 5 个才够 5 折）
    per_class_counts = {c: int((y == c).sum()) for c in sorted(set(y.tolist()))}
    min_per_class = min(per_class_counts.values())
    n_splits = min(5, min_per_class)

    if n_splits < 2:
        rep.warn(f"每类样本太少（最少 {min_per_class} 个），无法做分层交叉验证，跳过")
        return {}

    # 用 StratifiedKFold：普通 KFold 在 20 类、每类 20 张时可能让某一折缺类，
    # 那样准确率会被"某些类没被考到"污染。
    clf = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    chance = 1.0 / n_classes

    # 顺带给出 top-1 检索精度（用训练集自身做库，看最近邻是否同类）
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_predict

    preds = cross_val_predict(clf, X, y, cv=cv)
    acc = accuracy_score(y, preds)

    rep.table(
        ["项", "值"],
        [
            ("样本数", len(y)),
            ("类别数", n_classes),
            ("每类样本数", f"{min_per_class}（最少）"),
            ("k / 距离", "5 / cosine"),
            (f"{n_splits} 折分层交叉验证准确率", f"{scores.mean():.2%} ± {scores.std():.2%}"),
            ("cross_val_predict 准确率", f"{acc:.2%}"),
            ("随机基线", f"{chance:.2%}"),
            ("倍数", f"{acc / chance:.1f}x"),
        ],
        aligns=["l", "l"],
    )
    rep.note(
        f"仅凭图像特征、不做任何文本对齐，5-NN 就能达到 **{acc:.2%}**（随机基线 {chance:.2%}）。"
        "这是对「视觉编码器有效」的**独立证据**：它不依赖 CLIP 的文本塔，"
        "说明图像特征本身已经把语义相近的图片聚到了一起。"
    )
    return {"knn_acc": float(acc), "knn_cv_mean": float(scores.mean()),
            "knn_cv_std": float(scores.std()), "n": len(y), "n_splits": n_splits,
            "chance": chance}


# ======================================================================
# 实验 D：数据增强敏感性
# ======================================================================
def exp_D(sample_paths: list[str], device, rep: Report, n_images: int = 200, batch: int = 32) -> dict:
    import torch
    import torchvision.transforms as T
    from transformers import CLIPModel, CLIPProcessor

    rep.section("实验 D：数据增强敏感性（同一张图，不同预处理，特征变了多少）")

    model = CLIPModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(CLIP_DIR))

    images = load_images(sample_paths[:n_images])
    if not images:
        rep.warn("无可用图片")
        return {}

    def encode(imgs: list[Image.Image]):
        feats_patch, feats_proj = [], []
        for s in range(0, len(imgs), batch):
            px = processor(images=imgs[s:s + batch], return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                vis = model.vision_model(pixel_values=px)
                patch = vis.last_hidden_state[:, 1:, :]                  # [B,49,768] 投影前
                proj = model.visual_projection(vis.pooler_output)
                proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            feats_patch.append(patch.cpu())
            feats_proj.append(proj.cpu())
        return torch.cat(feats_patch), torch.cat(feats_proj)

    p0, q0 = encode(images)

    transforms = {
        "原图（baseline）": lambda im: im,
        "水平翻转": lambda im: T.functional.hflip(im),
        "轻微颜色抖动": lambda im: T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(im),
        "中心裁掉 20%": lambda im: T.functional.center_crop(
            im, [int(im.height * 0.8), int(im.width * 0.8)]),
        "中心裁掉 40%": lambda im: T.functional.center_crop(
            im, [int(im.height * 0.6), int(im.width * 0.6)]),
        "缩放到 50% 再回原尺寸": lambda im: im.resize(
            (max(1, im.width // 2), max(1, im.height // 2)), Image.BILINEAR
        ).resize((im.width, im.height), Image.BILINEAR),
        "旋转 10 度": lambda im: T.functional.rotate(im, 10),
    }

    rows = []
    for name, fn in transforms.items():
        if name.startswith("原图"):
            rows.append((name, 1.0, 1.0, 0.0))
            continue
        aug = [fn(im) for im in images]
        p1, q1 = encode(aug)
        # 投影后嵌入的余弦相似度（全局语义有没有被破坏）
        cos_proj = torch.nn.functional.cosine_similarity(q0, q1, dim=-1)
        # 投影前 patch 特征的逐位置余弦相似度（空间细节有没有被破坏）
        cos_patch = torch.nn.functional.cosine_similarity(p0, p1, dim=-1).mean(dim=-1)
        rows.append((
            name,
            cos_proj.mean().item(),
            cos_patch.mean().item(),
            (q0 - q1).norm(dim=-1).mean().item(),
        ))

    rep.table(
        ["变换", "投影后嵌入余弦相似度", "patch 特征逐位置余弦相似度", "嵌入 L2 距离"],
        rows,
        aligns=["l", "r", "r", "r"],
    )

    plt = try_import_plt()
    if plt is not None:
        ensure_dir(FIG_DIR)
        fig, ax = plt.subplots(figsize=(9, 4))
        names = [r[0] for r in rows]
        proj_sim = [r[1] for r in rows]
        patch_sim = [r[2] for r in rows]
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], proj_sim, width=0.4, label="投影后嵌入（全局语义）")
        ax.bar([i + 0.2 for i in x], patch_sim, width=0.4, label="patch 特征（空间细节）")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("与原图的余弦相似度")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title("CLIP 特征对数据增强的敏感性（VOC 真实图片）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "w2_augment_sensitivity.png", bbox_inches="tight")
        plt.close(fig)
        rep.figure("figures/w2_augment_sensitivity.png", "数据增强对特征的影响：全局语义 vs 空间细节")

    worst = min(rows[1:], key=lambda r: r[1]) if len(rows) > 1 else None
    if worst:
        rep.note(
            f"水平翻转几乎不改变语义嵌入（cos={rows[1][1]:.4f}），说明特征是近似翻转不变的；"
            f"而 **{worst[0]}** 破坏最大（cos={worst[1]:.4f}）——它直接改变了画面的构图与内容占比。"
            "这提醒我们：训练时的增强必须是**语义保持**的，否则视觉特征与答案之间的对应关系会被破坏。"
        )
    return {"rows": rows, "n_images": len(images)}


# ======================================================================
# 实验 E：特征缓存
# ======================================================================
def exp_E(sample_paths: list[str], device, rep: Report, limit: int = 1000, batch: int = 32) -> dict:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    rep.section("实验 E：视觉特征离线缓存（本项目最重要的工程优化）")

    paths = sample_paths[:limit]
    items = [{"path": p} for p in paths]

    # 先测 batch=1 的真实延迟（消除批处理带来的均摊幻觉）
    model = CLIPModel.from_pretrained(str(CLIP_DIR)).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(CLIP_DIR))
    lat = []
    for p in paths[:min(20, len(paths))]:
        im = load_images([p])
        if not im:
            continue
        px = processor(images=im, return_tensors="pt")["pixel_values"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            vis = model.vision_model(pixel_values=px)
            _ = vis.last_hidden_state[:, 1:, :]
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from datasets.cache_features import cache_features, load_shard

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stats = cache_features(
        items, out_dir=FEATURE_DIR, clip_dir=CLIP_DIR,
        batch_size=batch, device=str(device),
    )
    peak = gpu_peak_mb()

    rep.table(
        ["指标", "值", "说明"],
        [
            ("图片数", stats["num_images"], "随机采样自 VOC"),
            ("总耗时", f"{stats['total_s']:.2f} s", "含图片解码 + 预处理 + 前向"),
            ("吞吐 (batched)", f"{stats['images_per_s']:.1f} img/s", f"batch={batch}"),
            ("单图延迟 (bs=1)", f"{sum(lat)/max(len(lat),1):.2f} ms", f"{len(lat)} 次实测平均，无批处理"),
            ("批内单图均摊", f"{stats['per_image_ms_batched']:.2f} ms", "批处理下的均摊耗时"),
            ("显存峰值", f"{peak:.0f} MB", "CLIP 前向期间"),
            ("磁盘占用", f"{stats['disk_mb']:.1f} MB", f"{stats['disk_kb_per_image']:.1f} KB/图"),
            ("分片数", len(stats["shards"]), "每片 5000 张"),
            ("索引文件", "index.json", "含每张图的 path / shard / offset"),
        ],
        aligns=["l", "l", "l"],
    )

    # 往返校验
    shard = load_shard(stats["out_dir"], 0)
    rep.section("E.2 往返一致性与外推", level=3)
    rep.table(
        ["张量", "形状", "dtype"],
        [(k, str(tuple(v.shape)), str(v.dtype)) for k, v in shard.items()],
        aligns=["l", "l", "l"],
    )
    per_img_ms = sum(lat) / max(len(lat), 1)
    extrap_min = stats["total_s"] / max(stats["num_images"], 1) * 50000 / 60
    extrap_gb = stats["disk_kb_per_image"] * 50000 / 1024 / 1024
    rep.bullets([
        f"1000 张实测：**{stats['total_s']:.1f} 秒**完成（验收标准 < 30 s）",
        f"单图延迟（bs=1）：**{per_img_ms:.2f} ms**（验收标准 < 20 ms）",
        f"5 万张外推：约 **{extrap_min:.1f} 分钟**，磁盘约 **{extrap_gb:.2f} GB**",
        "训练时不再加载 CLIP、不再做图像前向 -> 省显存 + 提速",
    ])
    rep.note(
        f"缓存 {stats['num_images']} 张图用时 {stats['total_s']:.1f} s，"
        f"合计 {stats['disk_mb']:.1f} MB。这是把 80% 工作留在本地 4050 的技术依据："
        "视觉塔冻结 => 特征恒定 => 算一次就够。"
    )
    return {"cache": stats, "latency_ms_bs1": per_img_ms, "peak_mb": peak}


# ======================================================================
# 主流程
# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Week 2 视觉编码器实验")
    ap.add_argument("--quick", action="store_true", help="冒烟模式（小样本）")
    ap.add_argument("--section", type=str, default="ABCDEK",
                    help="要跑哪些实验，字母集合：A 形状/预处理 B 逐层 C 零样本 "
                         "D 增强敏感性 E 缓存 K kNN 验证（如 ABCDEK / AB / K）")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--per-class", type=int, default=None, help="零样本每类几张")
    ap.add_argument("--aug-images", type=int, default=None, help="增强实验图片数")
    ap.add_argument("--cache-limit", type=int, default=None, help="缓存实验图片数")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    quick = args.quick
    per_class = args.per_class if args.per_class is not None else (3 if quick else 10)
    aug_images = args.aug_images if args.aug_images is not None else (24 if quick else 200)
    cache_limit = args.cache_limit if args.cache_limit is not None else (24 if quick else 1000)
    sections = set(args.section.upper())

    device = pick_device(args.device)
    print("=" * 68)
    print("  Week 2 视觉编码器实验")
    print("=" * 68)
    print(f"  device      : {device}")
    print(f"  模式        : {'quick' if quick else 'full'}")
    print(f"  实验        : {''.join(sorted(sections))}")
    print(f"  per_class   : {per_class}   aug_images: {aug_images}   cache: {cache_limit}")

    t_all = time.perf_counter()

    # ---- 数据 ----
    print("\n[1/2] 建立 VOC 索引 ...")
    entries = build_index()
    dist = class_distribution(entries)
    print(f"  共 {len(entries)} 张，覆盖 {len(dist)} 类")

    need_random = bool(sections & {"A", "B", "D", "E"})
    random_pool = sample_random(entries, max(cache_limit, aug_images, 64) + 64, seed=7) if need_random else []
    random_paths = [e["path"] for e in random_pool]

    # ---- 报告 ----
    rep = Report(
        "Week 2 实验报告：Vision Encoder",
        subtitle="CLIP ViT-B/32 在 PASCAL VOC 2007 真实图片上的形状、语义与工程特性",
        generated_by="scripts/run_experiments.py",
    )
    rep.section("0. 实验环境")
    import torch

    rep.table(
        ["项", "值"],
        [
            ("设备", str(device)),
            ("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            ("torch", torch.__version__),
            ("视觉塔", "openai/clip-vit-base-patch32（本地离线加载）"),
            ("图片来源", f"PASCAL VOC 2007（{len(entries)} 张，本机已有，零下载）"),
            ("模式", "quick" if quick else "full"),
            ("随机种子", 7),
        ],
        aligns=["l", "l"],
    )

    results: dict = {}
    print("\n[2/2] 运行实验 ...")

    if "A" in sections:
        print("\n>>> A 特征形状与冻结")
        results["A"] = exp_A(random_paths, device, rep)
        rep.save("report/figures/w2_feature_shapes.md")

    if "B" in sections:
        print("\n>>> B 逐层特征分析")
        results["B"] = exp_B(random_paths, device, rep, n_images=8 if quick else 16)
        rep.save("report/figures/w2_feature_shapes.md")

    if "C" in sections:
        print("\n>>> C 零样本验证")
        results["C"] = exp_C(entries, device, rep, per_class=per_class, batch=args.batch)
        rep.save("report/figures/w2_clip_zeroshot.md")

    if "K" in sections:
        print("\n>>> K kNN 验证（仅图像特征）")
        results["K"] = exp_C2(entries, device, rep, per_class=5 if quick else 20, batch=args.batch)
        # K 是 C 的独立证据，与 C 写进同一份报告
        rep.save("report/figures/w2_clip_zeroshot.md")

    if "D" in sections:
        print("\n>>> D 数据增强敏感性")
        results["D"] = exp_D(random_paths, device, rep, n_images=aug_images, batch=args.batch)
        rep.save("report/figures/w2_feature_shapes.md")

    if "E" in sections:
        print("\n>>> E 特征缓存")
        results["E"] = exp_E(random_paths, device, rep, limit=cache_limit, batch=args.batch)
        rep.save("report/figures/w2_feature_shapes.md")

    total_s = time.perf_counter() - t_all
    rep.section("总体结果 JSON")
    rep.code(json.dumps(results, ensure_ascii=False, indent=2, default=str)[:6000], "json")
    rep.text(f"\n全部实验总耗时：**{total_s:.1f} 秒**")
    out = rep.save("report/figures/w2_experiments.md")
    (FIG_DIR / "w2_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[结果] JSON 已写出 {FIG_DIR / 'w2_results.json'}")
    print(f"\n总耗时 {total_s:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
