#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/verify_offline.py —— 离线加载验收（"离线变量固定"这一步的最终验收）

用本地目录路径直接加载 CLIP 与 Qwen，完全绕开 hub：
  - 若离线变量没设好 / 模型不完整，这里会直接失败（快速失败，不浪费时间）
  - 顺便实证三件事（结论写进报告，避免 off-by-one 返工）：
      (a) CLIPVisionModel.last_hidden_state 到底含不含 CLS
      (b) VisionEncoder 的输出 token 数是否稳定正确
      (c) Qwen tokenizer 的 <|image_pad|> 是否存在、是否越界、是否需要 resize

本地实测结论（2026-09 于 RTX 4050 / transformers 5.17.0）:
  * CLIP last_hidden_state = [B, 50, 768]，第 0 个是 CLS，丢 CLS 后 49 个 patch
  * <|image_pad|> id = 151655，embedding 表 151936 行 → 不越界，无需 resize
  * CLIP 的 pooler_output 不是第 0 个 hidden state（原实现误以为它是），
    所以判定 CLS 只能用 token 数，不能用 pooler 对比。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT))

PRETRAINED = Path(os.environ.get("MINIVLM_PRETRAINED", ROOT / "checkpoints" / "pretrained"))
CLIP_DIR = PRETRAINED / "clip-vit-base-patch32"
QWEN_DIR = PRETRAINED / "qwen2.5-0.5b-instruct"

FAIL: list[str] = []
WARN: list[str] = []


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    FAIL.append(msg)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    WARN.append(msg)


def sec(t: str) -> None:
    print("\n" + "-" * 62)
    print(f"  {t}")
    print("-" * 62)


def main() -> int:
    import torch
    from transformers import CLIPVisionModel, AutoTokenizer

    print("离线变量:", {k: os.environ.get(k) for k in
                     ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")})

    # ------------------------------------------------------------------
    sec("A. CLIP 视觉塔：离线加载 + 形状实证")
    # ------------------------------------------------------------------
    vision = CLIPVisionModel.from_pretrained(str(CLIP_DIR))
    vision.eval()
    print(f"  加载自 {CLIP_DIR}")
    print(f"  image_size={vision.config.image_size}  patch_size={vision.config.patch_size}  "
          f"hidden={vision.config.hidden_size}  layers={vision.config.num_hidden_layers}")

    px = torch.randn(2, 3, vision.config.image_size, vision.config.image_size)
    with torch.no_grad():
        out = vision(pixel_values=px, output_hidden_states=True)

    expected_patches = (vision.config.image_size // vision.config.patch_size) ** 2
    print(f"  last_hidden_state.shape = {tuple(out.last_hidden_state.shape)}")
    print(f"  理论 patch 数            = {expected_patches}")
    print(f"  hidden_states 层数       = {len(out.hidden_states)}")

    # 判定 1：CLS 是否在 last_hidden_state 里 —— 只看 token 数（pooler 对比不可靠）
    n_tok = out.last_hidden_state.shape[1]
    if n_tok == expected_patches:
        ok(f"last_hidden_state 有 {n_tok} 个 token = patch 数 → 不含 CLS，无需切片")
        cls_present = False
    elif n_tok == expected_patches + 1:
        ok(f"last_hidden_state 有 {n_tok} 个 token = patch 数 + 1 → 含 CLS，需切片")
        cls_present = True
    else:
        fail(f"token 数异常: {n_tok}，既不等于 {expected_patches} 也不等于 {expected_patches + 1}")
        cls_present = None

    # ------------------------------------------------------------------
    sec("B. VisionEncoder 封装：输出 token 数与冻结（防 off-by-one 回归）")
    # ------------------------------------------------------------------
    try:
        from models.vision_encoder import VisionEncoder

        enc = VisionEncoder(str(CLIP_DIR))                      # 默认 drop_cls=True
        with torch.no_grad():
            feats = enc(px)
        print(f"  drop_cls=True  -> {tuple(feats.shape)}  (num_tokens={enc.num_tokens})")
        if feats.shape[1] == expected_patches:
            ok(f"默认配置输出 {feats.shape[1]} 个 token（预期 {expected_patches}）")
        else:
            fail(f"默认配置输出 {feats.shape[1]} 个 token，预期 {expected_patches}")

        enc2 = VisionEncoder(str(CLIP_DIR), drop_cls=False)
        with torch.no_grad():
            feats2 = enc2(px)
        print(f"  drop_cls=False -> {tuple(feats2.shape)}  (num_tokens={enc2.num_tokens})")
        if feats2.shape[1] == n_tok:
            ok(f"drop_cls=False 输出 {feats2.shape[1]} 个 token = 底层 last_hidden_state 原样")
        else:
            fail(f"drop_cls=False 输出 {feats2.shape[1]}，与底层 {n_tok} 不一致")

        # 关键一致性：drop_cls=True 时不能把第一个 patch 删掉
        if cls_present is True:
            with torch.no_grad():
                same = torch.allclose(feats, out.last_hidden_state[:, 1:, :], atol=1e-5)
            if same:
                ok("drop_cls=True 的结果 == last_hidden_state[:,1:,:]（丢的是 CLS，不是 patch）")
            else:
                fail("drop_cls=True 的结果与 last_hidden_state[:,1:,:] 不一致")
        elif cls_present is False:
            with torch.no_grad():
                same = torch.allclose(feats, out.last_hidden_state, atol=1e-5)
            if same:
                ok("底层本就不含 CLS，drop_cls=True 未误删任何 token")
            else:
                fail("底层不含 CLS，但 drop_cls=True 改变了输出 → 正在误删第一个 patch！")

        n_train = sum(p.requires_grad for p in enc.parameters())
        if n_train == 0:
            ok("冻结生效：可训练参数为 0")
        else:
            fail(f"冻结失效：有 {n_train} 个参数需要梯度")
    except Exception as exc:  # noqa: BLE001
        fail(f"VisionEncoder 调用失败: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    sec("C. Qwen2.5 tokenizer：占位符 token 与词表")
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    print(f"  加载自 {QWEN_DIR}")
    print(f"  len(tokenizer) = {len(tok)}   vocab_size = {tok.vocab_size}")

    placeholder = "<|image_pad|>"
    pid = tok.convert_tokens_to_ids(placeholder)
    print(f"  convert_tokens_to_ids('{placeholder}') = {pid}")
    if isinstance(pid, int):
        ok(f"'{placeholder}' 已在词表中，id={pid} → 无需 add_special_tokens")
    else:
        fail(f"'{placeholder}' 不在词表中（返回 {pid}），需要自行注册占位符")

    n_tok_ph = len(tok(placeholder, add_special_tokens=False)["input_ids"])
    print(f"  tokenize('{placeholder}') 得到 {n_tok_ph} 个 id")
    if n_tok_ph == 1:
        ok("占位符是单一 token（可安全用于定位 visual token 位置）")
    else:
        fail(f"占位符被切成 {n_tok_ph} 个 token —— 无法用于定位")

    msgs = [{"role": "user", "content": f"{placeholder}\n这张图片里有什么？"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids_list = tok(text, add_special_tokens=False)["input_ids"]
    n_ph = sum(1 for i in ids_list if i == pid)
    print(f"  chat template 文本长度 = {len(text)} 字符")
    print(f"  编码后长度 = {len(ids_list)}，占位符出现 {n_ph} 次")
    if n_ph == 1:
        ok("chat template 往返后占位符保留且唯一")
    else:
        fail(f"占位符在编码后出现 {n_ph} 次，期望 1 次")
    print(f"  bos_token_id = {tok.bos_token_id}   (Qwen2 无 BOS，拼接时必须自己控制)")

    # ------------------------------------------------------------------
    sec("D. Qwen 权重：词表大小一致性（决定要不要 resize）")
    # ------------------------------------------------------------------
    try:
        from safetensors import safe_open
        sf = QWEN_DIR / "model.safetensors"
        with safe_open(str(sf), framework="pt") as f:
            keys = [k for k in f.keys() if "embed_tokens" in k or "lm_head" in k]
            print(f"  embedding 相关 key: {keys}")
            rows = None
            for k in keys:
                shape = tuple(f.get_slice(k).get_shape())
                print(f"    {k}: {shape}")
                if rows is None:
                    rows = shape[0]
        if rows is not None:
            print(f"  权重行数 = {rows}   len(tokenizer) = {len(tok)}   富余 = {rows - len(tok)} 行")
            if isinstance(pid, int) and pid < rows:
                ok(f"占位符 id {pid} 在权重范围内 → 不需要 resize_token_embeddings")
            else:
                fail(f"占位符 id {pid} 超出权重行数 {rows} → 必须 resize")
            if rows < len(tok):
                warn(f"权重比 tokenizer 少 {len(tok) - rows} 行：这些 id 无法使用，勿引入新 token")
    except Exception as exc:  # noqa: BLE001
        warn(f"无法读取 safetensors: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    if FAIL:
        print(f"  结论：{len(FAIL)} 项失败，{len(WARN)} 项警告")
        for m in FAIL:
            print(f"    - {m}")
        return 1
    print(f"  结论：离线验收通过（{len(WARN)} 项警告）")
    for m in WARN:
        print(f"    ! {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
