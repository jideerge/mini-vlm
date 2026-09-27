#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/inspect_sample.py —— 打印一条完整构造样本，肉眼确认无误（Week 1 验收项）

它串起整条链路，并逐段打印证据：

    合成图片(白底 + 2 个黑方块 + 1 个黑圆)
        -> CLIPImageProcessor           -> pixel_values [1,3,224,224]
        -> VisionEncoder (frozen)       -> [1,49,768]
        -> Projector                    -> [1,49,896]
        -> MultipleImageCollator        -> input_ids / labels
        -> inject_visual_embeddings     -> inputs_embeds（占位符已被替换）
        -> Qwen2.5-0.5B 前向             -> logits

四条断言（不通过就退出码 1）：
  A. 占位符数量 == 视觉 token 数 == 49
  B. loss 只在 answer 区间开启（answer 之前的 label 全为 -100）
  C. 占位符嵌入 0.0 != 注入后该位置的值，且非占位符位置一个字节都没动
  D. 只改图片，答案位置的 logits 必须变化（证明图片真的进了模型）

运行：
    & $env:MINIVLM_PY scripts/inspect_sample.py
可选：
    --image 路径   用真实图片替换合成图片
    --device cpu|cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from datasets.collator import (  # noqa: E402
    IMAGE_PLACEHOLDER,
    IGNORE_INDEX,
    MultipleImageCollator,
    inject_visual_embeddings,
)
from models.vision_encoder import VisionEncoder  # noqa: E402

PRETRAINED = Path(os.environ.get("MINIVLM_PRETRAINED", ROOT / "checkpoints" / "pretrained"))
CLIP_DIR = PRETRAINED / "clip-vit-base-patch32"
QWEN_DIR = PRETRAINED / "qwen2.5-0.5b-instruct"

QUESTION = f"{IMAGE_PLACEHOLDER}\n图中一共有几个黑色方块？"
ANSWER = "图中有 2 个黑色方块。"

FAIL: list[str] = []


def sec(t: str) -> None:
    print("\n" + "=" * 66)
    print(f"  {t}")
    print("=" * 66)


def check(cond: bool, msg: str, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(msg)


def make_synthetic_image(kind: str = "squares2") -> Image.Image:
    """白底 + 图形。两种 kind 用来验证'换图答案必须变'。"""
    img = Image.new("RGB", (336, 336), "white")
    d = ImageDraw.Draw(img)
    if kind == "squares2":
        d.rectangle([40, 60, 120, 140], fill="black")
        d.rectangle([200, 60, 280, 140], fill="black")
        d.ellipse([120, 200, 220, 300], fill="black")
    else:  # squares3
        d.rectangle([30, 50, 100, 120], fill="black")
        d.rectangle([140, 50, 210, 120], fill="black")
        d.rectangle([250, 50, 320, 120], fill="black")
        d.ellipse([120, 210, 220, 300], fill="black")
    return img


def show_piece(ids: list[int], labels: list[int], tok, limit: int = 64) -> None:
    """逐 token 打印，重点看前 64 个位置（视觉 token + 提问 + 答案开头）。"""
    print(f"  {'pos':>4} | {'token_id':>8} | {'label':>8} | token 文本")
    print("  " + "-" * 58)
    for i in range(min(len(ids), limit)):
        tid = ids[i]
        lab = labels[i]
        piece = tok.decode([tid])
        piece = piece.replace("\n", "\\n")
        if len(piece) > 20:
            piece = piece[:20] + "…"
        lab_s = "IGNORE" if lab == IGNORE_INDEX else str(lab)
        mark = ""
        if tid == tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER):
            mark = "   <-- 视觉占位符"
        elif lab != IGNORE_INDEX:
            mark = "   <-- 计算 loss"
        print(f"  {i:>4} | {tid:>8} | {lab_s:>8} | {piece!r}{mark}")
    if len(ids) > limit:
        print(f"  ... 其余 {len(ids) - limit} 个位置省略")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None, help="用真实图片替换合成图片")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-length", type=int, default=768)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPProcessor

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    sec("1. 加载三个组件")
    # ------------------------------------------------------------------
    processor = CLIPProcessor.from_pretrained(str(CLIP_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    vision = VisionEncoder(str(CLIP_DIR)).to(device)

    from models.projector import Projector  # 延迟导入，便于先看报错

    projector = Projector(
        in_dim=vision.hidden_size, out_dim=896, hidden=2048, depth=2
    ).to(device)
    n_proj = sum(p.numel() for p in projector.parameters())
    print(f"  CLIPProcessor   : {type(processor).__name__}")
    print(f"  Tokenizer       : {type(tokenizer).__name__}  len={len(tokenizer)}")
    print(f"  VisionEncoder   : hidden={vision.hidden_size}  num_tokens={vision.num_tokens}")
    print(f"  Projector       : 768 -> 2048 -> 896 (GELU)  参数量={n_proj:,}")

    collator = MultipleImageCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=args.max_length,
        # 必须等于视觉塔输出 token 数（CLIP ViT-B/32 @224 -> 49）
        num_image_token=vision.num_tokens,
    )
    print(f"  占位符 token id : {collator.image_token_id}  ({IMAGE_PLACEHOLDER})")
    print(f"  每图展开 token 数: {collator.num_image_token}")

    # ------------------------------------------------------------------
    sec("2. 构造样本（文本侧）")
    # ------------------------------------------------------------------
    image_a = Image.open(args.image).convert("RGB") if args.image else make_synthetic_image("squares2")
    image_b = make_synthetic_image("squares3") if not args.image else image_a

    print("  question :", repr(QUESTION))
    print("  answer   :", repr(ANSWER))

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
    )
    print("\n  --- chat template 渲染结果 ---")
    print("  " + prompt_text.replace("\n", "\n  "))

    batch = collator([{"image": image_a, "question": QUESTION, "answer": ANSWER}])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    print(f"\n  input_ids.shape  = {tuple(batch['input_ids'].shape)}")
    print(f"  labels.shape     = {tuple(batch['labels'].shape)}")
    print(f"  占位符数量        = {batch['placeholder_counts'][0]}")
    print(f"  answer 起始位置   = {batch['answer_starts'][0]}")
    print(f"  pixel_values     = {tuple(batch['pixel_values'].shape)}")

    # ------------------------------------------------------------------
    sec("3. 逐 token 对照（前 64 个位置）")
    # ------------------------------------------------------------------
    show_piece(ids, labels, tokenizer)

    # 断言 A
    n_ph = batch["placeholder_counts"][0]
    check(
        n_ph == vision.num_tokens,
        "A. 占位符数量 == 视觉 token 数",
        f"{n_ph} == {vision.num_tokens}",
    )

    # 断言 B
    ans_start = batch["answer_starts"][0]
    before = labels[:ans_start]
    after = labels[ans_start:]
    n_supervised = sum(1 for x in after if x != IGNORE_INDEX)
    check(
        all(x == IGNORE_INDEX for x in before) and n_supervised > 0,
        "B. loss 只在 answer 区间开启",
        f"answer 之前 {len(before)} 个全为 -100；answer 区间 {n_supervised} 个参与 loss",
    )
    decoded_answer = tokenizer.decode([t for t in after if t != IGNORE_INDEX])
    print(f"  参与 loss 的文本 : {decoded_answer!r}")

    # ------------------------------------------------------------------
    sec("4. 视觉侧：CLIP -> Projector -> 注入 LLM 嵌入")
    # ------------------------------------------------------------------
    with torch.no_grad():
        feats = vision(batch["pixel_values"].to(device))
        visual = projector(feats)
    print(f"  CLIP 输出    : {tuple(feats.shape)}")
    print(f"  Projector 后 : {tuple(visual.shape)}   dtype={visual.dtype}")

    model = AutoModelForCausalLM.from_pretrained(str(QWEN_DIR), dtype=torch.float32).to(device)
    model.eval()
    emb_layer = model.get_input_embeddings()
    print(f"  LLM embedding: {tuple(emb_layer.weight.shape)}")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = emb_layer(input_ids)
        # 占位符嵌入清零：该位置反正会被覆盖，清零可避免随机嵌入参与 loss
        ph_mask = (input_ids == collator.image_token_id).unsqueeze(-1)
        text_embeds = text_embeds.masked_fill(ph_mask, 0.0)
        inputs_embeds = inject_visual_embeddings(
            text_embeds, input_ids, visual, collator.image_token_id
        )

    ph_pos = (input_ids[0] == collator.image_token_id).nonzero().flatten()
    injected_norm = inputs_embeds[0, ph_pos[0]].norm().item()
    print(f"  占位符位置示例 : pos={ph_pos[0].item()}  text_embeds 处范数=0.0000  "
          f"注入后范数={injected_norm:.4f}")

    # 断言 C
    changed_at_ph = injected_norm > 0
    other = torch.ones_like(input_ids[0], dtype=torch.bool)
    other[ph_pos] = False
    untouched = torch.equal(
        inputs_embeds[0][other], text_embeds[0][other]
    )
    check(
        changed_at_ph and untouched,
        "C. 只有占位符位置被替换",
        f"占位符已注入={changed_at_ph}，非占位符位置未被改动={untouched}",
    )

    # ------------------------------------------------------------------
    sec("5. 前向：确认图片真的影响了输出")
    # ------------------------------------------------------------------
    def forward_logits(img: Image.Image) -> torch.Tensor:
        b = collator([{"image": img, "question": QUESTION, "answer": ANSWER}])
        ii = b["input_ids"].to(device)
        with torch.no_grad():
            f = vision(b["pixel_values"].to(device))
            v = projector(f)
            te = emb_layer(ii)
            te = te.masked_fill((ii == collator.image_token_id).unsqueeze(-1), 0.0)
            ie = inject_visual_embeddings(te, ii, v, collator.image_token_id)
            return model(
                inputs_embeds=ie,
                attention_mask=b["attention_mask"].to(device),
            ).logits

    logits_a = forward_logits(image_a)
    logits_b = forward_logits(image_b)
    print(f"  logits.shape = {tuple(logits_a.shape)}")

    # 只比较"答案位置"的 logits（那里才是模型真正要生成的内容）
    ans_slice = slice(max(ans_start - 1, 0), logits_a.size(1))
    diff = (logits_a[0, ans_slice] - logits_b[0, ans_slice]).abs().max().item()
    print(f"  换图后答案位置 logits 最大差值 = {diff:.6f}")
    check(diff > 1e-3, "D. 换图导致答案位置 logits 变化（图片真的进了模型）", f"max|Δ|={diff:.6f}")

    # 顺便看一眼 greedy 生成结果，纯属观感参考（未训练，不该好看）
    with torch.no_grad():
        gen_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("\n  greedy 生成（未训练，仅作管线连通性参考）：")
    print("   ", repr(tokenizer.decode(gen_ids[0], skip_special_tokens=False)))

    # ------------------------------------------------------------------
    sec("结论")
    if FAIL:
        print(f"  {len(FAIL)} 项失败：")
        for m in FAIL:
            print(f"    - {m}")
        return 1
    print("  4 项断言全部通过 —— 一条完整样本构造无误")
    print("  （注意：此时模型未训练，生成结果没有意义；这里只验证数据通路）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
