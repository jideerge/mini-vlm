#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/collator.py —— 多模态样本构造与批处理（全项目的心脏）

本模块负责把 (图片, 问题, 答案) 变成 LLM 真正吃进去的三样东西：

    input_ids      [L]        文本 token（含 N 个 <|image_pad|> 占位符）
    inputs_embeds  [L, 896]   占位符位置已被 Projector 输出替换
    labels         [L]        answer 区间为 token id，其余全部 -100

设计要点（都是踩过的坑）：
  1. **labels 用两次 tokenize 定位答案起点**，不做字符串匹配 —— 精确且不依赖空白
  2. **占位符嵌入显式清零** —— 该位置反正要被 Projector 输出覆盖；
     清零可以避免"随机初始化的占位符嵌入参与 loss"这种隐蔽污染
  3. **不在 tokenizer 里加特殊 token** —— <|image_pad|> 已存在于 Qwen2.5 词表（id 151655）
  4. **永远用 inputs_embeds 前向** —— 不走 model.generate() 的 input_ids 路径，
     避免 decoder-only 模型"训练正常但推理时忽略图片"的经典 bug

本地实测依据见 scripts/verify_offline.py 与 scripts/inspect_sample.py。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

# Qwen2.5 词表中已有该 token（id = 151655），不要自行注册
IMAGE_PLACEHOLDER = "<|image_pad|>"
IGNORE_INDEX = -100


def _as_bool_row(mask: Any, length: int) -> list[bool]:
    """把 attention_mask / 各种 mask 统一成 python bool 列表。"""
    if mask is None:
        return [True] * length
    if isinstance(mask, torch.Tensor):
        mask = mask.tolist()
    return [bool(x) for x in mask]


def pad_to_square(
    img: Any,
    fill: tuple[int, int, int] = (255, 255, 255),
    min_side: int = 224,
) -> Any:
    """把图片居中贴到正方形画布上。

    ⚠️ Week 2 实测结论（笔记 §4）：CLIPProcessor 做的是「最短边缩放 + 中心裁剪」，
    不是拉伸。VOC 的 4:3 图左右各有约 12.5% 会被裁掉，
    靠边的目标会直接消失 —— 对计数与空间关系任务是致命的
    （标注里有、画面里没有）。

    实测验证：500x250 的纯红图，不 pad 时输出里白边剩 0/224 行（全被裁掉）；
    pad 成正方形后白边剩 112/224 行，内容零丢失且长宽比不变。

    min_side=224 保证小图也会被放大到至少 224，避免 processor 再放大产生模糊。
    """
    from PIL import Image as _Image

    w, h = img.size
    s = max(w, h, min_side)
    if (w, h) == (s, s):
        return img
    canvas = _Image.new("RGB", (s, s), fill)
    canvas.paste(img.convert("RGB"), ((s - w) // 2, (s - h) // 2))
    return canvas


@dataclass
class MultipleImageCollator:
    """DataLoader 的 collate_fn。

    processor      : CLIPProcessor（图片 -> pixel_values）
    tokenizer      : Qwen tokenizer
    max_length     : 文本侧截断长度（视觉 token 另占 49 个位置）
    num_image_token: 每张图展开成多少个占位符，必须等于视觉塔的输出 token 数
                     CLIP ViT-B/32 @224 -> 49
    pad_inputs     : 是否把图片先 pad 成正方形（强烈建议保持 True，理由见 pad_to_square）
    expand_placeholders : 是否把 1 个 <|image_pad|> 展开成 num_image_token 个

    关于 expand_placeholders（关键设计，别关掉）：
        提问文本里只写 **1 个** <|image_pad|>，但视觉侧产出 **49 个** token，
        两者对不上就无法逐位置赋值。所以标准做法是：把占位符展开成 49 个连续位置，
        再把这 49 个位置的嵌入整段替换成 Projector 输出（LLaVA 等实现同理）。
        展开只发生在占位符处，回答区间整体后移，loss 掩码自动保持正确。
    """

    processor: Any
    tokenizer: Any
    max_length: int = 768
    num_image_token: int = 49
    expand_placeholders: bool = True
    pad_to_square_images: bool = True
    pad_fill: tuple[int, int, int] = (255, 255, 255)
    pad_to_multiple_of: int | None = 8

    image_token_id: int = field(init=False)
    pad_token_id: int = field(init=False)

    def __post_init__(self) -> None:
        self.image_token_id = int(
            self.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        )
        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError(f"tokenizer 里找不到 {IMAGE_PLACEHOLDER}")
        # Qwen2.5 的 pad_token 为 None（无 BOS/PAD），用 EOS 充当 pad
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("tokenizer 既没有 pad_token_id 也没有 eos_token_id")
        self.pad_token_id = int(pad)
        if self.num_image_token <= 0:
            raise ValueError(f"num_image_token 必须为正，收到 {self.num_image_token}")

    # ------------------------------------------------------------------
    # 文本 + 标签
    # ------------------------------------------------------------------
    def build_text_and_labels(
        self, question: str, answer: str, system: str | None = None
    ) -> tuple[list[int], list[int], int]:
        """返回 (input_ids, labels, answer_start)。

        - question 里必须已经含有 IMAGE_PLACEHOLDER（数量 = 视觉 token 数）
        - 只有 answer 区间参与 loss；assistant 头与提问部分全部 -100
        """
        tok = self.tokenizer
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": question}
        ]
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tok(answer, add_special_tokens=False)["input_ids"]

        input_ids = list(prompt_ids) + list(answer_ids)
        # answer 与前缀之间补一个 EOS，让模型学会"说完就停"
        if tok.eos_token_id is not None:
            input_ids.append(int(tok.eos_token_id))

        answer_start = len(prompt_ids)
        labels = [IGNORE_INDEX] * answer_start + list(input_ids[answer_start:])

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        return input_ids, labels, answer_start

    # ------------------------------------------------------------------
    # 占位符展开
    # ------------------------------------------------------------------
    def expand(self, ids: list[int], labels: list[int], answer_start: int) -> tuple[list[int], list[int], int]:
        """把每个 <|image_pad|> 展开成 num_image_token 个连续占位符。

        展开后：
          * 占位符个数 = 图片数 x num_image_token，可与视觉 token 逐位置对齐
          * 回答区间整体后移，loss 掩码仍然只覆盖 answer
        """
        if not self.expand_placeholders:
            return ids, labels, answer_start

        new_ids: list[int] = []
        new_labels: list[int] = []
        shift = 0
        for tid, lab in zip(ids, labels):
            if tid == self.image_token_id:
                new_ids.extend([tid] * self.num_image_token)
                new_labels.extend([IGNORE_INDEX] * self.num_image_token)
                shift += self.num_image_token - 1
            else:
                new_ids.append(tid)
                new_labels.append(lab)
        return new_ids, new_labels, answer_start + shift

    # ------------------------------------------------------------------
    # 单样本
    # ------------------------------------------------------------------
    def __call__(self, batch: Sequence[dict]) -> dict[str, Any]:
        texts, images, answer_starts = [], [], []

        for ex in batch:
            ids, labels, ans_start = self.build_text_and_labels(
                ex["question"], ex["answer"], ex.get("system")
            )
            ids, labels, ans_start = self.expand(ids, labels, ans_start)
            texts.append({"input_ids": ids, "labels": labels})
            images.append(ex["image"])
            answer_starts.append(ans_start)

        # ⚠️ 不能把 "labels" 交给 tokenizer.pad()：
        #    Qwen2Tokenizer.pad 只处理 input_ids / attention_mask / token_type_ids，
        #    自定义字段会被**原样返回不做补齐**，于是 labels 与 attention_mask 长度不一致，
        #    报 "The size of tensor a (56) must match the size of tensor b (51)"。
        #    所以这里显式用 pad_token_id / 0.0 手动补齐，labels 用 IGNORE_INDEX 填充。
        max_len = max(len(t["input_ids"]) for t in texts)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, attention_mask, labels = [], [], []
        for t in texts:
            ids, labs = t["input_ids"], t["labels"]
            n_pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * n_pad)
            attention_mask.append([1] * len(ids) + [0] * n_pad)
            labels.append(labs + [IGNORE_INDEX] * n_pad)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        # pad 位置统一为 IGNORE_INDEX（真实 token id 出现在 pad 槽里会导致 loss 算错）
        labels_t = torch.tensor(labels, dtype=torch.long).masked_fill(
            attention_mask_t == 0, IGNORE_INDEX
        )

        # 图片预处理（必须用 CLIP 的 processor，不是 Qwen2.5-VL 的）
        # 先 pad 成正方形再交给 processor，避免中心裁剪丢内容（Week 2 实测）
        if self.pad_to_square_images:
            images = [pad_to_square(im, fill=self.pad_fill) for im in images]
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]

        # 占位符个数校验：每个样本必须恰好 N 个
        placeholder_counts = (input_ids_t == self.image_token_id).sum(dim=1).tolist()

        # 快速失败：在 collator 里就拦住"问题忘了带占位符"这类错误，
        # 不要等到模型前向才报 —— 那时报错信息离根因太远，很难查。
        bad = [i for i, c in enumerate(placeholder_counts)
               if c != self.num_image_token]
        if bad:
            i0 = bad[0]
            q0 = batch[i0].get("question", "")
            raise ValueError(
                f"样本 {i0} 展开后占位符 {placeholder_counts[i0]} 个，"
                f"期望 {self.num_image_token} 个。\n"
                f"  question 前 80 字符: {q0[:80]!r}\n"
                f"  常见原因：传进来的是**原始问题**，没有先插入 <|image_pad|>。\n"
                f"  正确做法：用 JsonlVLDataset.build_question(q) 包一层再交给 collator。"
            )

        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "labels": labels_t,
            "pixel_values": pixel_values,
            "placeholder_counts": placeholder_counts,
            "answer_starts": answer_starts,
            "image_token_id": self.image_token_id,
        }


def inject_visual_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """把 Projector 输出的视觉嵌入写进占位符位置。

    inputs_embeds : [B, L, Dl]
    input_ids     : [B, L]
    visual_embeds : [B, N, Dl]   （N = 每张图的视觉 token 数）
    """
    out = inputs_embeds.clone()
    bsz = input_ids.size(0)
    for b in range(bsz):
        pos = torch.nonzero(input_ids[b] == image_token_id, as_tuple=False).flatten()
        if pos.numel() != visual_embeds.size(1):
            raise ValueError(
                f"样本 {b}: 占位符 {pos.numel()} 个，视觉 token {visual_embeds.size(1)} 个，不匹配"
            )
        out[b, pos] = visual_embeds[b].to(out.dtype)
    return out


# ----------------------------------------------------------------------
# ⚠️ 命名空间冲突（实测确认，不是理论问题）
# 本目录名为 datasets/，在仓库根运行时会**遮蔽** HuggingFace 的 datasets 包：
#     import datasets  ->  D:\...\9.VLM\datasets\__init__.py
# 后果：
#   * `import datasets` / `from datasets import load_dataset` 拿到的是本目录，不是 HF 库
#   * 将来 pip install datasets 之后仍然如此（当前目录优先级最高）
# 取舍：本项目数据是自建 JSONL，本来就不依赖 HF datasets 库；
#       若某天确实需要它，改成 `pip install datasets` 后用绝对导入或重命名本目录。
# ----------------------------------------------------------------------
