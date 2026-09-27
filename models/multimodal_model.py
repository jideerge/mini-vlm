#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/multimodal_model.py —— Mini-VLM 组装（全项目的心脏）

把三个部分接起来：

    VisionEncoder(冻结)  ->  visual features [B, 49, 768]
    Projector(可训练)    ->  visual embeddings [B, 49, 896]
    Qwen2.5(冻结/LoRA)   ->  Answer

关键实现点（每一条都是 Week 1-2 实测踩出来的）：
  1. **必须用 inputs_embeds 前向**：视觉嵌入没有对应的 token id
  2. **占位符位置显式清零**再注入：该位置反正会被覆盖，清零可避免随机嵌入污染计算图
  3. **loss 只在 answer 区间**：labels 已是 -100 掩码；用 logits[:, :-1] 与
     labels[:, 1:] 对齐（自回归标准 shift）
  4. **推理走手写自回归循环**：decoder-only 的 generate() 与 inputs_embeds 组合有坑
     （可能忽略图片），手写循环完全可控
  5. **支持预计算视觉特征**：视觉塔冻结 => 特征恒定 => 训练时可完全不加载 CLIP
     （省显存 + 提速，见 notes/week2 实验 E）

用法：
    from models.multimodal_model import MiniVLM, MiniVLMConfig
    m = MiniVLM(MiniVLMConfig(...))
    out = m(input_ids=..., attention_mask=..., labels=..., visual_features=cached)
    ids = m.generate(input_ids=..., attention_mask=..., visual_features=cached)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging as hf_logging

from models.projector import Projector, count_parameters
from models.vision_encoder import VisionEncoder

hf_logging.set_verbosity_error()

IGNORE_INDEX = -100

# 调试开关：设为 True 时 generate() 会逐步打印窗口信息（定位解码问题用）
_DEBUG_VERBOSE = False


@dataclass
class MiniVLMConfig:
    clip_path: str = "checkpoints/pretrained/clip-vit-base-patch32"
    llm_path: str = "checkpoints/pretrained/qwen2.5-0.5b-instruct"

    # 视觉塔
    freeze_vision: bool = True
    drop_cls: bool = True
    load_vision: bool = True          # False = 只用预计算特征，完全不加载 CLIP

    # Projector
    projector_depth: int = 2          # 0=Linear, 2=双层MLP, 3=三层MLP
    projector_hidden: int = 2048
    projector_dropout: float = 0.0

    # LLM
    freeze_llm: bool = True
    dtype: str = "float32"            # 只训 Projector 时 float32 最稳

    # 训练策略：projector_only | lora | qlora
    #   projector_only : 只训 Projector（Experiment A）
    #   lora           : Projector + LLM 的 LoRA 适配器，bf16（Experiment B）
    #   qlora          : 同 lora，但基础模型 4-bit 量化（Experiment C，需 bitsandbytes）
    strategy: str = "projector_only"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # 数据
    num_image_token: int = 49         # 必须 = 视觉塔输出 token 数
    max_seq_len: int = 768

    device: str = "cpu"


class MiniVLM(nn.Module):
    """视觉编码器 + Projector + LLM。"""

    def __init__(
        self,
        config: MiniVLMConfig,
        vision: VisionEncoder | None = None,
        llm: nn.Module | None = None,
        processor: Any = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.device_ = torch.device(config.device)
        dtype = getattr(torch, config.dtype)

        # ---------- 视觉塔 ----------
        if vision is not None:
            self.vision: VisionEncoder | None = vision
        elif config.load_vision:
            self.vision = VisionEncoder(
                config.clip_path, freeze=config.freeze_vision, drop_cls=config.drop_cls
            )
        else:
            self.vision = None

        if self.vision is not None:
            self.vision = self.vision.to(self.device_, dtype=dtype)
            self.vision.eval()
            self.vision_size = self.vision.hidden_size
            self.num_image_token = self.vision.num_tokens
        else:
            # 不加载视觉塔时必须由配置给出尺寸
            self.vision_size = 768
            self.num_image_token = config.num_image_token

        # ---------- 语言模型 ----------
        strategy = getattr(config, "strategy", "projector_only")
        self.strategy = strategy
        self.llm_frozen_base = True

        if llm is not None:
            self.llm = llm
        elif strategy == "qlora":
            # 4-bit 量化加载（需要 bitsandbytes；失败时给出明确提示）
            from models.lora import LoRAConfig, prepare_qlora

            self.llm = prepare_qlora(config.llm_path, LoRAConfig(use_qlora=True))
        else:
            from transformers import AutoModelForCausalLM

            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_path, dtype=dtype
            )
        if not (strategy == "qlora" and llm is None):
            self.llm = self.llm.to(self.device_)
        self.llm_size = int(self.llm.config.hidden_size)

        # ---------- LoRA 注入（必须在 freeze 之前：peft 会自己管理冻结） ----------
        self.lora_info: dict | None = None
        if strategy in ("lora", "qlora"):
            from models.lora import LoRAConfig, apply_lora, lora_report

            lcfg = LoRAConfig(
                r=config.lora_r, alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=tuple(config.lora_target_modules),
                use_qlora=(strategy == "qlora"),
            )
            self.llm = apply_lora(self.llm, lcfg)
            self.lora_info = lora_report(self.llm)
            # 注入后基础权重已被冻结，只留 lora_A / lora_B 可训练；
            # 不能再调 requires_grad_(False)，那会把 LoRA 也一起冻掉。
        elif config.freeze_llm:
            self.llm.requires_grad_(False)

        # ---------- 连接器（唯一可训练模块） ----------
        self.projector = Projector(
            in_dim=self.vision_size,
            out_dim=self.llm_size,
            hidden=config.projector_hidden,
            depth=config.projector_depth,
            dropout=config.projector_dropout,
        ).to(self.device_, dtype=dtype)

        # ---------- tokenizer / processor ----------
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError("tokenizer 里找不到 <|image_pad|>")

        if processor is not None:
            self.processor = processor
        else:
            try:
                from transformers import CLIPProcessor

                self.processor = CLIPProcessor.from_pretrained(config.clip_path)
            except Exception:  # noqa: BLE001
                self.processor = None

        # ---------- 一致性自检：把"写错维度"变成"启动即失败" ----------
        self.assert_consistent()

    # ------------------------------------------------------------------
    def assert_consistent(self) -> None:
        assert self.projector.in_dim == self.vision_size, (
            f"Projector.in_dim={self.projector.in_dim} != vision.hidden_size={self.vision_size}"
        )
        assert self.projector.out_dim == self.llm_size, (
            f"Projector.out_dim={self.projector.out_dim} != llm.hidden_size={self.llm_size}"
        )
        rows = int(self.llm.get_input_embeddings().weight.shape[0])
        assert self.image_token_id < rows, (
            f"占位符 id {self.image_token_id} 超出 embedding 行数 {rows}"
        )
        assert self.num_image_token == self.cfg.num_image_token, (
            f"num_image_token={self.num_image_token} 与配置 {self.cfg.num_image_token} 不一致"
        )
        # LoRA 策略下：可训练参数必须只来自 projector 与 lora_*
        # （防止"以为只训 LoRA，实际把整个 LLM 也训了"这类事故）
        if self.strategy in ("lora", "qlora"):
            from models.lora import assert_only_projector_and_lora

            bad = assert_only_projector_and_lora(self)
            assert not bad, (
                f"LoRA 策略下出现了意外的可训练参数（前 5 个）：{bad[:5]}\n"
                f"  只允许 projector.* 与含 lora_ 的参数"
            )

    # ------------------------------------------------------------------
    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> str:
        pv = count_parameters(self.vision) if self.vision is not None else 0
        pv_train = count_parameters(self.vision, True) if self.vision is not None else 0
        pl = count_parameters(self.llm)
        pl_train = count_parameters(self.llm, True)
        pj = count_parameters(self.projector)
        base = (
            f"vision : {pv:>12,}  trainable={pv_train:,}\n"
            f"llm    : {pl:>12,}  trainable={pl_train:,}\n"
            f"proj   : {pj:>12,}  trainable={count_parameters(self.projector, True):,}\n"
            f"total  : {self.total_parameters:>12,}  "
            f"trainable={self.trainable_parameters:,} "
            f"({self.trainable_parameters / max(self.total_parameters, 1):.2%})"
        )
        if self.lora_info:
            base += (f"\n  ├─ 其中 LoRA 适配器：{self.lora_info['num_lora_params']:,} 参数 / "
                     f"{self.lora_info['num_lora_tensors']} 个张量"
                     f"（{self.lora_info['by_module']}）")
        base += f"\n  └─ 策略：{self.strategy}"
        return base

    # ------------------------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """图片 -> 视觉嵌入 [B, N, llm_hidden]。

        ⚠️ 这里**不能**用 `torch.no_grad()`：
        视觉塔的冻结是靠 `requires_grad_(False)` 实现的（参数不接收梯度），
        而不是靠切断计算图。如果用 no_grad 包住整段，
        Projector 的输出会变成不需要梯度的张量，`loss.backward()` 直接报
        "element 0 of tensors does not require grad and does not have a grad_fn"。
        （这正是本脚本第一版踩到的坑。）
        """
        assert self.vision is not None, "未加载视觉塔；请改用预计算特征"
        pv = pixel_values.to(self.device_, dtype=next(self.vision.parameters()).dtype)
        feats = self.vision(pv)                       # 视觉塔无梯度参数，但有计算图
        proj_dtype = next(self.projector.parameters()).dtype
        return self.projector(feats.to(proj_dtype))

    # ------------------------------------------------------------------
    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        zero_placeholder: bool = True,
    ) -> torch.Tensor:
        """拼装 inputs_embeds。

        input_ids       : [B, L]  含 N 个占位符（N = num_image_token）
        visual_features : [B, N, llm_hidden]（已过 Projector）
        """
        emb_layer = self.llm.get_input_embeddings()
        llm_dtype = emb_layer.weight.dtype      # 子模块 dtype 可能不同，统一到 LLM

        ids = input_ids.to(self.device_)
        # 文本嵌入是常量，不需要梯度（图像占位符位置稍后会被完全替换）
        with torch.no_grad():
            text_embeds = emb_layer(ids).detach().clone().to(llm_dtype)

        ph_mask = (ids == self.image_token_id)
        if zero_placeholder:
            # 该位置反正要被覆盖；清零可避免"随机初始化的占位符嵌入"参与计算
            text_embeds = text_embeds.masked_fill(ph_mask.unsqueeze(-1), 0.0)

        vis = visual_features.to(self.device_, dtype=llm_dtype)
        # 用 torch.cat 组装，保证视觉分支的计算图**不被切断**
        # （不能用 out = text_embeds; out[b] = vis[b]——那样得到的是纯推理张量）
        chunks: list[torch.Tensor] = []
        for b in range(ids.size(0)):
            pos = torch.nonzero(ph_mask[b], as_tuple=False).flatten()
            if pos.numel() != vis.size(1):
                raise ValueError(
                    f"样本 {b}: 占位符 {pos.numel()} 个，视觉 token {vis.size(1)} 个，不匹配"
                )
            mask_row = torch.zeros(ids.size(1), dtype=torch.bool, device=self.device_)
            mask_row[pos] = True
            left = text_embeds[b, ~mask_row]
            row = torch.cat([left[: int(pos[0].item())], vis[b],
                             left[int(pos[0].item()):]], dim=0)
            chunks.append(row.unsqueeze(0))
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        precomputed_features: bool = False,
        answer_only_loss: bool = True,
        return_logits: bool = False,
    ) -> dict[str, Any]:
        """前向。三个视觉入口三选一：

        * `pixel_values`                       现算：视觉塔 + Projector（最短路径，最慢）
        * `precomputed_features=False` 的 `visual_features`
                                               缓存的是**视觉塔原始特征** [B,49,768]，
                                               仍在计算图内跑 Projector —— **训练用这个**
        * `precomputed_features=True` 的 `visual_features`
                                               缓存的是**Projector 之后的** [B,49,896]，
                                               直接注入。只适用于 Projector 也被冻结的推理场景，
                                               训练时用它会直接报 "does not require grad"
        """
        if visual_features is not None and precomputed_features:
            vis = visual_features.to(self.device_)
        elif visual_features is not None:
            # 缓存的是视觉塔原始特征 -> 这里跑 Projector，保证梯度能回传
            proj_dtype = next(self.projector.parameters()).dtype
            vis = self.projector(visual_features.to(self.device_, dtype=proj_dtype))
        elif pixel_values is not None:
            vis = self.encode_images(pixel_values)
        else:
            raise ValueError("必须提供 pixel_values 或 visual_features 之一")

        inputs_embeds = self.build_inputs_embeds(input_ids, vis, attention_mask)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=(attention_mask.to(self.device_)
                            if attention_mask is not None else None),
            use_cache=False,
        )

        result: dict[str, Any] = {}
        if labels is None:
            if return_logits:
                result["logits"] = outputs.logits
            return result

        labels = labels.to(self.device_)
        if answer_only_loss:
            loss, n_tok, n_correct = masked_lm_loss_and_acc(outputs.logits, labels)
            result["loss"] = loss
            result["n_answer_tokens"] = n_tok
            result["n_correct"] = n_correct
        else:
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            result["loss"] = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        if return_logits:
            result["logits"] = outputs.logits
        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        precomputed_features: bool = False,
        micro_batch: int = 4,
        answer_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """手写自回归解码，返回 (新生成 token 张量 [B, max_new_tokens], 各样本有效长度)。

        `micro_batch`：生成时每个子批次处理多少条。因为本实现每步重算全序列
        （不用 KV cache），显存随 batch 线性增长 —— 实测 batch=8 时 6GB 显存会 OOM，
        所以默认拆成 4 条一组。

        `answer_start`：每个样本"答案第一个 token"的位置 [B]。
        强烈建议传入（collator 知道 labels，能精确算出来）。
        不传时退回"用户段终止符之后若干 token"的启发式估计，会多花几个 token 生成
        对话结构（实测会先吐出 '\\n<|im_start|>assistant\\n' 四个 token）。
        """
        if input_ids.size(0) > micro_batch:
            bufs, lens = [], []
            for s in range(0, input_ids.size(0), micro_batch):
                vf = visual_features[s:s + micro_batch]
                b, l = self.generate(
                    input_ids=input_ids[s:s + micro_batch],
                    visual_features=vf,
                    attention_mask=(attention_mask[s:s + micro_batch]
                                    if attention_mask is not None else None),
                    max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=temperature, top_k=top_k, eos_token_id=eos_token_id,
                    precomputed_features=precomputed_features, micro_batch=micro_batch,
                    answer_start=(answer_start[s:s + micro_batch]
                                  if answer_start is not None else None),
                )
                bufs.append(b)
                lens.extend(l)
            return torch.cat(bufs, dim=0), lens

        self.eval()
        eos = int(eos_token_id if eos_token_id is not None
                  else (self.tokenizer.eos_token_id or 151643))
        ids = input_ids.to(self.device_)
        mask = (attention_mask.to(self.device_) if attention_mask is not None
                else torch.ones_like(ids))

        if precomputed_features:
            vis = visual_features.to(self.device_)
        else:
            proj_dtype = next(self.projector.parameters()).dtype
            vis = self.projector(visual_features.to(self.device_, dtype=proj_dtype))
        embeds = self.build_inputs_embeds(ids, vis, mask)

        # ⚠️ 解码循环踩过的 5 个坑（逐条记录，都是实测出来的）：
        #  1) **锚点必须在"答案之前"，而不是"答案之后"。**
        #     训练数据形如：  ... assistant \n  <answer> <|im_end|> <pad...>
        #     正确的生成条件是"喂到 `assistant \n`"，模型才会接着吐答案。
        #     若喂到答案的 `<|im_end|>` 再预测，等于问"序列结束符之后说什么"，
        #     模型只会输出 padding token（'\n'）并无限重复。
        #  2) **不能取第一个 `<|im_end|>`。** Qwen 的 chat template 里
        #     system / user 每一段都以 `<|im_end|>` 结尾（实测序列里 eos 出现在
        #     位置 [19, 81, 91]）。必须用**最后一个**来定位答案区间；
        #     取第一个会把锚点定到 system 段落中间，预测出 ' You' 这种 prompt 续写。
        #  3) **每个样本的答案长度不同**（4~8 个 token），所以截断长度必须逐样本算，
        #     不能用一个全局的 max 长度。
        #  4) 右 padding 的 batch 里，序列的"末尾"不是张量的最后一个位置。
        #  5) 把变长序列 `cat` 成 [总token数, D] 交给模型**不行**：
        #     Qwen2 推断不出 batch 维，会把整段当成 1 行，position_ids 错乱，
        #     RoPE 报 "size of tensor a (14) must match the size of tensor b (64)"。
        #     **必须保持矩形张量 [B, L] + attention_mask。**
        #
        # 最终方案：矩形张量 + attention_mask，把输入截断到各样本的"答案前一个位置"，
        # 之后每步把已生成的 token 追加在截断点之后，每步重算全序列（不用 KV cache）。
        # 序列只有约 100 个 token，重算成本远低于调试错的解码循环的代价。
        # 锚点（答案区间的起点）：
        #   训练序列结构 = <|im_end|>(用户段) \n <|im_start|> assistant \n <answer> <|im_end|>
        #   「用户段终止符」的下一位是 `\n`，再往后两位才是答案正文；
        #   这里取"最后一个 <|im_end|> 之前"的位置作为生成起点，
        #   保证模型是从 prompt 末尾（含 assistant 头）开始续写，而不是从答案中间续写。
        eos_mask = (ids == eos)
        B = ids.size(0)
        start_pos: list[int] = []       # 生成起点（= 已确定的前缀长度）
        if answer_start is not None:
            a = answer_start.to(self.device_).tolist()
            start_pos = [int(x) for x in a]
            for i in range(B):
                start_pos[i] = min(max(start_pos[i], 1), int(mask[i].sum().item()))
        else:
            # 启发式回退：用户段终止符之后 4 个 token
            # （即 '\n' '<|im_start|>' 'assistant' '\n'），精度不如显式传入
            for i in range(B):
                valid = int(mask[i].sum().item())
                row_eos = torch.nonzero(eos_mask[i, :valid], as_tuple=False).flatten()
                if row_eos.numel() < 2:
                    start_pos.append(valid)
                else:
                    start_pos.append(min(int(row_eos[-2].item()) + 5, valid))

        max_start = max(start_pos)
        # Each row has its own answer start. Generated tokens must be written
        # immediately after that row's prefix, not after the longest row's prefix.
        # Otherwise shorter rows contain an attention-mask gap and their next
        # logits are read from a padding position (the old batched decode bug).
        work_ids = torch.full((B, max_start + max_new_tokens), eos,
                              dtype=torch.long, device=self.device_)
        work_ids[:, :max_start] = ids[:, :max_start]
        work_mask = torch.zeros_like(work_ids)
        for i, p in enumerate(start_pos):
            work_mask[i, :p] = 1

        gen_buf = torch.full((B, max_new_tokens), eos, dtype=torch.long, device=self.device_)
        gen_len = [0] * B
        finished = [False] * B

        for step in range(max_new_tokens):
            if all(finished):
                break
            window_end = max(start_pos[i] + gen_len[i] for i in range(B))
            cur_ids = work_ids[:, :window_end]
            cur_mask = work_mask[:, :window_end]
            if _DEBUG_VERBOSE:
                print(f"    [verbose] step={step} 窗口={tuple(cur_ids.shape)} "
                      f"start_pos={start_pos[0]} gen_len={gen_len[0]} "
                      f"末尾有效位置={start_pos[0] + gen_len[0] - 1}")
            embeds = self.build_inputs_embeds(cur_ids, vis, cur_mask)
            out = self.llm(inputs_embeds=embeds, attention_mask=cur_mask, use_cache=False)
            logits = out.logits.float()

            nxt_ids: list[int] = []
            for i in range(B):
                if finished[i]:
                    nxt_ids.append(eos)
                    continue
                # 取"当前窗口最后一个有效位置"的 logits：
                # 窗口 = 前缀(start_pos[i] 个) + gen_len[i] 个已生成 token，
                # 所以末位下标 = start_pos[i] - 1 + gen_len[i]
                # （step 0 时 gen_len=0，退化为 start_pos[i]-1 = 前缀末位）。
                at = start_pos[i] - 1 + gen_len[i]
                at = min(max(at, 0), logits.size(1) - 1)
                step_logits = logits[i, at]
                if do_sample:
                    lg = step_logits / max(temperature, 1e-6)
                    if top_k:
                        k = min(top_k, lg.size(-1))
                        thresh = torch.topk(lg, k).values[-1]
                        lg = lg.masked_fill(lg < thresh, float("-inf"))
                    tok_id = int(torch.multinomial(F.softmax(lg, dim=-1), 1).item())
                else:
                    tok_id = int(step_logits.argmax().item())
                nxt_ids.append(tok_id)
                write_at = start_pos[i] + gen_len[i]
                work_ids[i, write_at] = tok_id
                work_mask[i, write_at] = 1
                gen_buf[i, gen_len[i]] = tok_id
                gen_len[i] += 1
                if tok_id == eos:
                    finished[i] = True

        return gen_buf, gen_len

    @torch.no_grad()
    def generate_text(self, *args, **kwargs) -> list[str]:
        """返回文本列表：**只取第一个终止符之前的答案正文**。

        两个截断细节：
        1. 训练样本是完整对话（prompt + answer + `<|im_end|>` + padding），
           所以模型的续写里可能接上"下一轮对话"（实测出现过 'user\\n'、'Assistant' 等）。
           答案的边界就是**第一个终止符**，其后的内容全部丢弃。
        2. 答案区间的首个 token 是 `assistant\\n` 里的那个换行符
           （实测监督位置从答案区起点后的第一个 token 开始），展示时应 lstrip 掉。
        """
        buf, gen_len = self.generate(*args, **kwargs)
        eos = int(self.tokenizer.eos_token_id or 151643)
        out: list[str] = []
        for i, n in enumerate(gen_len):
            ids = buf[i, :n].tolist()
            if eos in ids:                     # 遇到第一个终止符就结束
                ids = ids[: ids.index(eos)]
            out.append(self.tokenizer.decode(ids, skip_special_tokens=True).lstrip())
        return out


# ======================================================================
# 损失
# ======================================================================
def masked_lm_loss_and_acc(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    """只对 labels != -100 的位置算交叉熵，同时返回 token 级准确率。

    用 `logits[:, :-1, :]` 对齐 `labels[:, 1:]`：
    位置 t 的 logits 用来预测位置 t+1 的 token —— 自回归语言模型的标准 shift。
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_labels = shift_labels.reshape(-1)
    keep = flat_labels != IGNORE_INDEX
    n_tok = int(keep.sum().item())
    if n_tok == 0:
        return logits.sum() * 0.0, 0, 0

    sel_logits = shift_logits.reshape(-1, shift_logits.size(-1))[keep].float()
    sel_labels = flat_labels[keep]
    loss = F.cross_entropy(sel_logits, sel_labels)
    n_correct = int((sel_logits.argmax(dim=-1) == sel_labels).sum().item())
    return loss, n_tok, n_correct


# ======================================================================
# 配置读取
# ======================================================================
def load_yaml_config(path: str | Path) -> dict:
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def config_from_yaml(path: str | Path = "configs/model_clip_b32_qwen05.yaml",
                     **overrides) -> MiniVLMConfig:
    raw = load_yaml_config(path)
    v, pj, l, d = raw["vision"], raw["projector"], raw["llm"], raw.get("data", {})
    tr = raw.get("training", {})
    lora = raw.get("lora", {})
    root = Path(__file__).resolve().parents[1]
    cfg = MiniVLMConfig(
        clip_path=str(root / v["name_or_path"]),
        llm_path=str(root / l["name_or_path"]),
        freeze_vision=bool(v["freeze"]),
        drop_cls=bool(v["drop_cls"]),
        projector_depth=int(pj["depth"]),
        projector_hidden=int(pj["hidden_dim"]),
        projector_dropout=float(pj.get("dropout", 0.0)),
        freeze_llm=bool(tr.get("freeze_llm", True)),
        strategy=str(tr.get("strategy", "projector_only")),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_target_modules=tuple(lora.get("target_modules",
                                           ["q_proj", "k_proj", "v_proj", "o_proj"])),
        num_image_token=int(d.get("num_image_token", 49)),
        max_seq_len=int(tr.get("max_seq_len", 768)),
    )
    for k, val in overrides.items():
        setattr(cfg, k, val)
    return cfg
