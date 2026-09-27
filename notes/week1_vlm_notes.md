# Week 1 笔记：一张图片究竟是怎么进入 LLM 的？

> 本文所有形状与数字均为**本机实测**（RTX 4050 Laptop 6GB / Python 3.12.4 / torch 2.6.0+cu124 / transformers 5.17.0）。
> 复现命令见文末。凡是没实测的地方，我会明确标注"待验证"。

---

## 一、总链路与张量形状

```text
图片 PIL.Image (336x336)
  │  CLIPProcessor
  ▼
pixel_values          [B, 3, 224, 224]        ← 强制拉到 224x224
  │  VisionEncoder (CLIP ViT-B/32, 冻结)
  ▼
visual features       [B, 50, 768]            ← 50 = 1 CLS + 49 patch
  │  丢弃 CLS
  ▼
patch features        [B, 49, 768]            ← 49 个 patch token
  │  Projector (自己写的 MLP, 唯一可训练)
  ▼
visual embeddings     [B, 49, 896]            ← 896 = Qwen2.5-0.5B 的 hidden_size
  │  与文本嵌入拼接（替换占位符位置）
  ▼
inputs_embeds         [B, L, 896]             ← 实测 L = 104
  │  Qwen2.5-0.5B (冻结 / LoRA)
  ▼
logits                [B, L, 151936]          ← 151936 = 词表大小
```

**实测样本（一条真实构造样本）**

```text
input_ids.shape = (1, 104)     其中 49 个位置是视觉占位符
answer 起始位置   = 88
参与 loss 的位置  = 88..98（11 个 token）：'图中有 2 个黑色方块。<|im_end|>'
logits.shape    = (1, 104, 151936)
```

### 每个 Tensor 的维度含义

| 维度 | 符号 | 实测值 | 说明 |
| --- | --- | --- | --- |
| B | batch | 1（训练时 8） | 每次前向喂几张图 |
| N | visual token 数 | **49** | `(224/32)² = 49`，由 image_size 与 patch_size 决定 |
| Dv | 视觉特征维度 | **768** | CLIP ViT-B/32 的 hidden_size |
| Dl | LLM 嵌入维度 | **896** | Qwen2.5-0.5B 的 hidden_size |
| L | 序列长度 | **104** | = 49 视觉 + 55 文本/特殊 token（该样本） |
| V | 词表大小 | **151936** | embedding 权重行数 |

**核心约束（硬性）**：`Projector.out_dim` 必须等于 LLM 的 `hidden_size`，否则拼接这一步根本做不了。本项目即 `768 → 896`。

---

## 二、为什么 Vision Encoder 不能直接接 LLM？

三个层面，缺一不可：

**1. 维度对不上（最表层）**

CLIP 输出 768 维，Qwen2.5 嵌入是 896 维。矩阵没法直接相加或替换，必须有一个映射 `768 → 896`。

**2. 语义空间不对齐（本质原因）**

CLIP 的视觉特征是**对比学习**训出来的：它的几何含义是"与某段文本的相似度"，与之对齐的是 CLIP 自己的文本塔（`text_projection`）。而 LLM 的嵌入空间是**自回归语言建模**训出来的，每个位置的向量含义是"下一个 token 的分布前提"。

这两个空间没有任何共享的训练信号——CLIP 从没见过 Qwen 的 tokenizer，Qwen 也从没见过图像。所以直接塞进去，对 LLM 来说就是**噪声**：维度即使凑对了，语义也是乱的。

**3. 训练信号缺失（工程原因）**

即便形状凑上，如果只冻结两边、不做任何对齐训练，梯度无法告诉视觉特征"该怎么变才对语言任务有用"。Projector 的作用正是提供一个**可训练的翻译层**，让梯度能从这里回传到"如何把图像说成人话"这件事上。

> **一句话**：CLIP 说的是"图文相似度"这门语言，LLM 说的是"下一个词"这门语言，Projector 是它们之间的翻译官。

---

## 三、Projector 在解决什么问题？

**它解决的是"跨模态对齐"，而不是"特征压缩"或"升维"这么简单。**

数学上，它是一个可学习映射：

$$P: \mathbb{R}^{768} \to \mathbb{R}^{896}, \qquad P(V) = W_2 \,\mathrm{GELU}(W_1 V)$$

三种候选结构（本项目 Week 3 的消融对象）：

| 结构 | 参数量（实测） | 说明 |
| --- | ---: | --- |
| Linear | 689,024 | 最简单，只有一次线性变换 |
| 2-layer MLP | 3,410,816 | LLaVA-1.0 的选择，非线性带来更强表达力 |
| 3-layer MLP | 7,607,168 | 更深，但收益未必更大且更易过拟合 |

**为什么需要非线性？** 视觉特征与语言嵌入之间的关系不是简单线性可分的——"49 个 patch 各自该被翻译成什么语义"需要非线性变换才能表达。这正是 LLaVA 从 Linear 升级到 2 层 MLP 的原因。

**关键洞察**：Projector 参数量只有 3.4M，而 LLM 是 0.5B。**我们只训练这 0.7% 的参数，就要让整个系统"学会看图"**——这就是 Experiment A 要回答的问题。

---

## 四、为什么图像可以表示成"一系列 Token"？

**因为 Transformer 只认一件事：一串等长的向量。**

它不关心这串向量从哪来——文本 token 的嵌入、音频帧的嵌入、图像 patch 的嵌入，在数学上完全同构。所以只要能把图像变成"若干个 D 维向量"，它就能和文本 token 一起进同一个 Transformer。

具体到 CLIP ViT-B/32：

```text
224x224 图像
  → 切成 32x32 的 patch        → 7 x 7 = 49 个 patch
  → 每个 patch 展平成 3*32*32 = 3072 维向量
  → 线性投影到 768 维           → 49 个 patch token
  → 加位置编码 + 一个 CLS token  → 50 个 token
  → 12 层 Transformer 自注意力
  → 输出 [B, 50, 768]
```

于是"图中有几个黑方块"这个问题，对 LLM 来说就变成：

```text
[视觉token x49][这][张][图][里][有][几][个][黑][方][块][？]
                └──────────── 普通的文本 token ────────────┘
```

**全部丢进同一次 Transformer 前向，靠 self-attention 自己建立对应关系。**

注意一个实测细节：**文本里只写 1 个 `<|image_pad|>`，但张量里必须展开成 49 个连续位置**，才能和 49 个视觉向量逐位置对齐。这是所有 VLM 都要做的一步（详见第七节）。

---

## 五、VLM 与普通 LLM 的最大区别是什么？

**最大区别不在架构，而在"输入嵌入的来源是多路的"。**

| 方面 | 普通 LLM | VLM |
| --- | --- | --- |
| 嵌入来源 | 只有 tokenizer → embedding 表 | embedding 表 + Projector 输出 |
| 输入构造 | `input_ids` 就够 | 必须自己拼 `inputs_embeds` |
| 序列内容 | 全是文本 token | 文本 token + 视觉占位符（被替换） |
| 可训练部分 | 全部 | 通常只训 Projector（+ LoRA） |
| 前向调用 | `model(input_ids=...)` | `model(inputs_embeds=...)` |
| 推理陷阱 | 少 | `generate()` 与 `inputs_embeds` 组合有坑（见第八节） |

**一句话**：LLM 的输入是"查表得到的"，VLM 的输入是"查表 + 计算 + 拼接得到的"。**这个拼接逻辑就是本项目的核心工作量。**

---

## 六、49 个 visual token 对训练成本意味着什么？

这是一个必须算清楚的账：

**1. 上下文预算被挤占**

每个样本固定占用 49 个 token 的位置。本样本总长 104，其中 **47% 是视觉 token**。如果换成 CLIP-B/16（224px），就是 196 个 patch token，序列长度变成 104 - 49 + 196 = **251**，接近 2.4 倍。

**2. 注意力成本是序列长度的平方**

$$ \text{self-attention 开销} \propto L^2 $$

- 49 visual token：L ≈ 104 → $104^2 = 10{,}816$
- 196 visual token：L ≈ 251 → $251^2 = 63{,}001$

**约 5.8 倍**。这解释了为什么第一版选 B/32 而不是 B/16：不是效果最好，而是**链路验证阶段成本最低**。

**3. 显存与训练时间**

序列越长，激活值越多。在 6GB 显存上，49 token 让 0.5B 模型 + bf16 LoRA 跑得动；196 token 就要减 batch 或上梯度检查点。

**4. 但视觉 token 有个巨大的成本优势**

因为视觉塔**冻结**，同一张图的视觉特征全程不变 → **可以离线预计算并缓存**：

```text
每张图 49 x 768 x fp16 = 75 KB
5 万张图               = 约 3.8 GB
```

训练时只读特征、不加载 CLIP、不做图像前向。这能把训练速度提升约 2–4 倍，是"80% 工作留在本地 4050"的技术依据。

> **结论**：视觉 token 数是一个**成本旋钮**。Week 7 的"视觉 Token 数量消融"实验就是量化这个旋钮的性价比。

---

## 七、实测踩到的坑（这一节最值钱）

### 坑 1：`drop_cls` 会误删第一个 patch

`CLIPVisionModel.last_hidden_state` 的形状是 **[B, 50, 768]**，第 0 个 token **就是 CLS**。如果以为"不含 CLS"而跳过切片，或者以为"要丢"却用错索引，都会静默地丢掉/错位一个 patch——模型照样能训，但输入从一开始就是错的。

**顺带纠正一个错误判据**：不能用 `last_hidden_state[:,0,:] ≈ pooler_output` 来判断第 0 个是不是 CLS。CLIP 的 `pooler_output` 是 CLS 再过一层 `tanh` 投影的结果，两者本就不相等，这个判据恒为 False。**判断只能靠 token 数**（50 = 49+1）。

**现在的实现**做了自动探测：

```python
if self.drop_cls and out.shape[1] == self.patch_tokens + 1:
    out = out[:, 1:, :]      # 确实多一个才切
```

### 坑 2：占位符必须从 1 个展开成 49 个

提问文本里只写一个 `<|image_pad|>`，但视觉侧产出 49 个向量。直接在 1 个位置上赋 49 个向量是不成立的（会报 shape mismatch）。

**正确做法**：把占位符**展开成 49 个连续位置**，再整段替换。

```text
文本侧：  <|im_start|>user\n [PAD] \n图中一共有几个黑色方块？<|im_end|>
展开后：  <|im_start|>user\n [PAD][PAD]...[PAD] \n图中...<|im_end|>
                          └──── 49 个 ────┘
注入后：  <|im_start|>user\n [v1][v2]...[v49] \n图中...<|im_end|>
```

展开只发生在占位符处，回答区间整体后移，**loss 掩码自动保持正确**（实测 answer_start 从 40 变 88）。

### 坑 3：`tokenizer.pad()` 不处理自定义字段

`Qwen2Tokenizer.pad()` 只认 `input_ids` / `attention_mask` / `token_type_ids`。把 `labels` 一起传进去，它会被**原样返回不做补齐**，于是 labels 长度 56 而 attention_mask 长度 51，报：

```text
RuntimeError: The size of tensor a (56) must match the size of tensor b (51)
```

**正确做法**：手动补齐三个张量，`labels` 用 `IGNORE_INDEX` 填充，并额外把 `attention_mask == 0` 的位置强制设为 `-100`。

### 坑 4：`datasets/` 目录名遮蔽了 HuggingFace 的 `datasets` 库

实测确认（不是理论推测）：

```python
>>> import datasets
D:\...\9.VLM\datasets\__init__.py      # 拿到的是本目录，不是 HF 库
```

后果：将来 `from datasets import load_dataset` 会失败或拿到错误对象，即使 `pip install datasets` 也一样（当前目录优先级最高）。本项目数据自建 JSONL，本来不依赖它；若某天需要，得改绝对导入或重命名目录。

### 坑 5：`<|image_pad|>` 不需要自己注册

计划书初稿曾假设"纯文本 Qwen2.5 没有这个 token，要 `add_special_tokens` + `resize_token_embeddings`"。**实测推翻**：

```text
<|image_pad|> id  = 151655        （已在词表，且是单一 token）
embedding 权重行数 = 151936        （比 151655 大，不越界）
len(tokenizer)    = 151665
```

所以 **不要** 注册、**不要** resize。resize 会重新绑定 `tie_word_embeddings=True` 的输入/输出嵌入并引入未训练参数，导致"训练正常但输出退化"这类极难排查的问题。

---

## 八、推理为什么不用 `model.generate()`

decoder-only 模型（Qwen2）的 `generate()` 与 `inputs_embeds` 组合有已知坑：`prepare_inputs_for_generation` 在后续 step 可能不再收到嵌入，表现为**"训练时 loss 正常下降，推理时却完全无视图片"**——这是 VLM 最贵的一类 bug（本地跑通、上云烧完钱才发现）。

另外实测还看到一条相关警告：

```text
UserWarning: Passing `repetition_penalty` with `inputs_embeds` and without `input_ids`
to `generate` will apply the penalty only to newly generated tokens, not to the prompt.
```

`generation_config.json` 里预置了 `repetition_penalty`，与 `inputs_embeds` 路径配合并不干净。

**项目纪律**：
- 推理第一版手写自回归循环，完全掌控拼接与 KV cache；
- 不管走哪条路，**必须验证"换一张图，答案位置的 logits 会变"**（本项目实测差值 2.35，远大于阈值 1e-3）；
- 该检查已固化为 `scripts/inspect_sample.py` 的断言 D。

---

## 九、本机实测参数速查表

```text
图片预处理      : CLIPProcessor -> [B, 3, 224, 224]
视觉塔          : CLIP ViT-B/32, image_size=224, patch_size=32, hidden=768, layers=12
                  输出 [B, 50, 768]（含 CLS）-> 丢 CLS -> [B, 49, 768]
Projector       : 768 -> 2048 -> GELU -> 896      参数量 3,410,816
LLM             : Qwen2.5-0.5B-Instruct
                  hidden=896, layers=24, heads=14, kv_heads=2
                  tie_word_embeddings=True, bos_token_id=None, max_position=32768
                  词表 151643（base）/ 151665（含 special）/ 权重 151936 行
特殊 token      : <|im_start|>=151644  <|im_end|>=151645
                  <|vision_start|>=151652  <|vision_end|>=151653
                  <|vision_pad|>=151654   <|image_pad|>=151655  <|video_pad|>=151656
pad             : 无 pad_token，用 eos(<|endoftext|>=151643) 充当
序列长度        : 单张图 + 短问答 ≈ 104（其中 49 为视觉 token）
```

---

## 十、Week 1 验收问题（自答）

| 问题 | 答案要点 |
| --- | --- |
| 为什么 Vision Encoder 不能直接接 LLM？ | ① 维度不同（768 vs 896）② 语义空间无共享训练信号（对比学习 vs 自回归）③ 缺梯度通路 |
| Projector 在解决什么问题？ | 跨模态对齐：把 CLIP 的"图文相似度空间"翻译成 LLM 的"下一词空间"，且是唯一可训练模块 |
| 为什么图像能表示成一系列 Token？ | Transformer 只要求"一串等长向量"；图像切 patch → 投影 → 就是 49 个 768 维向量，与文本同构 |
| VLM 与普通 LLM 的最大区别？ | 输入嵌入多了一路来源：必须自己拼 `inputs_embeds` 并处理占位符替换，而不是直接喂 `input_ids` |
| 49 个 visual token 意味着什么？ | 占序列 47%；注意力成本 ∝ L²，换 B/16 约涨 5.8 倍；但视觉塔冻结 → 可离线缓存，75KB/图 |

---

## 十一、复现命令

```powershell
cd D:\学习\深度学习\dl_project\9.VLM
. .\scripts\env.ps1

& $env:MINIVLM_PY scripts/check_env.py --deep      # 环境 + 离线真实性
& $env:MINIVLM_PY scripts/verify_offline.py        # 离线加载 + 9 项断言
& $env:MINIVLM_PY short_time_check.py              # 视觉塔冒烟（形状）
& $env:MINIVLM_PY models/projector.py              # 三种 Projector 参数量表
& $env:MINIVLM_PY scripts/inspect_sample.py        # 打印一条完整样本（4 项断言）
```

`inspect_sample.py` 的 4 项断言（全通过才算 Week 1 验收完成）：

```text
A. 占位符数量 == 视觉 token 数          49 == 49
B. loss 只在 answer 区间开启            前 88 个全为 -100，后 11 个参与 loss
C. 只有占位符位置被替换                  占位符已注入，非占位符位置未被改动
D. 换图导致答案位置 logits 变化           max|Δ| = 2.35  (> 1e-3)
```

---

## 十二、下一周要做的事（Week 2 预习）

1. 跑 `scripts/inspect_sample.py --image <真实图片>`，确认真实图片路径也通过（注意：`--image` 会关闭"换图对比"，需另找一张图自行验证）
2. 补 `scripts/cache_features.py`：把 1000 张图离线算成特征缓存，统计单图耗时与缓存体积
3. 补 `configs/model_clip_b32_qwen05.yaml` 的真实内容：`num_image_token: 49`、`vision_hidden: 768`、`llm_hidden: 896`
4. 把 `tests/test_projector.py` 与 `tests/test_collator.py` 写成真实断言（目前是 0 字节）
