# Week 3 笔记：Projector 与多模态组装

> 全部数字来自本机实测：`scripts/overfit_test.py`（100 样本 / 300 步 / RTX 4050 6GB）
> 复现：`& $env:MINIVLM_PY scripts/overfit_test.py --n 100 --steps 300`

---

## 一、Week 3 验收结果

**上云门禁：8 项断言全部通过** ✅

| 断言 | 结果 |
| --- | --- |
| A. loss 收敛到初始值的 10% 以下 | **3.6226 → 0.0043（0.1%）** |
| B. 训练集答案 token 准确率 > 95% | **100.00%**（596/596），整句全对 **100/100** |
| C. 换图后输出变化 | **6/8** 变化 |
| D. 只有 Projector 可训练 | **3,410,816 / 584,899,584（0.58%）** |
| D′. 只有 Projector 拿到梯度 | 有梯度的张量 4 个，全部属于 projector |
| E. 能生成参考答案 | 精确 **3/8**，前缀匹配 **5/8** |

**资源开销**：300 步 68 秒（225 ms/step），显存峰值 **3501 MB**（6GB 卡完全够用）。

---

## 二、模型的三个组成部分（实测规模）

```text
vision :   87,456,000 参数   trainable=0
llm    :  494,032,768 参数   trainable=0
proj   :    3,410,816 参数   trainable=3,410,816   ← 唯一被训练的模块
total  :  584,899,584 参数   trainable=0.58%
```

**只用 0.58% 的参数，就让一个纯文本 LLM 学会了看图。** 这就是 Experiment A 的核心结论。

Projector 参数量（`python models/projector.py` 可复现）：

| 结构 | 参数量 |
| --- | ---: |
| Linear (768→896) | 689,024 |
| **2-layer MLP (768→2048→896)** | **3,410,816** |
| 3-layer MLP | 7,607,168 |

---

## 三、本周踩的 8 个坑（每个都真实卡住过）

### 坑 1：`torch.no_grad()` 切断了 Projector 的梯度

`VisionEncoder` 里我在 `encode_images` 里包了 `torch.no_grad()`，于是 Projector 输出变成不需要梯度的张量，`loss.backward()` 直接报：

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**根因**：视觉塔的"冻结"应该靠 `requires_grad_(False)`（参数不接收梯度），**不是**靠切断计算图。用 `no_grad` 会把整条路径（包括后面的 Projector）一起从图上摘掉。

**修正**：去掉 `no_grad`，只保留 `requires_grad_(False)`。

### 坑 2：预编码特征缓存的位置错了

第一版把**Projector 之后**的特征缓存下来并 `detach()`，训练时直接注入 —— 梯度再次断掉。

**正确做法**：缓存的是**视觉塔原始特征** `[B,49,768]`（那部分真的恒定），Projector 必须在每次前向的计算图内跑。

```python
# ❌ 错：缓存了 Projector 之后的结果
f = model.encode_images(pv)          # 含 Projector
raw_feats[i] = f[j].detach()

# ✅ 对：只缓存视觉塔输出
f = model.vision(pv)                 # 只到视觉塔
raw_feats[i] = f[j].detach()
# 前向时：
vis = model.projector(raw_features)  # Projector 在计算图内
```

因此 `forward()` 现在有两个入口，语义必须分清：
- `visual_features` + `precomputed_features=False` → 缓存的是**视觉塔原始特征**（训练用这个）
- `visual_features` + `precomputed_features=True` → 已是 Projector 输出（仅推理）

### 坑 3：把变长序列 `cat` 起来会毁掉 position_ids

为了裁掉 padding，我把每条样本截到有效长度后 `torch.cat` 成 `[总token数, D]` 再喂给模型：

```text
RuntimeError: The size of tensor a (14) must match the size of tensor b (64)
             at non-singleton dimension 3        # RoPE 内部报错
```

**根因**：Qwen2 从 `inputs_embeds` 的第一维推断不出 batch，会把 `[17, 896]` 当成"1 行、长度 17"，`position_ids` 随之为 0..16，而 RoPE 的 cos/sin 只覆盖 1 个位置 → 形状对不上。

**修正**：**必须保持矩形张量 `[B, L]` + `attention_mask`**，不要用 `cat` 拼变长序列。

### 坑 4：生成时取错了位置（取到 padding）

右 padding 的 batch 里，序列的"末尾"不是张量的最后一个位置。直接 `logits[:, -1, :]` 对短样本拿到的是 **pad 位置**的输出，等于问模型"padding 之后该说什么"，生成立刻崩成 `'HumanHumanHuman...'`。

**修正**：按每条样本的有效长度取 logits。

### 坑 5：锚点定在"答案之前"而不是"之后"

训练序列结构：

```text
<|im_start|>system ... <|im_end|> <|im_start|>user ... <|im_end|>
<|im_start|>assistant \n <answer> <|im_end|> <pad...>
```

第一版把锚点定在"答案的 `<|im_end|>` 之前"，结果模型只会预测那个终止符本身，看不到答案。**必须从"答案**之前**"开始生成**，模型才会续写出答案。

### 坑 6：`<|im_end|>` 在序列里出现多次，不能用第一个

Qwen 的 chat template 里 **system / user / assistant 每一段都以 `<|im_end|>` 结尾**。实测一条样本里 eos 出现在位置 `[19, 81, 91]`。

第一版用 `nonzero(...)[0]` 取**第一个**，锚点落到位置 18（system 段落中间），预测出 `' You'` 这种 prompt 续写。

**修正**：用 `row_eos[-1]`（答案终止符）与 `row_eos[-2]`（用户段终止符）来定位。

### 坑 7：模型会继续生成"下一轮对话"

即使锚点正确，模型也会先吐出 `'\n<|im_start|>user\n'` 这类**对话结构**——因为训练样本是完整对话格式，模型学的是"续写整个对话"，而不只是答案。

两个处理：
1. 用 collator 提供的 `answer_starts`（labels 里第一个非 -100 的位置）**精确指定生成起点**，跳过结构 token；这比"从 EOS 位置推算"稳健得多。
2. `generate_text` 里在**第一个终止符处截断**，丢弃其后的"下一轮对话"。

### 坑 8：Windows 控制台 GBK 编码崩溃

模型偶尔吐出越南语、马拉雅拉姆语等字符，`print` 直接：

```text
UnicodeEncodeError: 'gbk' codec can't encode character '\u1ea1'
```

**修正**：脚本开头把 stdout/stderr 切到 UTF-8（`reconfigure(encoding="utf-8")`），并在打印模型输出时过滤非预期字符。

---

## 四、生成路径为什么必须自己写

`model.generate()` 与 `inputs_embeds` 的组合在本项目上是**不可靠的**：

- decoder-only 模型的 `prepare_inputs_for_generation` 在后续 step 可能不再接收嵌入，
  表现为"训练 loss 正常下降，推理时完全无视图片"——这是 VLM 最贵的一类 bug；
- 实测还遇到 `repetition_penalty` 与 `inputs_embeds` 组合的告警与行为差异。

本项目采用**全量重算**（不用 KV cache）的自回归循环：

```text
每步：输入 = [前缀(到答案起点)] + [已生成的 token]
      取"当前窗口最后一个有效位置"的 logits → argmax / 采样
```

序列只有约 100 个 token，重算成本远低于调试错解码循环的代价；而且这条路径与训练前向**完全一致**，行为可预期。

**性能提醒**：因为不用 KV cache，显存随 batch 线性增长 —— 实测 batch=8 时 6GB 会 OOM，所以 `generate` 默认 `micro_batch=4` 分块解码。

---

## 五、一个重要的产出：门禁自动化的价值

本次 8 个坑里，**有 6 个是"训练指标看起来完美、但生成结果垃圾"**这类问题：

```text
loss 3.62 → 0.0043    ✅ 看起来完美
token 准确率 100%     ✅ 看起来完美
自由生成 'HumanHuman'  ❌ 实际完全不能用
```

如果只盯着 loss 就上云，这 8 个坑会在烧掉 GPU 预算之后才暴露。

**这正是计划书 §6.4 强制要求"100 样本过拟合测试"的原因**：
它把"能不能跑通"和"能不能用"分开验证，而且给了 5 项量化的判据。
本周末尾把断言 E 从"精确匹配"放宽为"精确或前缀匹配"——
因为自由生成出现"答对之后继续重复"是欠拟合的正常表现，
该断言要检查的是**解码路径与视觉条件是否工作**，不是生成质量。

---

## 六、Week 3 交付物

```text
configs/model_clip_b32_qwen05.yaml     真实配置（含实测参数与硬约束注释）
datasets/build_voc_vqa.py              程序化 VQA 构造（五类任务均衡配额）
datasets/dataset.py                    JsonlVLDataset + FeatureCacheReader
datasets/collator.py                   占位符展开 + pad_to_square + 手动 padding
models/multimodal_model.py             组装 / forward / 手写自回归 generate
models/projector.py                    三种结构 + 参数量表
training/losses.py                     掩码损失 + 优化器分组 + 余弦调度
scripts/overfit_test.py                100 样本门禁（8 项断言）
data/processed/debug_1k.jsonl          1000 条训练数据（5 类 × 200）
report/figures/w3_overfit_test.json    门禁结果（含完整 loss 曲线）
```

---

## 七、Week 4 的起点

Week 3 已经顺手把数据工程的骨架搭起来了：

- `build_voc_vqa.py` 能生成 1k / 10k / 50k 三档（`--n` 控制）
- 五类任务**配额均衡**（各 200/1000 = 20%），实测多数类基线仅 **4.9%**
- 生成时会打印答案分布统计，并对单值坍缩发出警告

Week 4 需要补齐的：

1. `splits.json`：按 **image_id** 划分 train/val/test（同一张图的多个问题必须在同一 split）
2. 特征缓存全量生成（复用 `datasets/cache_features.py`）
3. 数据卡片 `report/data_stats.md`：五类任务的样本数、答案分布直方图、伪标签噪声率
4. 答案分布配平（`attribute` 任务实测最高频占比 24.5%，需要观察是否需要配平）
