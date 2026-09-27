# Week 6 笔记：LoRA —— 一个诚实的否定结果

> 全部数字来自本机实测（RTX 4050 6GB，本地 ¥0）。
> 实验记录：`report/experiments.md` 的 EXP-003。

---

## 一、假设与验证设计

**假设**（来自 Week 4 的结论）：数据量不是瓶颈（1k→10k 只涨 2 个百分点），
那么瓶颈可能在**冻结 LLM 的表达能力** —— Projector 只能把视觉特征"翻译"成 LLM 的语言，
却改不了 LLM 自己的行为习惯（例如"答完不停"）。

**验证方式**：给 LLM 注入 LoRA，让它本身也能被微调，其余条件与 EXP-002 完全一致。

```powershell
& $env:MINIVLM_PY scripts/train.py --strategy lora `
    --data data/processed/baseline_10k.jsonl --out outputs/lora_10k `
    --steps 1200 --batch 4 --accum 8 --lr 1e-3 --lora-lr 2e-4 `
    --warmup 100 --eval-every 100 --gen-samples 200 --no-gpu-cache
```

### 配置

| 项 | 值 |
| --- | --- |
| LoRA | r=16, alpha=32, dropout=0.05, **只做 attention 的 q/k/v/o_proj** |
| LoRA 参数量 | **2,162,688**（192 个张量，24 层 × 4 投影 × 2 矩阵） |
| Projector | 3,410,816 |
| **总可训练** | **5,573,504 / 587,062,272（0.95%）** |
| 学习率 | projector **1e-3**，LoRA **2e-4**（严格分开 —— Week 1 定下的纪律） |
| 其余 | 与 EXP-002 完全一致（10k 数据、1200 步、等效 batch 32） |

---

## 二、结果：LoRA 没有帮助

**严格同口径对比**（两个模型都评 val 前 200 条）：

| 指标 | Projector-only（A） | **+ LoRA（B）** | 变化 |
| --- | ---: | ---: | ---: |
| 总体精确匹配 | **32.0%** | 28.5% | **−3.5%** |
| 总体前缀匹配 | **51.0%** | 50.0% | −1.0% |
| 最优 val_loss | 0.1051 | **0.0949** | **−0.010** ✅ |
| val token 准确率 | 95.5% | **96.0%** | +0.5% |

**按任务**（val 200）：

| 任务 | A 精确 | B 精确 | 多数类基线 | 谁更好 |
| --- | ---: | ---: | ---: | --- |
| counting | 56.2% | 56.2% | 12.1% | 平 |
| existence | 20.9% | **25.6%** | 12.8% | **B** |
| listing | 33.3% | 33.3% | 10.6% | 平 |
| spatial | **48.6%** | 34.3% | 51.0% | **A** |
| attribute | **9.5%** | 0.0% | 22.8% | **A** |
| **超过基线的任务数** | **3 / 5** | 3 / 5 | | 平 |

**test 集 522 条全量**：A = 27.0% 精确，B = 23.0% 精确 —— 同样 A 更好。

---

## 三、为什么 val 上好、test 上差？——一个必须警惕的信号

| split | A 精确 | B 精确 |
| --- | ---: | ---: |
| val（489 条） | 32.0%（200 条子集） | 28.5%（200 条子集） |
| **test（522 条全量）** | **27.0%** | **23.0%** |

B 在 val 上的某些任务看起来很亮眼（existence 12.5% → 25.6%），
但**在 test 上没有兑现**（11.5%，与 A 的 12.5% 持平）。

这说明 LoRA 版的模型**在验证集上过拟合了**。教训：
**任何"提升"都必须在 test 集上确认，val 上的波动很容易被误读成改进。**

---

## 四、最关键的诊断：视觉接地（grounding）

我加了一个对照实验：把视觉特征**错位一位**（图片与问题不再对应），看准确率掉多少。
**掉得越多，说明模型越依赖视觉信息。**

| 模型 | 原图精确 | 特征错位后 | **差值（接地强度）** |
| --- | ---: | ---: | ---: |
| Projector-only | 32.0% | 29.0% | **+3.0%** |
| + LoRA | 28.5% | 25.0% | **+3.5%** |

**结论**：LoRA 并没有让模型更多地依赖视觉信息（接地强度基本相同）。
它主要是在**适配语言分布**，而不是提升视觉理解。

这解释了为什么 val_loss 降了（语言建模更贴合训练数据）但生成质量没提升
（真正需要的是看图能力）。

---

## 五、机制层面的解释

LoRA 的更新量 `BA` 从零开始（B 初始化为零），在 1200 步、lr=2e-4 下，
适配器几乎没有被显著训练 —— val_loss 只从 0.1051 降到 0.0949（−10%）。

**但真正重要的不是"训练不足"，而是这个事实**：

> **容量不是瓶颈。** 只训 0.58% 的 Projector 就能达到的效果，
> 加到 0.95%（含 LLM 的 LoRA）反而更差。
> 说明限制模型表现的是**输入信息的质量**，而不是可训练参数的数量。

---

## 六、诚实的局限说明

这个否定结论有一个重要的前提没被充分探索：

**LoRA 的超参没有调过。** 可能的原因包括：

| 可能问题 | 为什么可能 | 如何验证 |
| --- | --- | --- |
| LoRA lr 偏低 | 2e-4 是常见值，但 LoRA 从零初始化，可能需要更大 lr 或更多步 | 试 5e-4 / 1e-3，或步数翻倍 |
| 只做了 attention | 没有动 MLP 层（`gate_proj`/`up_proj`/`down_proj`），容量有限 | 把 MLP 也加进 target_modules |
| r=16 偏小 | 0.5B 模型上 r=16 通常够用，但不绝对 | 试 r=32 / r=64 |
| alpha/r 比例 | alpha=32, r=16 → scaling=2，可能偏大 | 试 alpha=16 |

**所以准确的结论应该是**：
> 在"r=16 + 只做 attention + lr=2e-4 + 1200 步"这一套默认配置下，
> **LoRA 没有带来可确认的收益**（val_loss 略降，但 test 生成质量与视觉接地都未提升）。

而不是"LoRA 对这个任务无效"。这个区分很重要 —— 写报告时要如实表述。

---

## 七、下一步：转向"视觉输入质量"

既然容量不是瓶颈、数据量也不是，下一个最可能的瓶颈是**视觉信息的质量**：

```text
CLIP ViT-B/32 @224  ->  49 个 patch token  ->  7×7 的网格
```

**7×7 的网格对"判断左右"这种空间任务太粗了** ——
每个 patch 覆盖 32×32 像素，人物左右关系的细节很可能在这个粒度下就丢了。
这与实测吻合：spatial 任务在 A 下 48.6%，仍低于"永远猜多数边"的 51.0%。

**Week 7 要做的实验（视觉 Token 数量消融）**：

| 设置 | N_visual | 说明 | 成本 |
| --- | ---: | --- | --- |
| CLIP-B/32 @224（当前） | 49 | 基线 | 已有结果 |
| CLIP-B/32 @256 | 64 | 提高输入分辨率 | 需下载 config 变体或改 image_size |
| **CLIP-B/16 @224** | **196** | patch 变小，空间分辨率 ×2 | 需下载 B/16 权重（~600MB） |
| 49 → 池化到 16 | 16 | 反向验证"token 越多越好" | 本地可做 |

这个实验**比继续调 LoRA 超参更有信息量**，因为它直接检验一个机制性假设：
空间任务失败，是不是因为视觉分辨率不够。

---

## 八、顺手修掉的一个危险 bug（**静默的部分加载**）

`scripts/evaluate.py` 从**配置文件**读 `strategy`，而配置里是 `projector_only`。
用它去加载 LoRA checkpoint 时：

- LoRA 权重**被静默丢弃**（模型里没有对应参数）
- 评测结果从 50% 掉到 **0.8%**
- **脚本一声不吭，还打印"已加载"**

更糟的是我第一版诊断脚本打了"键不匹配 0 个"——**误报**，
因为 `ckpt.get("lora") or {}` 在加载为空时返回空字典，`missing` 自然是空的。

**两处修复**：

1. `Trainer.load_checkpoint(strict=True)`：checkpoint 里有、模型里没有的参数
   → **直接报错**并提示"常见原因：用 projector_only 的配置加载 LoRA checkpoint"。
2. `scripts/evaluate.py`：从 **checkpoint 的 `config` 字段**读回真正用过的 strategy，
   而不是相信当前配置文件。

> **教训**：静默的部分加载比直接报错危险得多。
> 任何"加载权重"的代码都应该默认 `strict=True`，并明确报告加载了多少张量。

---

## 九、Week 6 交付物

```text
models/lora.py                      LoRA/QLoRA 注入 + 自检 + QLoRA 明确报错
models/multimodal_model.py          新增 strategy 支持（projector_only/lora/qlora）
training/trainer.py                 checkpoint 保存/加载 LoRA + strict 校验
scripts/train.py                    --strategy / --lora-lr / --lora-r / --grad-ckpt
scripts/evaluate.py                 从 checkpoint 读回 strategy（防静默丢权重）
configs/model_clip_b32_qwen05.yaml  策略说明 + LoRA 参数量估算
outputs/lora_10k/                   checkpoint（21MB，含 192 个 LoRA 张量）
report/figures/train_curve_lora_10k.png
```

### 关于 QLoRA（Experiment C）

**本机未安装 bitsandbytes**（计划书 R1 风险，概率：高）。

代码里已经做了明确处理：`models/lora.py::prepare_qlora` 在导入失败时抛出带解决方案的错误，
而不是让训练跑到一半才崩：

```text
QLoRA 需要 bitsandbytes，但导入失败：
  解决方式（任选）：
   1) 用 WSL2 + Linux 环境（推荐，坑最少）
   2) 改用纯 bf16 LoRA（use_qlora=False，Windows 上的默认路径）
   3) 把 QLoRA 实验放到云端 Linux 实例上跑
```

**鉴于 Experiment B 已经是否定结果**，Experiment C（QLoRA）的优先级应当下调：
QLoRA 与 LoRA 的差别只是"基础模型是否量化"，既然 LoRA 没带来收益，
QLoRA 也不太可能改变结论 —— **除非**先确认 LoRA 是训练不足（那就该先调 LoRA 超参）。

**建议**：把云预算留给 **Week 7 的视觉 Token 消融**（需要跑多组对照），
而不是 QLoRA。

---

## 十、费用

| 项 | 花费 |
| --- | ---: |
| LoRA 训练（1200 步，42 分钟） | **¥0** |
| 累计 | **¥0** |
