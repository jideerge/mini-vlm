# Mini-VLM 项目计划书
## 基于视觉编码器与大语言模型的轻量级视觉语言模型构建与训练

| 项目代号 | Mini-VLM |
| --- | --- |
| 项目定位 | 个人主力简历项目（第一个"大型项目"） |
| 周期 | 8 周（约 80–120 小时，每周 10–15 小时） |
| 硬件 | 本地 RTX 4050 Laptop (6GB) + 按需租用云 GPU |
| 预算上限 | ¥1000（目标实际支出 ¥200–500） |
| 起始条件 | 已掌握 Transformer / ViT（CNN → ResNet → YOLO → Transformer → ViT 已完成） |
| 计划书版本 | v1.0 |
| 制定日期 | 待填写 |

---

## 0. 一页速览（Executive Summary）

### 0.1 项目目标（重新定义后的准确表述）

> **不是"做一个能调用的多模态模型"，而是从 Vision Encoder + LLM + Projector 出发，亲手完成一个小型 VLM 的核心连接、训练、评测和部署。**

以此为原则，本项目的**自主实现范围**与**明确不实现范围**如下：

| 我们亲手实现 | 我们直接复用（不造轮子） |
| --- | --- |
| Dataset / Data Collator | LLM 的 Transformer 层 |
| Vision feature extraction 封装 | Tokenizer |
| Projector（Linear / 2-layer MLP / 3-layer MLP） | AdamW 优化器实现 |
| 多模态 Embedding 拼接逻辑（visual token 注入） | Attention kernel / FlashAttention |
| Training pipeline（含 loss mask） | CLIP / SigLIP 预训练权重 |
| Evaluation pipeline（VQA / Caption / Counting） | CUDA kernel |
| Inference pipeline | 分布式训练框架（accelerate 仅作薄封装） |
| Gradio Demo / 一键复现脚本 | — |

判断标准很简单：**凡是"连接两个预训练模型"的部分，我们自己写；凡是"训练一个模型本身"的部分，我们用成熟实现。** 这正是 LLaVA 一类工作的核心思想。

### 0.2 最终系统形态

```text
                    图片  +  问题
                     │
                     ▼
              Vision Encoder (CLIP, frozen)
                     │
              Visual Features [B, N, Dv]
                     │
                     ▼
                Projector (自己实现, trainable)
                     │
             Visual Embeddings [B, N, Dl]
                     │
                     ├──────────────┐
                     ▼              ▼
              LLM Embedding  ←  Text Token Embedding
                     │
                     ▼
          ┌──────────────────┐
          │    LLM (Qwen2.5) │
          │  Frozen / LoRA   │
          └────────┬─────────┘
                   │
                   ▼
                 Answer
```

### 0.3 最终交付物清单（Definition of Done）

| # | 交付物 | 验收形式 |
| --- | --- | --- |
| 1 | 可运行的 Mini-VLM 模型 | `image + question → answer` 端到端跑通 |
| 2 | 5 万级图像指令数据集 | 5 类任务，含数据卡片（data card） |
| 3 | 两组以上训练实验 | Projector-only、Projector+LoRA（可选 QLoRA） |
| 4 | 至少 4 张实验表格 + 训练曲线 | Ablation 表格 + loss 曲线图 |
| 5 | 三任务评测体系 | VQA 准确率 / Caption 指标 / Counting 准确率 |
| 6 | Gradio Web Demo | 上传图 → 提问 → 回答；可公网访问 |
| 7 | GitHub 仓库 | 一条命令复现推理 |
| 8 | 技术报告（中文，8–15 页） | 架构、实验、消融、失败分析 |
| 9 | 成本台账 | 记录每一次云 GPU 的实际花费 |

### 0.4 最低成功标准 vs 理想成功标准

**最低成功标准（必须达成，否则项目不算完成）**

- 模型能完成 `Image + Question → Answer`
- 至少 10,000 条图像指令数据完成训练
- 至少完成 Projector-only 与 LoRA 两组实验
- 至少 3 个视觉任务有量化评测结果
- Gradio Demo 可运行
- GitHub 仓库一条命令复现推理

**理想成功标准（时间允许则追求）**

```text
50k 数据 + CLIP/SigLIP + 2-layer Projector
+ 1.5B~3B LLM + LoRA/QLoRA
+ 多个 Ablation + Gradio Demo
+ Training curves + Ablation tables + Qualitative examples
```

---

## 1. 技术栈与关键技术决策

### 1.1 技术栈总表

| 模块 | 选择 | 备注 |
| --- | --- | --- |
| Vision Encoder | CLIP ViT-B/32（256×256 → 49 token）为主，SigLIP-base / CLIP-B/16 为对照 | 冻结 |
| LLM | Qwen2.5-0.5B-Instruct（本地主力）；1.5B / 3B 为云端对照 | 冻结 / LoRA / QLoRA |
| Projector | 自己实现的 Linear → 2-layer MLP → 3-layer MLP | 唯一必训模块 |
| 微调 | LoRA / QLoRA (4-bit) | Hugging Face PEFT |
| 量化 | bitsandbytes 4-bit NF4 | 仅 QLoRA 实验用 |
| 框架 | PyTorch + Transformers + Accelerate | — |
| 数据 | COCO Caption + VQAv2 + 程序化生成（Counting/Attribute/Spatial） | 全部可离线获取 |
| 评测 | 自建 mini-CN-VQA + 自建评测脚本 | 不依赖外部 API |
| Demo | Gradio | — |
| 环境 | Python 3.10 + venv/conda（Windows 原生）+ 可选 WSL2 | 见 §2 |

### 1.2 关键决策与理由

**决策 1：视觉编码器用 CLIP ViT-B/32 起步，而不是 B/16 或 SigLIP-so400m**

- B/32 在 224×224 下产生 **49 个 patch token**（50 减去被丢弃的 CLS），B/16 为 196 个，SigLIP-so400m 为 729 个。
- visual token 数量直接影响序列长度与显存。49 个 token 让 6GB 显存和 0.5B LLM 的组合非常舒适，训练速度快 3–4 倍。
- 项目的第一阶段目标是**把链路打通**，不是追求视觉分辨率。等 Week 7 做 Ablation 时再对照 B/16（196 token）。

**决策 2：LLM 用 Qwen2.5-0.5B-Instruct 起步**

- 中文能力强，tokenizer 对中文友好（本项目要做中文视觉问答）。
- 0.5B 在 6GB 显存下可以 LoRA 甚至接近全量微调；1.5B 只能 QLoRA；3B 必须上云。
- 每个规模上限解锁一次新的实验档位，天然形成"模型规模"这一维 Ablation。

**决策 3：预计算并缓存视觉特征（本项目最重要的工程优化）**

因为视觉编码器**冻结**，同一张图片的视觉特征在整个训练过程中完全不变。因此在数据预处理阶段一次性计算：

```text
所有图片 → CLIP → [N, Dv] 特征 → 存到磁盘（.npy / 分片 .pt）
```

训练时 Dataset 只读特征、不读图片、不加载 CLIP。收益：

| 指标 | 不缓存 | 缓存后 |
| --- | --- | --- |
| 训练时显存占用 | +CLIP 权重与激活 | 0（CLIP 完全不加载） |
| 单步耗时 | 含 CLIP 前向 | 仅 Projector + LLM |
| 训练速度 | 基准 | **约 2–4× 提升** |
| 5 万张图存储 | 图片本体 GB 级 | 49×768 fp16 ≈ **75KB/图，合计约 3.8GB** |

这是把 80% 工作留在本地 RTX 4050 的关键技巧，也是简历上很有说服力的一笔工程细节。

**决策 4：不用 `inputs_embeds` + `model.generate()` 的常规写法**

decoder-only LLM 的 `generate()` 与 `inputs_embeds` 组合在多数版本中存在 `prepare_inputs_for_generation` 不传递嵌入的问题，会导致"训练正常但推理时完全忽略图片"这一**最隐蔽的 bug**。项目规定：

- **训练**：手动拼接 `inputs_embeds`，显式构造 `labels`（只对 answer 部分计算 loss，其余置 -100）。
- **推理**：第一版使用**自回归循环**（自己写 greedy / sampling），完全掌控拼接与 KV cache 逻辑。这条路径可控、可调试，也更符合"亲手完成核心连接"的项目目标。

**决策 5：先本地、后云端，且云端只跑"已经验证过的东西"**

任何脚本必须在本地用 100 / 1000 条子集跑通并通过验收，才允许上云。详见 §6 成本控制与止损机制。

---

## 2. 环境搭建（Week 1 第一件事，半天完成）

### 2.1 依赖清单

```text
# requirements.txt（版本随时间变化，安装时以"能装上"为准，不要死锁小版本）
torch>=2.4
torchvision
transformers>=4.45
peft>=0.13
accelerate>=1.0
bitsandbytes>=0.43        # 仅 QLoRA 需要；Windows 原生支持不稳定
datasets
pillow
numpy
pyyaml
tqdm
tensorboard
gradio>=4.0
scikit-learn
nltk                      # Caption 指标
rouge-score
pycocotools               # COCO 标注解析（Windows 需 VS Build Tools）
```

### 2.2 环境方案（Windows 用户的现实选择）

| 方案 | 优点 | 缺点 | 建议 |
| --- | --- | --- | --- |
| A. Windows 原生 venv + CUDA 版 PyTorch | 简单直接 | bitsandbytes 4-bit 在 Windows 原生易报错 | **默认方案**，先不用 bitsandbytes，LoRA 走 bf16 |
| B. WSL2 + Ubuntu | 与云端环境一致，bitsandbytes 无坑 | 首次配置需 1–2 小时，显存分配受 WDDM 影响 | **推荐**，若打算做 QLoRA 就走这条 |

**环境一致性铁律**：`requirements.txt` + `environment.yml` 二选一固定下来，本地与云端用**同一份**，云端 diff 不超过 1–2 个包。避免"本地能跑、云上报错，白烧一小时 GPU"。

### 2.3 环境自检脚本（必须先写这个）

```python
# scripts/check_env.py —— 上云前、本地开工前都跑一次
import torch, platform, sys
print("python      :", sys.version.split()[0], platform.platform())
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
    print("vram (GB)   :", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
    print("bf16 support:", torch.cuda.is_bf16_supported())
import transformers, peft
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__)
```

### 2.4 离线资源预下载（只做一次，省下大量等待时间）

```text
models/pretrained/
├── clip-vit-base-patch32/          # ~600MB
└── qwen2.5-0.5b-instruct/          # ~1GB
data/raw/
├── coco/train2017/                 # 图片
├── coco/val2017/
└── annotations/
    ├── captions_train2017.json
    └── instances_train2017.json    # 用于程序化生成 Counting/Attribute/Spatial
```

> **技巧**：`HF_HOME` 与 `HF_HUB_OFFLINE=1` 环境变量固定下来，训练脚本一律离线加载本地路径，杜绝"云端下载模型超时"这类浪费。

---

## 3. 数据集设计（Week 4 的核心，也是简历含金量最高的部分）

### 3.1 统一数据格式

```json
{
  "id": "coco_train2017_0000001234_counting_1",
  "image": "coco/train2017/0000001234.jpg",
  "task": "counting",
  "question": "图中一共有几个人？",
  "answer": "3 个人。",
  "meta": { "source": "coco_instances", "split": "train" }
}
```

单一 JSONL 文件，一条一行，字段固定。**不要中途改 schema**，否则后面所有脚本都要返工。

### 3.2 五类任务与构造方式

| 任务 | 示例问题 | 数据来源与构造方法 |
| --- | --- | --- |
| ① Caption | 描述这张图片。 | COCO Captions（英文），用固定模板 + 小型翻译/改写流程转中文；或直接保留英文做双语对照 |
| ② Object | 图中有什么物体？ | COCO instances 的类别集合 → 拼接成"图中有 A、B 和 C" |
| ③ Counting | 图中有几辆车？ | COCO instances 逐类别计数（过滤遮挡面积 < 阈值、crowd 标注） |
| ④ Attribute | 那辆汽车是什么颜色？ | 用 CLIP 对候选框做颜色分类打伪标签，或只保留 COCO 中颜色可判定的类别 |
| ⑤ Spatial | 猫在人的左边还是右边？ | 由 COCO bbox 的 x 中心比较程序化生成左右/上下关系 |

**关键权衡说明（必须诚实写进报告）**：

- ①②③⑤ 可以做到**高质量、零成本、完全程序化**。
- ④ Attribute 与开放词汇 Object 依赖伪标签，质量有噪声。**伪标注过程本身就是可以写进报告的工程内容**：用 CLIP 零样本分类对 bbox 打标签，再人工抽检 200 条估算噪声率。
- 需要明确记录：**答案分布必须检查**。若 counting 答案中"1"占比超过 50%，模型会退化成"永远答 1"。处理方式是按答案值做分层采样 + 按类别配平。

### 3.3 数据规模三档制（贯穿整个项目）

| 档位 | 规模 | 用途 | 要求 |
| --- | --- | --- | --- |
| Debug | 1,000 | 打通 pipeline、快速迭代代码 | 必须能在 15 分钟内跑完一个 epoch |
| Baseline | 10,000 | 第一版可用模型、本地主力实验 | 五类任务各 2,000 条 |
| Final | 50,000 | 最终模型、Ablation 主战场 | 五类任务配比合理，答案分布已配平 |

**数据划分纪律**：

- 固定划分：train / val / test = 90 / 5 / 5，**按 image_id 划分**（同一张图的多个问题必须落在同一个 split，否则数据泄漏，指标虚高）。
- 划分结果落盘成 `splits.json`，所有实验共用同一份，保证可比性。
- 永久保留一份 `eval/mini_cn_vqa_200.jsonl`（200 条人工校验的中文问答），作为**最终对外汇报的唯一指标来源**。

### 3.4 数据处理流水线

```text
data/raw/ (图片 + COCO 标注)
      │
      ▼  preprocess/build_instructions.py     → 生成 JSONL 指令数据
      │
      ▼  preprocess/cache_features.py         → CLIP 特征缓存 (关键优化)
      │     ├── features/train_shard_000.pt   # [num, 49, 768] fp16
      │     ├── features/val_shard_000.pt
      │     └── features/index.json           # image_id → (shard, offset)
      ▼
data/processed/
      ├── debug_1k.jsonl
      ├── baseline_10k.jsonl
      ├── final_50k.jsonl
      └── splits.json
```

### 3.5 数据工程验收清单（Week 4 周末必须全绿）

- [ ] 随机抽 20 条人工阅读，问题通顺、答案正确、无空字段
- [ ] 五个任务的样本数与配比统计表输出到 `report/data_stats.md`
- [ ] counting 类答案分布直方图（确认没有单值坍缩）
- [ ] 检查 image_id 在 train/val/test 之间**零交集**
- [ ] 检查问答重复率（完全相同 question+answer 的重复条目 < 1%）
- [ ] 一条命令即可从零重建全部数据：`python -m scripts.build_data --config configs/data.yaml`
- [ ] 特征缓存完整性校验（可加载、shape 正确、无 NaN）

---

## 4. 代码仓库结构

```text
Mini-VLM/
│
├── configs/
│   ├── data.yaml                 # 数据构建配置
│   ├── model_clip_b32_qwen05.yaml
│   ├── train_projector_only.yaml
│   ├── train_lora.yaml
│   └── train_qlora.yaml
│
├── datasets/
│   ├── __init__.py
│   ├── dataset.py                # JsonlVLDataset：读 JSONL + 读特征缓存
│   ├── preprocess.py             # 指令数据构建、特征缓存
│   └── collator.py               # 多模态 Collator：拼接 embeds + labels mask
│
├── models/
│   ├── vision_encoder.py         # VisionEncoder 封装（frozen，含 forward 校验）
│   ├── projector.py              # Linear / MLP2x / MLP3x
│   ├── multimodal_model.py       # 组装：拼接逻辑 + forward + generate
│   └── lora.py                   # LoRA / QLoRA 注入
│
├── training/
│   ├── trainer.py                # 训练循环、优化器分组、日志、checkpoint
│   └── losses.py                 # label 构造与 loss mask
│
├── evaluation/
│   ├── evaluate_vqa.py
│   ├── evaluate_caption.py
│   ├── evaluate_counting.py
│   ├── evaluate_hallucination.py
│   └── metrics.py
│
├── demo/
│   └── app.py                    # Gradio
│
├── scripts/
│   ├── check_env.py
│   ├── build_data.py
│   ├── overfit_test.py           # 100 样本过拟合测试（上云门禁）
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py              # 一条命令复现推理
│   └── cost_ledger.py            # 成本台账
│
├── tests/
│   ├── test_projector.py         # 参数量 / shape / 梯度 / 注入自检
│   └── test_collator.py          # loss mask 与占位符数量校验
│
├── report/
│   ├── report.md                 # 技术报告
│   ├── data_stats.md
│   ├── experiments.md            # 实验记录（每次云端运行一条）
│   └── figures/
│
├── requirements.txt
├── environment.yml
├── .gitignore                    # data/ models/ outputs/ 一律不入库
└── README.md
```

### 4.1 三个最关键的代码接口（提前定死，避免返工）

> 这三个接口一旦定下来，后续 5 周的代码都围绕它们展开。**先写接口，再写实现。**

**① Vision Encoder 封装**

```python
class VisionEncoder(nn.Module):
    """冻结的视觉编码器：image [B,3,H,W] -> features [B,N,Dv]"""
    def __init__(self, model_name_or_path: str, freeze: bool = True, drop_cls: bool = True):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name_or_path)
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()
        self.hidden_size = self.model.config.hidden_size          # ViT-B/32 -> 768
        self.patch_tokens = (self.model.config.image_size // self.model.config.patch_size) ** 2
        self.keep_cls = not drop_cls
        # 输出 token 数：B/32@224 -> 49 patch（丢 CLS）；保留 CLS 则为 50
        self.num_tokens = self.patch_tokens + (1 if self.keep_cls else 0)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values).last_hidden_state   # [B, 1+49, 768]
        return out if self.keep_cls else out[:, 1:, :]                  # [B, 49, 768]
```

> **CLS token 的处理必须显式决定并写进配置**：Visual Token 数量（49 还是 50）会直接影响占位符个数、序列长度和后续所有 Ablation 的可比性。项目默认 **丢弃 CLS，只保留 patch token**（与 LLaVA 的用法一致），并把 `drop_cls` 作为一个配置项固定下来。
>
> ⚠️ **已实测修正**：CLIP 的 `last_hidden_state` **确实包含 CLS**（共 50 个 token），所以 `drop_cls` 是真的切片操作，不是空操作。`forward` 里加了自动探测，只有当 token 数等于 `patch_tokens + 1` 时才切片，避免"以为不用丢"造成误删第一个 patch。详见 §4.2。

**② Projector（三种版本，一个类搞定，便于消融）**

```python
class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, depth: int = 2):
        super().__init__()
        if depth == 0:                      # Linear
            self.net = nn.Linear(in_dim, out_dim)
        elif depth == 2:                    # 2-layer MLP
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        elif depth == 3:                    # 3-layer MLP
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, out_dim))
        else:
            raise ValueError(depth)
    def forward(self, x):                   # [B,N,Dv] -> [B,N,Dl]
        return self.net(x)
```

**③ 多模态拼接（全项目的心脏）**

```python
def build_multimodal_inputs(visual_embeds, input_ids, labels, image_token_id, text_embeds):
    """
    visual_embeds : [B, N, Dl]  Projector 输出
    input_ids     : [B, L]      含 N 个 image_token_id 占位符
    labels        : [B, L]      answer 部分为 token id，其余为 -100
    """
    # 1. 先取文本嵌入（此时 image_token 位置是随机初始化的 embedding，会被覆盖）
    inputs_embeds = text_embeds.clone()
    # 2. 逐个样本把 image_token 位置替换成 visual_embeds（注意 batch 内 N 可能不同）
    for b in range(input_ids.size(0)):
        pos = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        assert pos.numel() == visual_embeds.size(1), "占位符数量与视觉 token 数不匹配"
        inputs_embeds[b, pos] = visual_embeds[b].to(inputs_embeds.dtype)
    return inputs_embeds, labels
```

**必须写进 README 的三个 off-by-one 陷阱**：

1. 手动拼接 embeds 时不要再让 tokenizer 自动加 BOS / 特殊 token，否则位置错位。
2. `labels` 只在 answer 区间有值，question、image token、pad 全部 `-100`。
3. 推理时如果走 `generate()`，必须验证"换一张图答案会变"（见 §7.3 必设的三个基线中的"无视觉基线"）。

### 4.2 占位符 Token 与 CLS：本地实测结论（**已实证，不要再按推测改**）

> 本节内容已由 `scripts/verify_offline.py` 在本机实测确认，原始推测已被推翻。
> 复现命令：`python scripts/verify_offline.py`

**实测环境**：Python 3.12.4 / torch 2.6.0+cu124 / transformers 5.17.0 / RTX 4050 Laptop 6GB

| # | 原推测 | 实测结果 | 处理 |
| --- | --- | --- | --- |
| 1 | 纯文本 Qwen2.5 的 tokenizer **没有** `<\|image_pad\|>`，需自行注册 + resize | **词表里已经有**，id = **151655**，是单一 token；权重表 **151936** 行，**不越界** | **不要** `add_special_tokens`，**不要** `resize_token_embeddings` |
| 2 | CLIP 的 `last_hidden_state` 不含 CLS，`drop_cls` 是空操作 | 形状 **[B, 50, 768]**，第 0 个**就是 CLS**，丢弃后为 49 个 patch | `drop_cls` 必须真的切片 `out[:, 1:, :]`（已修复 `vision_encoder.py`） |
| 3 | 可用 `ls[:,0,:] ≈ pooler_output` 判断 CLS | **CLIP 的 pooler_output 不是第 0 个 hidden state**（pooler 是另一层投影），该判定恒为 False | 判断 CLS 只能靠 **token 数**，不能靠 pooler 对比 |

**关键数字（直接抄进配置，别猜）**

```text
CLIP ViT-B/32 @224      : image_size=224, patch_size=32  -> 49 patch + 1 CLS = 50
                          丢弃 CLS 后 visual token 数 N = 49, hidden = 768
Qwen2.5-0.5B-Instruct   : hidden_size = 896, num_layers = 24, heads = 14, kv_heads = 2
                          vocab_size = 151643, len(tokenizer) = 151665
                          embedding 权重行数 = 151936  （比 tokenizer 多 271 行，属于预留，勿新增 token）
                          tie_word_embeddings = True, bos_token_id = None, max_pos = 32768
占位符                  : <|image_pad|> = 151655；同族还有 <|vision_start|>=151652,
                          <|vision_end|>=151653, <|vision_pad|>=151654, <|video_pad|>=151656
```

**由此确定的两条代码纪律**

1. **Projector 的 `out_dim` 必须等于 896**（Qwen2.5-0.5B 的 `hidden_size`），这是硬约束，写进 `configs/model_clip_b32_qwen05.yaml`。
2. **`resize_token_embeddings` 一律不要调用**。一旦 resize，`tie_word_embeddings=True` 的输入/输出嵌入会被重新绑定并引入未训练参数，"训练正常但输出退化"极难排查。

**Week 2/3 前置检查清单（可执行）**

- [ ] `python scripts/verify_offline.py` 输出 `结论：离线验收通过（0 项警告）`
- [ ] `<|image_pad|>` 的 id 确认 = 151655，且 `tokenize` 后只产生 1 个 id
- [ ] 占位符 id < embedding 权重行数（151655 < 151936）
- [ ] `VisionEncoder(...)(torch.randn(2,3,224,224)).shape == (2, 49, 768)`
- [ ] `VisionEncoder(..., drop_cls=False)` 输出 `(2, 50, 768)`
- [ ] `Qwen2.5-0.5B` 的 `hidden_size`（896）与 Projector `out_dim` 一致
- [ ] ⚠️ **不要**使用 `Qwen2.5-VL` 的 processor / image processor —— 本项目的 LLM 是纯文本 Qwen2，图像预处理必须由 CLIP 的 processor 负责

---

## 5. 八周路线图（逐周可执行版）

> 约定：**每周日做验收 + 写周记**；验收不通过不得进入下一周的新内容（但可以并行修 bug）。

### 阶段总览

| 周 | 主题 | 本地/云端 | 花费 | 关键产出 |
| --- | --- | --- | --- | --- |
| W1 | 真正理解 VLM + 环境 | 本地 | ¥0 | 架构图笔记 + 环境自检通过 |
| W2 | Vision Encoder | 本地 | ¥0 | `vision_encoder.py` + 特征 shape 报告 |
| W3 | 自己实现 Projector | 本地 | ¥0 | 三种 Projector + 参数量表 + 注入自检 |
| W4 | 数据工程 | 本地 | ¥0 | 1k/10k/50k 数据集 + 数据卡片 |
| W5 | 组装 Mini-VLM + Baseline | 本地 + 首次上云 | ¥100–150 | **可用的 v0 模型** + 10k 训练结果 |
| W6 | LoRA / QLoRA | 云端为主 | ¥200–300 | 三个策略对照实验 |
| W7 | 正式实验与消融 | 云端为主 | ¥200–300 | 4 张表 + 训练曲线 |
| W8 | Demo + GitHub + 报告 | 本地 | ¥0–100 | 全部交付物 |

---

### Week 1：真正理解 VLM（不训练任何模型）

**本周唯一要回答的问题：一张图片究竟是怎么进入 LLM 的？**

**学习任务**

1. 复习 ViT：`Image → Patch → Patch Embedding → Transformer → Visual Feature`
2. CLIP 为什么能做视觉编码器：图像/文本双塔 + 对比学习目标（只理解到"能提供语义对齐好的视觉特征"这一层即可）
3. LLM 侧只需明确：`Token → Embedding → Transformer → Hidden State → Next Token`，以及 $P(x_t \mid x_{<t})$
4. **Projector 是本项目最重要的新概念**：

$$V \in \mathbb{R}^{N \times 768}, \quad L \in \mathbb{R}^{N \times 896}, \quad P: \mathbb{R}^{768} \to \mathbb{R}^{896}, \quad P(V) = W_2 \,\sigma(W_1 V)$$

**必读材料（只读该读的部分）**

| 材料 | 读什么 | 时长 |
| --- | --- | --- |
| LLaVA: Visual Instruction Tuning (arXiv 2304.08485) | Abstract、Figure 1、Model Architecture、Visual Instruction Tuning 小节 | 2h |
| HF Multimodal Chat Templates 文档 | Processor → Chat Template → Image+Text → Model 的数据流 | 1h |
| PEFT 文档 (LoRA / trainable parameters) | LoRA 基本用法与可训练参数统计 | 1h |
| PEFT Quantization 指南 | 4-bit + LoRA 的组合方式，不深究数学 | 0.5h |
| Qwen2.5-VL 官方文档 | 仅作**参考实现**阅读，看它如何组织 processor 与 model | 1h |

**动手任务**

- [ ] 画出下面这张图，并对**每一个 Tensor 标注 shape**（含 batch 维的示例数值）：

```text
Image [B,3,224,224]
  ↓ Vision Encoder (frozen)
Visual Features [B, 49, 768]         # 50 = 49 patch + 1 CLS，项目默认丢弃 CLS → 49
  ↓ Projector (trainable)
Visual Embeddings [B, 49, 896]
  ↓ concat with text embeds
Inputs Embeds [B, 49+L_text, 896]
  ↓ LLM
Logits [B, 49+L_text, vocab]
```

- [✅️] 搭建环境，跑通 `scripts/check_env.py`，记录 GPU 型号、显存、bf16 支持
- [✅️] 预下载 CLIP 与 Qwen2.5-0.5B 权重到本地
- [✅️] 输出一份 `notes/week1_vlm_notes.md`

**周末验收（口头答不出就不算过）**

1. 为什么 Vision Encoder 不能直接接 LLM？（提示：维度、语义空间、序列长度三个层面）
    1.维度不匹配 必须把视觉输出特征的维度转化为llm接受向量维度
    2.语义空间不匹配 CLIP的是对比学习得到的视觉-文本对齐空间，不是LLM的词嵌入空间
    3.序列长度与接口不匹配 视觉特征是连续向量不是词表里的离散token id
2. Projector 在解决什么问题？如果不训练它会发生什么？
   Projector解决两件事：1）维度变换2）语义对齐
   如果不训练Projector则视觉嵌入将是随机投影，与llm的词嵌入分布不匹配。
3. 为什么图像最终可以表示成"一系列 Token"？
    按ViT的做法，将一张图片切分为多个patch，每个patch经过Transformer后输出一个上下文相关的向量
4. VLM 与普通 LLM 的最大区别是什么？
    具有视觉模态
5. 50 个 visual token 会挤占多少上下文预算？对训练成本意味着什么？

---

### Week 2：Vision Encoder（仍然是 0 训练）

**目标**：把 CLIP 视觉塔用熟，产出可复用的封装。

**动手任务**

- [ ] 实现 `models/vision_encoder.py`：`image [B,3,H,W] → features [B,N,D]`
- [ ] 取 8 张图做前向，打印并记录：`N = ?`、`D = ?`、CLS token 是什么、patch token 是什么
- [ ] **实验 A**：对比 `output_hidden_states` 下不同层的特征（最后一层 vs 倒数第二层），观察差异
- [ ] **实验 B**：零样本验证编码器有效性 —— 用 CLIP 做图文匹配（`logits_per_image`），确认 top-1 命中合理
- [ ] **实验 C**：数据增强敏感性 —— 同一张图在 resize/crop 前后特征余弦相似度是多少
- [ ] 把 1000 张图的特征缓存下来（为 Week 4 铺路），测量：缓存文件大小、单图缓存耗时、总耗时

**交付物**

```text
models/vision_encoder.py
report/figures/w2_feature_shapes.md     # N/D 记录
report/figures/w2_clip_zeroshot.md      # 零样本匹配结果（附 2-3 个成功/失败例子）
data/processed/features/probe_1k.pt     # 特征缓存与大小统计
```

**验收标准**

- [ ] 能准确说出 `N` 与输入分辨率、patch size 的关系式
- [ ] 冻结生效验证：`sum(p.requires_grad for p in encoder.parameters()) == 0`
- [ ] 单图特征提取耗时 < 20ms（GPU），缓存 1000 张 < 30 秒

> 本周不训练。目标是**把视觉侧完全搞懂并且能稳定输出**。

---

### Week 3：自己实现 Projector（本项目真正的"自己写"模块）

**第一版 Linear**：`nn.Linear(768, 896)`
**第二版 2-layer MLP**：`768 → 2048 → GELU → 896`
**第三版 3-layer MLP**：`768 → 2048 → GELU → 2048 → GELU → 896`

**动手任务**

- [x] 实现 `models/projector.py`（`Projector(in_dim, out_dim, hidden, depth)`，depth ∈ {0,2,3}）
- [x] **参数量实测表**（⚠️ 下行为**实测值**；原稿手算有误，以本表为准）：

| Projector | 结构 | 参数量（实测） |
| --- | --- | --- |
| Linear | 768→896 | **689,024**（0.69M） |
| 2-layer MLP | 768→2048→896 | **3,410,816**（3.41M） |
| 3-layer MLP | 768→2048→2048→896 | **7,607,168**（7.61M） |

> 复现：`python models/projector.py`（直接运行即打印参数量表与输出形状）
> 原稿写的 3,410,944 与 7,606,272 分别差 128 和 896 —— **偏置项手算极易出错，一律以代码实测为准**。

- [ ] **数值自检**：随机输入 `[2, 49, 768]`，确认输出 `[2, 49, 896]`、dtype 一致、无 NaN
- [ ] **注入自检（关键）**：把 Projector 输出替换进 LLM 的 embeds 位置，验证：
  - 换一张图，LLM 的输出 logits **确实发生变化且变化显著**
  - 把 visual embeds 置零，输出应退化为"纯文本 LLM"的行为
  - 这一步是 §5.3 那个"最隐蔽 bug"的第一道防线
- [ ] **梯度自检**：反向传播后确认 `projector` 有梯度、`vision_encoder` 与 `llm` 无梯度
- [ ] **Unit test**：为上述四类自检写成 `tests/test_projector.py`，纳入上云前门禁

**本周产出的小 Ablation 预告**

| Projector | 参数量 | 最终效果 |
| --- | --- | --- |
| Linear | 待测 | 待测 |
| 2-layer MLP | 待测 | 待测 |
| 3-layer MLP | 待测 | 待测 |

---

### Week 4：数据工程（最需要认真的一周）

> 真正的 AI 实习不只是"会调用模型"，更重要的是**会处理数据**。

**任务分解（按天推进）**

| 天 | 任务 |
| --- | --- |
| 周一 | 确定 JSONL schema；下载 COCO val2017 + train2017 子集 + 标注文件 |
| 周二 | 实现 `build_instructions.py`：Caption + Object 两类 |
| 周三 | 实现 Counting + Spatial（基于 instances bbox，含遮挡/面积过滤） |
| 周四 | 实现 Attribute（CLIP 伪标注 + 抽检 200 条估噪声率） |
| 周五 | 生成 1k / 10k / 50k 三档 + 固定 splits.json + 答案分布配平 |
| 周六 | 实现 `dataset.py` + `collator.py`；跑通 DataLoader（打印真实 batch shape） |
| 周日 | 特征缓存全量生成 + 数据卡片 + 验收清单逐项核对 |

**`collator.py` 的核心职责（写清楚，别偷懒）**

```text
输入：list of {features: [49,768], question: str, answer: str}
  1. question/answer 套 Qwen2.5 chat template，插入 49 个 <|image_pad|> 占位符
  2. tokenize 到 max_len（建议 768），右侧 pad
  3. 构造 labels：仅 answer 区间保留 token id，其余 -100
  4. 一次前向拿到 text embeddings，再执行 §4.1 的替换逻辑
  5. 返回 {inputs_embeds, attention_mask, labels, pixel_values(可选), task_ids}
```

**数据卡片（`report/data_stats.md`）模板**

| 任务 | 1k 档 | 10k 档 | 50k 档 | 平均答案长度 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Caption | | | | | |
| Object | | | | | |
| Counting | | | | | 答案分布见直方图 |
| Attribute | | | | | 伪标签噪声率 ≈ ?% |
| Spatial | | | | | |
| **合计** | 1,000 | 10,000 | 50,000 | | |

**验收标准**

- [ ] §3.5 的七条数据工程验收清单全部打勾
- [ ] DataLoader 一个 batch 的真实 shape 打印并记录（`inputs_embeds: [B, L, 896]`）
- [ ] 手动检查 10 条样本的 `labels` 是否正确（question 部分必须是 -100）
- [ ] 数据重建一条命令可复现

---

### Week 5：组装 Mini-VLM + 第一个 Baseline

**本周是整个项目的核心周。核心要解决的问题是：Visual Tokens 如何插入 Text Tokens？**

```text
用户输入：<image> 这张图片里有什么？

实际张量：
[V1][V2]...[V49][这][张][图][片][里][有][什][么]
        ↓ 同一次 Transformer 前向
                     Answer
```

**Baseline 配置（Experiment A）**

```text
Vision Encoder  ❄️ Frozen
Projector       🔥 Trainable
LLM             ❄️ Frozen
```

**为什么先做这个？** 因为它回答一个根本问题：**仅仅训练 Projector，能不能让 LLM 用上视觉信息？** 这是后续所有实验的参照点。

**任务顺序（严格按序，不要跳）**

1. `models/multimodal_model.py`：组装 + forward + 自回归 generate
2. `training/trainer.py`：优化器分组、lr schedule、grad clip、日志、checkpoint
3. **`scripts/overfit_test.py`：100 样本过拟合测试**（见 §6.2）
4. 本地 1k 数据跑 1 epoch，确认 loss 正常下降
5. 本地 10k 数据跑 Baseline（RTX 4050，预期 1.5–3 小时）
6. 首次上云：10k 数据 Baseline 正式训练（预算 ¥100–150）
7. 评测：VQA 准确率 / Caption 指标 / Counting 准确率

**推荐超参（Week 5 Baseline）**

| 参数 | 值 | 说明 |
| --- | --- | --- |
| projector lr | 1e-3 | Projector 是随机初始化，需要较大 lr |
| batch size | 8（梯度累积 4 → 等效 32） | 6GB 显存下的稳妥起点 |
| optimizer | AdamW (weight_decay 0.0) | Projector 无需 weight decay |
| schedule | cosine, warmup 100 步 | — |
| precision | bf16 | 4050 支持 |
| max_seq_len | 768 | 49 visual + ~150 text |
| epochs | 1–2 | 10k 数据先看趋势 |
| grad_clip | 1.0 | — |
| seed | 42 | 所有实验固定 |

> ⚠️ **重要**：Projector 的 lr（1e-3 量级）与 LoRA 的 lr（2e-4 量级）**不同**。Week 6 起必须对两组参数分别设置 lr，否则训练必然失败。

**周五/周六：首次上云操作手册**

```bash
# 上云前门禁（本地全部通过才允许租卡）
python scripts/check_env.py
python -m pytest tests/ -q
python scripts/overfit_test.py --config configs/train_projector_only.yaml --n 100
python scripts/train.py --config configs/train_projector_only.yaml --data data/processed/debug_1k.jsonl --dry-run

# 云端执行顺序（从租卡到关机，控制在 2 小时内）
1. 同步代码与数据（git clone + rsync 特征缓存）
2. check_env.py
3. 1k 数据跑 50 步 smoke test          # 3 分钟
4. 启动正式训练（nohup + log 落盘）
5. 训练结束 → 立即下载 checkpoint 与 log
6. 评测 → 保存结果
7. 【关机】确认实例已停止计费
```

**验收标准**

- [ ] 100 样本过拟合测试通过（loss < 0.1 或 train token 准确率 > 95%）
- [ ] 换图输出会变（注入验证在真实模型中再次通过）
- [ ] Baseline 在 mini-CN-VQA-200 上有非平凡的准确率（显著高于"永答最常见答案"的多数类基线）
- [ ] 至少 3 个任务有量化结果
- [ ] 云端花费记账完成

---

### Week 6：LoRA / QLoRA

**核心公式**

$$W' = W + BA, \quad W \text{ 冻结}, \; A, B \text{ 训练}$$

```text
LLM       ❄️ Frozen
LoRA      🔥 Train
Projector 🔥 Train
```

**三个实验（对照必须严格控制变量）**

| 实验 | Vision Encoder | Projector | LLM | 目的 |
| --- | --- | --- | --- | --- |
| A | ❄️ Frozen | 🔥 Train | ❄️ Frozen | Baseline：视觉对齐是否足够 |
| B | ❄️ Frozen | 🔥 Train | 🔥 LoRA (bf16) | 指令跟随能力提升多少 |
| C | ❄️ Frozen | 🔥 Train | 🔥 QLoRA (4-bit) | 显存换效果的性价比 |

**LoRA 推荐初始超参**

| 参数 | 值 |
| --- | --- |
| target_modules | `q_proj, k_proj, v_proj, o_proj`（先只做 attention） |
| r / alpha / dropout | 16 / 32 / 0.05 |
| lora lr | 2e-4 |
| projector lr | 1e-3 |
| 其他 | 与 Week 5 保持一致 |

**必须记录的对照指标**

| 指标 | A | B | C |
| --- | --- | --- | --- |
| GPU 显存峰值 (GB) | | | |
| 可训练参数量 (M) / 占比 | | | |
| 单步耗时 (s/step) | | | |
| 总训练时长 | | | |
| 训练成本 (¥) | | | |
| val loss 曲线 | | | |
| VQA 准确率 | | | |
| Counting 准确率 | | | |
| 生成质量（人工 20 例打分 1–5） | | | |

**QLoRA 注意事项**

- 4-bit 量化后**只能训练 LoRA 适配器**，不能全量微调（否则质量崩塌）
- 若 Windows 原生 bitsandbytes 报错：切 WSL2，或降级到 LoRA bf16 完成 B 实验，C 实验移到云端
- 量化推理时注意 `bnb_4bit_compute_dtype=torch.bfloat16`，否则速度异常

**验收标准**

- [ ] A / B / C 三组结果在同一测试集上可比
- [ ] 每组训练曲线都保存到 `report/figures/`
- [ ] exp C 若因环境失败，必须在报告中如实记录失败原因（**失败的实验记录同样是合格的工程产出**）

---

### Week 7：正式实验与消融

**这一周不学新东西，只做实验。像研究生一样工作。**

#### 实验一：Projector 结构

| Projector | 参数量 | VQA | Caption(BLEU-4) | Counting |
| --- | --- | --- | --- | --- |
| Linear | | | | |
| 2-layer MLP | | | | |
| 3-layer MLP | | | | |

#### 实验二：视觉 Token 数量

| 设置 | N_visual | 序列长度 | VQA | 单步耗时 |
| --- | --- | --- | --- | --- |
| CLIP-B/32 @224 | 49 | | | |
| CLIP-B/32 @256 | 64 | | | |
| CLIP-B/16 @224 | 196 | | | |
| 49 → 池化到 16 | 16 | | | |

> 研究 $N_{visual}$ 对性能与成本的影响。这一维实验在文献中讨论很多，是很好的报告素材。

#### 实验三：LLM 是否微调

`Frozen vs LoRA vs QLoRA`（复用 Week 6 结果，补充不同 r 值：8 / 16 / 32）

#### 实验四：数据规模

| Data | VQA | Caption | Counting |
| --- | --- | --- | --- |
| 1k | | | |
| 10k | | | |
| 50k | | | |

> 画一张 $Dataset\ Size \rightarrow Performance$ 曲线。**这条曲线即使不完美也是报告的核心价值之一**（是否出现收益递减、哪个任务最先饱和）。

#### 补充实验（时间允许）

- 学习率敏感性：projector lr ∈ {3e-4, 1e-3, 3e-3}
- 幻觉率评测：提问"图中是否有 X"（X 为图中不存在的物体），统计误报率

**最终至少产出 4 张表**

**表 1：模型规模**

| Model | Parameters | Trainable | Trainable % |
| --- | ---: | ---: | ---: |
| Baseline (Projector-only) | | | |
| LoRA | | | |
| QLoRA | | | |

**表 2：Projector**

| Projector | VQA | Caption |
| --- | --: | --: |
| Linear | | |
| MLP | | |

**表 3：数据量**

| Data | VQA |
| --- | --: |
| 1k | |
| 10k | |
| 50k | |

**表 4：训练策略**

| Strategy | Test |
| --- | --: |
| Projector only | |
| LoRA | |
| QLoRA | |

**验收标准**

- [ ] 每张表都有对应的实验记录（`report/experiments.md` 中一行一条：日期、config、数据、耗时、花费、指标、结论）
- [ ] 所有曲线图导出到 `report/figures/`
- [ ] 至少 3 个定性案例（成功 2 个 + 失败 1 个），失败案例要给出原因分析

---

### Week 8：Demo + GitHub + 报告

**Demo（Gradio）**

```text
┌──────────────────────────┐
│        Mini-VLM           │
├──────────────────────────┤
│       [Image]            │
├──────────────────────────┤
│ Question:                │
│ 图中有几个人？            │
├──────────────────────────┤
│ Answer:                  │
│ 图中有两个人。            │
└──────────────────────────┘
```

Demo 必须包含的功能：

- [ ] 图片上传 + 文本提问 → 回答
- [ ] 预设问题按钮（描述图片 / 图中有几个 X / 什么颜色 / 左右关系）
- [ ] 可调生成参数（max_new_tokens、temperature）与显示推理耗时
- [ ] 显示当前使用的模型与训练策略（方便展示 Ablation）
- [ ] 失败案例提示（当答案明显不合理时给出免责说明）

**一键复现（README 第一条命令）**

```bash
# 目标：clone 之后一条命令就能跑推理
python scripts/inference.py \
  --model models/pretrained/qwen2.5-0.5b-instruct \
  --vision models/pretrained/clip-vit-base-patch32 \
  --projector outputs/baseline_10k/projector.pt \
  --image examples/demo.jpg \
  --question "图中有几个人？"
```

**README 必备章节**

1. 项目简介 + 架构图（ASCII 或图片）
2. 环境安装（含 Windows / WSL2 两种路径）
3. 数据构建（一条命令）
4. 训练（三条命令对应 A/B/C）
5. 评测（一条命令产出表格）
6. Demo 启动
7. 实验结果表格 + 训练曲线
8. 已知问题与失败案例分析
9. 成本台账
10. 参考工作（LLaVA、PEFT、Qwen2.5-VL 等）

**技术报告结构（8–15 页）**

```text
1. 引言与动机
2. 相关工作（LLaVA / BLIP-2 / Qwen-VL 的连接方式对比）
3. 方法
   3.1 架构总览与张量流
   3.2 Vision Encoder 的选择与冻结策略
   3.3 Projector 设计与参数量分析
   3.4 多模态 Token 注入与 Loss Mask
   3.5 视觉特征缓存（工程优化）
4. 数据工程
   4.1 五类任务的构造方法
   4.2 数据规模与配比
   4.3 伪标签噪声分析与答案分布配平
5. 实验设置（硬件、超参、评测协议）
6. 实验结果
   6.1 主结果
   6.2 Projector 消融
   6.3 视觉 Token 数量消融
   6.4 参数高效微调策略对照
   6.5 数据规模曲线
7. 定性分析（成功案例 + 失败案例 + 幻觉现象）
8. 成本与工程实践（本地/云端分工、止损机制）
9. 局限与未来工作（更高分辨率、更多数据、更强 LLM、DPO 对齐等）
10. 附录（完整超参、复现命令）
```

**验收标准**

- [ ] Demo 上传图片可正常问答
- [ ] README 在干净环境中按步骤可复现
- [ ] 报告完成，含至少 4 张表 + 3 张图
- [ ] 简历条目定稿

---

## 6. 成本控制与止损机制

### 6.1 预算分配

| 项目 | 预算 | 实际（滚动填写） |
| --- | ---: | ---: |
| 本地 RTX 4050 | ¥0 | ¥0 |
| 数据集 | ¥0 | ¥0 |
| Hugging Face / GitHub / Python / PyTorch | ¥0 | ¥0 |
| 云 GPU | ¥600 | |
| 云盘 / 临时存储 | ¥100 | |
| 最终 Demo 部署 | ¥100 | |
| 机动预算 | ¥200 | |
| **合计（上限）** | **¥1000** | |

### 6.2 阶段花费计划

| 阶段 | 周 | 预算 | 用途 |
| --- | --- | ---: | --- |
| 阶段 1 | W1–W4 | ¥0 | 全部本地：环境、数据、开发、调试 |
| 阶段 2 | W5 | ¥100–150 | 首次正式训练（10k，Baseline） |
| 阶段 3 | W6 | ¥200–300 | LoRA / QLoRA 对照 |
| 阶段 4 | W7 | ¥200–300 | 正式消融实验（含 50k） |
| 阶段 5 | W8 | ¥0–100 | Demo 部署（GPU 需求极低） |
| **理想总计** | | **¥500–800** | 1000 是上限，不是目标 |

### 6.3 单次训练的真实成本估算（为什么这笔钱很可能花不完）

以 Qwen2.5-0.5B + 49 visual token + 特征缓存为例，单样本序列约 250–400 token：

| 场景 | 数据 | 卡 | 预估时长 | 预估花费 |
| --- | --- | --- | --- | --- |
| Baseline | 10k | RTX 4090 (¥2/h) | 0.5–1.5h | ¥1–3 |
| LoRA | 50k | RTX 4090 | 2–4h | ¥4–8 |
| QLoRA | 50k (1.5B) | RTX 4090 | 3–5h | ¥6–10 |
| 全量微调 | 50k (0.5B) | A100 40G (¥8/h) | 3–6h | ¥24–48 |

**结论：本项目的真实瓶颈是时间与实验设计，不是钱。** 1000 元预算的真正作用是"允许你把实验做扎实"（多跑几个 seed、多跑几组消融、升级到 1.5B/3B），而不是"必须花掉"。

**建议**：把省下的钱用于最后一次"强化训练"——3B QLoRA + 50k 数据，单次约 ¥30–60，可以显著提升 Demo 观感。

### 6.4 止损机制（本次项目最重要的纪律）

**绝对不能出现**：`效果不好 → 租 GPU → 重训 → 又不好 → 再租 GPU`。

**上云门禁（Checklist，必须全部通过）**

```text
□ 代码检查（review 自己的 diff，确认没有 debug 用的临时修改）
□ 配置检查（config 里的路径、数据量、epoch 数、lr 全部核对一遍）
□ 100 samples overfit test 通过     ← 硬门禁
□ 1000 samples 跑通，loss 正常下降  ← 硬门禁
□ dry-run 打印出正确的 batch shape 与可训练参数统计
□ 预估本次花费并记录
```

**100 样本过拟合测试的量化标准**

| 检查项 | 通过标准 |
| --- | --- |
| train loss | 从初始值下降到 < 0.1（或下降到初始值的 10% 以下） |
| answer token 准确率 | > 95% |
| 换图输出变化 | 同一问题换图，答案必须不同 |
| 梯度检查 | 仅 Projector / LoRA 有梯度，冻结部分梯度为 None |
| 生成检查 | 至少能完整吐出训练集里的标准答案（允许表述差异） |

> **如果模型连 100 个样本都无法接近 100% 拟合，禁止上云 GPU。** 这是整个项目成本控制最重要的一条规则。

**其他止损规则**

1. 任何本地可跑的实验（1k 数据、0.5B 模型、短序列），一律先在本地跑通，云端只做"规模化复现"。
2. 每次云端训练前先跑 3 分钟 smoke test（50 步），确认 loss 在下降、显存没有爆炸。
3. 未通过门禁就上云，导致浪费 > ¥50，则该周暂停云端实验，回到本地排查。
4. 单次云端训练任务时长上限 **4 小时**；超过则必须设置 checkpoint 保存点并拆分成多次。
5. 单周累计花费超过该周预算 120%，必须停下来复盘并调整后续实验计划。
6. **训练日志与 checkpoint 必须在实例关闭前下载到本地**（`nohup` 输出 + `scp`/`rsync`），绝不依赖云端磁盘。
7. 所有云端实例使用按量计费 + 用完即停；训练脚本末尾打印总耗时，并在日志第一行记录开始时间。

**成本台账（每次云端运行必须记录）**

| 日期 | 实验名 | 卡型 | 时长 | 花费 | 结果 | 是否达到预期 |
| --- | --- | --- | ---: | ---: | --- | --- |
| | | | | | | |

---

## 7. 评测体系

### 7.1 三个必做任务

| 任务 | 指标 | 数据 |
| --- | --- | --- |
| 图像描述 Caption | BLEU-4、ROUGE-L、CIDEr（可选） | COCO val 子集（500 张） |
| 视觉问答 VQA | 精确匹配 + 关键词包含率 | 自建 mini-CN-VQA-200 |
| 目标计数 Counting | 数值准确率 / ±1 容忍准确率 | 程序化生成的测试集（500 条） |
| 补充：幻觉率 | 不存在物体的误报率 | 对抗测试集（200 条） |

### 7.2 指标定义（写清楚，避免自欺）

```text
exact_match   : 归一化（去标点/空格/大小写）后完全一致
contains      : 标准答案的关键内容出现在生成结果中（关键词表定义）
counting_acc  : 提取生成文本中的数字，与真值比较；同时报告 ±1 容忍准确率
hallucination : 提问"图中是否有 X"（X 不存在），回答"有"即为一次幻觉
```

### 7.3 必设的三个基线（没有基线的指标没有意义）

| 基线 | 说明 |
| --- | --- |
| 多数类基线 | 永远回答训练集中最高频的答案 |
| 无视觉基线 | 只给问题不给图片，测模型是否在"看图说话" |
| 训练集抄写基线 | 检索训练集中最相似问题的答案 |

> 只有当 Mini-VLM 明显超过这三个基线时，才可以说"模型真的用上了视觉信息"。这是报告中最有说服力的一段。

### 7.4 评测协议纪律

- 所有实验**必须**在同一份测试集、同一套 prompt、同一组生成参数下评测
- 生成参数固定：`greedy / temperature=0`（保证可复现）；另附一组 `temperature=0.7` 的样例用于展示
- 每个指标附样本量，小样本差异不做强结论（避免把噪声当提升）
- 报告结论时同时给出**代价**（参数量、显存、耗时、花费），避免只看单一指标

---

## 8. 风险登记表

| # | 风险 | 概率 | 影响 | 应对措施 |
| --- | --- | --- | --- | --- |
| R1 | bitsandbytes 在 Windows 原生不可用，QLoRA 失败 | 高 | 中 | 改用 WSL2；或 QLoRA 实验直接放云端；报告如实记录 |
| R2 | 6GB 显存不足（1.5B 及以上） | 高 | 中 | 0.5B + bf16 LoRA 本地跑；1.5B/3B 放云端；开启 gradient checkpointing |
| R3 | **推理时图片被忽略（训练正常但换图不变）** | 中 | 高 | 从 Week 3 起强制做"注入自检"；写进 unit test；上云门禁必查 |
| R4 | loss 不下降 / 变为 NaN | 中 | 高 | 先跑 100 样本过拟合；降 lr；检查 labels 的 -100 是否设置正确；检查 lr 分组 |
| R5 | 数据答案分布坍缩（如 counting 全是 1） | 中 | 中 | 生成阶段做配平；训练前必出分布直方图 |
| R6 | 数据泄漏（同图跨 split） | 中 | 高 | 按 image_id 划分；训练前脚本硬校验 |
| R7 | 云端浪费（跑完忘关机） | 中 | 中 | 训练脚本末尾提示；设置闹钟；遵守 4 小时上限 |
| R8 | tokenizer / chat template 与 Qwen 版本不匹配 | 低 | 中 | 固定 transformers 版本；打印一条完整构造样本人工核对 |
| R9 | CLIP 特征缓存与图片不对应（索引错位） | 低 | 高 | 缓存索引落盘 + 抽样 20 条做"图 ↔ 特征"一致性校验 |
| R10 | 时间超支（Week 4/5 拖延） | 中 | 高 | 严格执行 1k→10k→50k 的档位制；先保最低成功标准，再追理想标准 |
| R11 | 追求 SOTA 导致无止境调参 | 中 | 中 | 明确"这是工程项目不是刷榜"；每个实验最多调 2 轮 |

---

## 9. 成功标准与简历表达

### 9.1 最终简历条目（模板，数字待实验后填入）

> **Mini-VLM: Lightweight Vision-Language Model**（个人项目 / 8 周）
>
> 基于预训练 CLIP Vision Encoder 与 Qwen2.5 系列 LLM 构建视觉语言模型，自主实现视觉特征 Projector 与多模态 Token 融合模块（含 loss mask 与自回归推理逻辑）；通过离线视觉特征缓存将训练速度提升约 X 倍，使 80% 训练工作在单张 6GB 消费级 GPU 上完成；构建 N 万级中文图像指令数据集（覆盖描述、识别、计数、属性、空间五类任务），采用 Projector-only 与 LoRA 两阶段策略完成视觉-语言对齐；系统研究 Projector 结构、视觉 Token 数量、数据规模及参数高效微调策略对性能的影响（X 组消融实验），并完成视觉问答、图像描述与目标计数评测及 Gradio Demo 部署。

**填写纪律**：所有数字必须来自 `report/experiments.md` 中的真实记录。宁可写"约 2×"也不要写没测过的"10×"。

### 9.2 面试时可展开的技术点（这些才是项目的真正收益）

1. 为什么 Projector 的 lr 要比 LoRA 大一个量级？
2. 视觉特征缓存为什么可行？什么情况下不可行（视觉塔需要解冻时）？
3. `inputs_embeds` 拼接的 off-by-one 陷阱具体是什么？
4. 只训 Projector 时，LLM 是如何"学会看图"的？
5. 你的 QLoRA 实验和 LoRA 实验差异有多大，为什么？
6. 数据规模从 10k 到 50k，哪个任务先饱和？
7. 你怎么判断模型是真的看图，而不是在猜常见答案？

---

## 10. 学习资源清单（只看这些，不要扩张）

| 顺序 | 材料 | 重点章节 | 建议时间 |
| --- | --- | --- | --- |
| 1 | [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | Abstract、Figure 1、Model Architecture、Visual Instruction Tuning、Experiments | W1，2h |
| 2 | [HF Multimodal Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating_multimodal) | Processor → Chat Template → Image+Text → Model | W1，1h |
| 3 | [HF PEFT 文档](https://huggingface.co/docs/peft/) | LoRA、Adapter、trainable parameters | W1/W6，2h |
| 4 | [HF PEFT Quantization 指南](https://huggingface.co/docs/peft/developer_guides/quantization) | 4-bit + LoRA 组合，不深究量化数学 | W6，1h |
| 5 | [Qwen2.5-VL 文档](https://huggingface.co/docs/transformers/main/model_doc/qwen2_5_vl) | 作为**参考实现**，看它如何组织 processor 与 model | W5，1.5h |
| 6 | BLIP-2 论文（可选） | Q-Former 与 Projector 的对比思路 | W7，1h |

> **反面清单**：不要再看第 11 个课程/视频。本项目采用"**论文 + 官方文档 + 源码**"三件套，遇到问题直接读源码，这是唯一能真正提升工程能力的方式。

---

## 11. 每周固定节奏（每周约 12 小时）

| 星期 | 时长 | 内容 |
| --- | --- | --- |
| 周一 | 1.5h | 理论学习（论文 / 文档） |
| 周二 | 2.0h | 阅读源码（transformers / peft 对应实现） |
| 周三 | 2.0h | 写代码 |
| 周四 | 1.5h | Debug + 补测试 |
| 周五 | 2.0h | 实验（本地为主；W5 起含云端操作） |
| 周六 | 3.0h | 项目推进 + 大块任务（数据构建、训练、评测） |
| 周日 | 休息 / 整理笔记 + **写周记 + 验收清单核对 + 更新成本台账** |

**每周周记模板（10 分钟写完，价值极高）**

```markdown
## Week N 周记
- 本周目标：
- 实际完成：
- 卡住的问题 & 解决方式：
- 花掉的云 GPU 费用：
- 验收清单：□ 全部通过  □ 部分通过（列出未通过项）
- 下周调整：
```

---

## 12. 立即行动：今天/本周的前三步

> **不要现在租 GPU。** 第一阶段直接从本地开始。

**Step 1（今天，1 小时）**
- [ ] 建立仓库骨架（按 §4 的目录结构创建空文件）
- [ ] 写 `requirements.txt` 与 `scripts/check_env.py`
- [ ] 建虚拟环境，装 PyTorch，跑通 check_env，记录 GPU 信息

**Step 2（本周内，3–4 小时）**
- [ ] 预下载 `clip-vit-base-patch32` 与 `qwen2.5-0.5b-instruct` 到 `models/pretrained/`
- [ ] 跑通 `VisionEncoder` 前向，记录 `N`、`D`，保存一张特征图可视化
- [ ] 读完 LLaVA 论文的 Architecture 部分，画出 §1 的架构图并标注每个 Tensor 的 shape

**Step 3（本周末，2 小时）**
- [ ] 实现三种 Projector，打印参数量，与 §Week 3 的表格核对
- [ ] 完成"注入自检"：换图 → 输出变化；置零 → 退化为纯文本
- [ ] 回答 Week 1 的五个验收问题，写进 `notes/week1_vlm_notes.md`

**当且仅当 100 样本过拟合测试通过后，才决定第一笔云 GPU 预算怎么花。**

---

## 附录 A：知识点自评表（进度追踪用）

| 模块 | 重要程度 | 起始状态 | Week 4 | Week 8 |
| --- | --- | --- | --- | --- |
| Transformer | ★★★★★ | 已掌握 | — | — |
| ViT | ★★★★★ | 已掌握 | — | — |
| CLIP / SigLIP | ★★★★★ | 待学 | | |
| LLM 基础 | ★★★★★ | 部分 | | |
| Multimodal Token | ★★★★★ | 待学 | | |
| Projector | ★★★★★ | 待学 | | |
| Instruction Tuning | ★★★★★ | 待学 | | |
| LoRA | ★★★★★ | 待学 | | |
| QLoRA / Quantization | ★★★★ | 待学 | | |
| VLM Evaluation | ★★★★★ | 待学 | | |

> 起始状态评估：你已经完成其中约 **30%–40%**（Transformer + ViT），所以现在启动这个项目并不突兀。

## 附录 B：最终能力图谱（项目完成时你应该形成的认知）

```text
               Modern AI
                   │
          ┌────────┴────────┐
          │                 │
       Vision              LLM
          │                 │
         ViT            Transformer
          │                 │
        CLIP              LoRA
          │                 │
          └──────┬──────────┘
                 │
             Projector
                 │
                 ▼
                VLM
                 │
       ┌─────────┼─────────┐
       │         │         │
      VQA     Caption    Reasoning
```

**学习路线的位置**：

```text
CNN → ResNet → YOLO → Transformer → ViT → 【CLIP → LLM → LoRA → VLM ← 你在这里】
```

## 附录 C：项目最重要的原则（贴在显示器上）

> **不要把目标定义成"训练出一个很厉害的 VLM"。**
> **真正的目标是把现代多模态模型的完整技术链打通。**

- 亲手实现的是**连接**，不是**底座**。
- 每一行代码都要能解释"为什么这么做"，而不只是"能跑"。
- 每一次云端训练前，先问：这个实验在本地能不能先验证？
- 每一个数字都必须来自真实记录。
- 效果不理想不是失败，**没有记录、没有分析、没有复现**才是失败。

---

*计划书完 —— 执行过程中如与实际情况冲突，以"最低成功标准"为准，动态微调周计划，并把调整写进周记。*
