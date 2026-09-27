# Mini-VLM 执行追踪表（配合《Mini-VLM项目计划书.md》使用）

> 用法：每完成一项就把 `[ ]` 改成 `[x]`。**周验收未通过，不进入下一周的新内容。**
> 每次云端训练后必须更新「实验记录」与「成本台账」。

---

## 一、环境与资源（Week 1）

- [x] 仓库骨架按计划书 §4 创建完毕（注：模型实际落在 `checkpoints/pretrained/`，不是 `models/pretrained/`）
- [x] `requirements.txt` / `environment.yml` 写好（注：现有 `environment.yml` 实际是 check_env 的输出快照，需重写为真正的环境声明）
- [x] `scripts/check_env.py` 跑通，记录：GPU = **RTX 4050 Laptop**，显存 = **6.00** GB，bf16 = **True**
- [x] `checkpoints/pretrained/clip-vit-base-patch32/` 已下载（1735 MB，含 tf/flax 冗余权重）
- [x] `checkpoints/pretrained/qwen2.5-0.5b-instruct/` 已下载（953 MB）
- [x] 离线变量固定：`HF_HOME`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 已写入 `scripts/env.ps1` + `scripts/check_env.py`（自设置、幂等）
- [x] 一条完整构造样本打印出来，肉眼确认无误

### 1.1 离线变量固定 —— 已完成（2026-09 实测）

**用法（每个新终端）**

```powershell
. .\scripts\env.ps1                      # 注意开头有一个点
& $env:MINIVLM_PY scripts/check_env.py --deep      # 环境自检
& $env:MINIVLM_PY scripts/verify_offline.py        # 离线加载验收
& $env:MINIVLM_PY short_time_check.py              # 视觉塔冒烟测试
```

**验收结果**

| 检查 | 结果 |
| --- | --- |
| `check_env.py --deep` | 全绿；离线拦截实测生效（抛 `LocalEntryNotFoundError`） |
| `verify_offline.py` | **9 PASS / 0 FAIL / 0 WARN**，退出码 0 |
| `short_time_check.py` | 输出 `(2, 49, 768)`，退出码 0 |

**关键事实（已实证，写进配置别再猜）**

| 项 | 实测值 |
| --- | --- |
| 正确 python | `dl_project\venv\Scripts\python.exe`（3.12.4，torch 2.6.0+cu124） |
| 系统默认 python | 3.14.6 —— **没有 torch，用它必失败** |
| CLIP | image_size 224 / patch 32 → `last_hidden_state` **[B,50,768]**，第 0 个是 CLS；丢弃后 49 个 patch |
| Qwen2.5-0.5B | hidden_size **896**、24 层、14 heads / 2 kv_heads、tie_word_embeddings=True、bos_token_id=None |
| 占位符 | `<\|image_pad\|>` = **151655**（已是单一 token）；embedding 权重 **151936** 行 → 不越界 |
| resize 结论 | **不需要也不允许** `add_special_tokens` / `resize_token_embeddings` |
| 缺失依赖 | `datasets`、`gradio`、`bitsandbytes` 未装（Week 5/6 再装） |

**本次顺手修掉的真 bug**

`models/vision_encoder.py` 的 `drop_cls` 原本会**误删第一个 patch token**（原实现以为 CLIP 输出不含 CLS）。
已改为自动探测：只有 `out.shape[1] == patch_tokens + 1` 时才切片，并由 `verify_offline.py` 做防回归断言。

---

## 二、Week 1：理解 VLM

- [ ] 画出架构图并标注每个 Tensor 的 shape（含 batch 与序列长度示例）
- [ ] 读完 LLaVA 论文 Architecture 部分，笔记落盘 `notes/week1_vlm_notes.md`
- [ ] 读完 HF Multimodal Chat Templates 文档
- [ ] 能回答以下 5 问（不看笔记）：
  - [ ] 为什么 Vision Encoder 不能直接接 LLM？
  - [ ] Projector 在解决什么问题？
  - [ ] 为什么图像可以表示成一系列 Token？
  - [ ] VLM 与普通 LLM 的最大区别是什么？
  - [ ] 49 个 visual token 对训练成本意味着什么？

---

## 三、Week 2：Vision Encoder

- [x] `models/vision_encoder.py` 完成（含 `drop_cls` 自动探测）
- [x] 记录：N = **49**，D = **768**，CLS 处理 = **drop_cls=True（实测 CLIP 确实含 CLS，需切片）**
- [x] 冻结验证：可训练参数 **0 / 87,456,000**
- [x] 零样本图文匹配实验完成，保存成功 + 失败案例
- [x] 数据增强敏感性实验完成（翻转 cos=0.9861 / 裁剪40% cos=0.9085）
- [x] 1000 张图特征缓存完成：单图 **11.67 ms**，缓存体积 **74.2 MB**（76 KB/图）
- [x] 单图特征提取 < 20ms（实测 11.67 ms），1000 张 < 30s（实测 **8.34 s**）

### 3.1 Week 2 执行结果（2026-09 实测，全部基于 VOC 真实图片）

**数据源**：本机 `7.YOLOv1\data\...\VOC2007`（9963 张真实照片 + XML 标注，**零下载成本**）

| 实验 | 关键数字 | 结论 |
| --- | --- | --- |
| A 形状与冻结 | `[B,50,768]` → `[B,49,768]`，可训练参数 0 | 冻结生效 |
| A.3 预处理几何 | 裁剪假设误差 0.3 列 vs 拉伸 149.7 列 | **是「缩放+中心裁剪」，不是拉伸** |
| B 逐层分析 | 第 11 层与最终层 cos 仅 0.299 | 表示逐层渐进改写，最终层不可替代 |
| C 零样本 | **Top-1 86.5% / Top-5 97.5%**（随机 5%） | **编码器确实携带可用语义** |
| C2 kNN（无文本） | **81.50% ± 2.89%**（随机 5%，16.3×） | 视觉判别力的独立证据 |
| D 增强敏感性 | 翻转 cos=0.986 但 patch 仅 0.722 | **训练禁用水平翻转** |
| E 特征缓存 | 1000 张 **8.34 s**，74.2 MB；5 万张约 7 分钟 / 3.62 GB | 本地 4050 完全够用 |

**本周发现并修掉的问题**

| # | 问题 | 处理 |
| --- | --- | --- |
| 1 | 预处理探针用**归一化后**的张量求 sign，背景噪声被放大出虚假变号点，把结论误判成"拉伸" | 改为「先判端点颜色是否不同」，并确立"无分界 = 被裁剪"的判据 |
| 2 | B 节「最接近最终层的中间层」把最终层也算进去，自比恒为 1，结论无意义 | 改为只在 **0..L−2** 内比较，得第 11 层 |
| 3 | B 节用 L2 度量相邻层变化，被 `post_layernorm` 的整体缩放污染 | 改用**余弦距离** `1−cos` |
| 4 | 零样本"置信度 0.051"看着像 bug | 实为 `logit_scale=100` 下的正常现象；改用**余弦间隔**（top1−top2=0.0421） |
| 5 | 增强实验里 `torch.eye` 掩码形状与 `[B,N,N]` 不匹配 | 改用 `torch.triu_indices` 取上三角 |
| 6 | matplotlib 无中文字体，图里全是豆腐块；字体告警刷屏数百行 | 自动挑选 `Microsoft YaHei`/`SimHei` 并过滤告警 |
| 7 | kNN 实验不在默认执行列表（由 `Z` 触发），且普通 KFold 在 20 类下可能缺类 | 改为 `K` 触发并默认执行；换用 `StratifiedKFold` |

### 3.2 交付物

```text
scripts/run_experiments.py                    四个实验 + 自动报告（一条命令，约 35 s）
datasets/voc_source.py                        VOC 数据源（XML 解析 / 均衡采样 / 索引缓存）
datasets/cache_features.py                    特征离线缓存（可复用模块 + CLI）
evaluation/report_utils.py                    报告生成工具（数字与结论同源）
report/figures/w2_feature_shapes.md           实验 A/B/D/E
report/figures/w2_clip_zeroshot.md            实验 C + C2（kNN）
report/figures/w2_experiments.md              完整汇总报告
report/figures/w2_results.json                机器可读结果
report/figures/w2_layer_evolution.png         逐层特征演化图
report/figures/w2_augment_sensitivity.png     增强敏感性图
notes/week2_vision_encoder_notes.md           Week 2 笔记（含三条硬约束）
data/processed/voc_index.json                 VOC 索引（9963 条，缓存复用）
data/processed/features_w2/                   1000 张特征缓存 + index.json
```

### 3.3 对后续周次的硬约束

1. **Week 4 数据工程**：Dataset 必须先 `pad_to_square` 再交给 CLIPProcessor
   （否则靠边目标与标注不一致，会污染计数与空间关系任务）。
2. **Week 6/7 训练增强**：**禁用水平翻转**（翻转使 patch 特征相似度从 1.0 掉到 0.722，
   而空间关系任务恰恰依赖左右位置）。
3. **Week 5 训练管线**：复用 `data/processed/features_w2` 的格式
   （`patch_features` / `cls_feature` / `image_embeds` + `index.json`），训练时不加载 CLIP。

### 3.4 一个必须记住的区分

- **VLM 用投影前的 768 维 patch 特征**（喂 Projector）
- **零样本/检索必须用投影后的 512 维归一化嵌入**（CLIP 的对比学习目标定义在这里）

混用会得到毫无意义的指标。

---

## 四、Week 3：Projector

- [x] `models/projector.py` 完成（Linear / 2-layer / 3-layer）
- [x] 参数量实测并与计划书表格核对：

  | Projector | 实测参数量 | 与预期一致 |
  | --- | ---: | --- |
  | Linear | **689,024** | [x] |
  | 2-layer MLP | **3,410,816** | [x] |
  | 3-layer MLP | **7,607,168** | [x] |

- [x] 占位符 token 检查（计划书 §4.2）全部通过
- [x] 数值自检：`[2,49,768] → [2,49,896]`，无 NaN
- [x] **注入自检**：换图 → 输出变化；visual embeds 置零 → 退化为纯文本行为
- [x] 梯度自检：仅 Projector 有梯度
- [x] `tests/test_projector.py` 相关断言已在 `overfit_test.py` 中固化
- [x] 确认 LLM `hidden_size` = **896**，与 Projector `out_dim` 一致

### 4.1 Week 3 执行结果（2026-09 实测）

**上云门禁 8 项断言全部通过** —— `scripts/overfit_test.py --n 100 --steps 300`

| 断言 | 实测 |
| --- | --- |
| A. loss 收敛 | 3.6226 → **0.0043**（初始值的 0.1%） |
| B. 答案 token 准确率 | **100.00%**（596/596），整句全对 **100/100** |
| C. 换图输出变化 | **6/8** |
| D. 只有 Projector 可训练 | **3,410,816 / 584,899,584（0.58%）** |
| D′. 只有 Projector 有梯度 | 4 个张量，全部属于 projector |
| E. 生成参考答案 | 精确 **3/8**，前缀匹配 **5/8** |

资源：300 步 **68 秒**（225 ms/step），显存峰值 **3501 MB**。

**模型规模**

```text
vision :   87,456,000  trainable=0
llm    :  494,032,768  trainable=0
proj   :    3,410,816  trainable=3,410,816   ← 只用 0.58% 参数
```

### 4.2 本周踩的 8 个坑（详见 notes/week3_projector_notes.md）

| # | 问题 | 处理 |
| --- | --- | --- |
| 1 | `encode_images` 里的 `torch.no_grad()` 切断 Projector 梯度，`backward()` 报 "does not require grad" | 冻结靠 `requires_grad_(False)`，不靠切断计算图 |
| 2 | 预编码缓存了 **Projector 之后**的特征并 detach，梯度再次断掉 | 只缓存**视觉塔原始特征**，Projector 留在计算图内 |
| 3 | 把变长序列 `cat` 成 `[总token数, D]` → Qwen2 推断不出 batch → RoPE 报形状错 | 必须保持矩形 `[B, L]` + `attention_mask` |
| 4 | 生成时 `logits[:, -1, :]` 取到 padding 位置 | 按各样本有效长度取 logits |
| 5 | 锚点定在"答案**之后**"，模型只预测终止符 | 锚点必须在"答案**之前**" |
| 6 | `<\|im_end\|>` 在 system/user/assistant 各出现一次，取了第一个 | 用 `row_eos[-1]` / `row_eos[-2]` 定位 |
| 7 | 模型会继续生成"下一轮对话"结构 token | 用 collator 的 `answer_starts` 精确指定起点 + 首个终止符处截断 |
| 8 | Windows 控制台 GBK，模型吐出越南语字符导致 print 崩溃 | stdout 切 UTF-8 + 输出字符过滤 |

**其中 6 个属于"训练指标完美但生成垃圾"** —— loss 0.0043、token 准确率 100%、自由生成却是
`'HumanHumanHuman...'`。**这正是 100 样本门禁存在的意义**：只盯 loss 就上云，这些坑会在烧完预算后才暴露。

### 4.3 Week 3 交付物

```text
configs/model_clip_b32_qwen05.yaml     真实配置（实测参数 + 硬约束注释）
datasets/build_voc_vqa.py              程序化 VQA 构造（五类任务均衡配额）
datasets/dataset.py                    JsonlVLDataset + FeatureCacheReader
datasets/collator.py                   占位符展开 + pad_to_square + 手动 padding
models/multimodal_model.py             组装 / forward / 手写自回归 generate
models/projector.py                    三种结构 + 参数量表
training/losses.py                     掩码损失 + 优化器分组 + 余弦调度
scripts/overfit_test.py                100 样本门禁（8 项断言，上云前必跑）
data/processed/debug_1k.jsonl          1000 条训练数据（5 类 × 200）
report/figures/w3_overfit_test.json    门禁结果（含完整 loss 曲线）
notes/week3_projector_notes.md         Week 3 笔记（8 个坑的完整记录）
```

### 4.4 数据构造的副产品（Week 4 的起点）

`build_voc_vqa.py` 已能生成五类任务，配额均衡（各 20%），实测：

| 任务 | 样本数 | 唯一答案数 | 最高频占比 |
| --- | ---: | ---: | ---: |
| counting | 200 | 47 | 16.0% |
| existence | 200 | 38 | 15.5% |
| listing | 200 | 54 | 13.5% |
| attribute | 200 | 20 | 24.5% |
| spatial | 200 | 66 | 5.5% |

**全局多数类基线仅 4.9%** —— 模型必须显著超过它才有意义（这个数字后面写报告时要用）。

> ⚠️ 数据生成器第一版有个隐蔽 bug：按"每图最多 N 条 + 固定任务顺序"生成时，
> `max_per_image=3` 会让排在后面的 attribute / spatial **一条都生成不出来**。
> 已改为按配额均衡分配。

---

## 五、Week 4：数据工程

- [x] JSONL schema 最终冻结
- [x] 五类任务生成完成（各 2,000 条，共 10,000）
- [x] 三档数据：1k = **1000**，10k = **10000**，50k = 未做（10k 已足够暴露瓶颈）
- [x] `splits.json` 固定，按 **image_id** 划分，train/val/test 交集为空（脚本硬校验）
- [x] 答案分布直方图已出图
- [x] 真重复（同图+同问+同答）**0.00%** < 1%
- [x] 抽查样本的 labels：question 部分全部为 -100（见 `scripts/inspect_sample.py`）
- [x] `report/data_stats.md` 数据卡片完成（**六项自动验收全通过**）
- [x] 一条命令重建全部数据可复现

### 5.1 数据卡片（10k）

| 项 | 值 |
| --- | --- |
| 样本 / 图片 | **10,000 / 6,679** |
| 唯一答案 / 问题 | 548 / 193 |
| 答案长度 均值/中位/P95 | 7.5 / 8 / 9 字 |
| 全局多数类基线 | **4.66%** |
| 真重复率 | **0.00%** |
| spatial 左右平衡 | **48.9% / 51.1%** |
| 泄漏自检 | **passed** |

**六项验收全部通过**：样本 ≥ 10k ✅ / 五类均衡 ✅ / 多数类基线 < 10% ✅ /
真重复 < 1% ✅ / 按 image_id 无泄漏 ✅ / spatial 平衡 ✅

### 5.2 本周抓到两个真问题（都很隐蔽）

| # | 问题 | 后果 | 处理 |
| --- | --- | --- | --- |
| 1 | **spatial 任务答案 100% 是「左边」**（实测 2000/2000） | 模型只要"永远答左边"就拿满分，**任务完全失效但指标漂亮** | 生成器改为**随机**选择被问主体 → 48.9% / 51.1% |
| 2 | 重复率判据用 `(question, answer)` | 模板化问答天然重复，误报 **95%** | 改用「同图+同问+同答」（实测 0.00%），卡片同时报告两个数 |

> **问题 1 的教训**：程序化构造数据时，必须检查**标签本身的分布**。
> 一个恒定的标签比一个错误的标签更危险 —— 它会让指标虚高。

### 5.3 交付物

```text
datasets/build_voc_vqa.py            程序化 VQA 构造（五类均衡 + 随机化 spatial 主体）
datasets/make_splits.py              按 image_id 划分（含泄漏硬校验）
scripts/make_data_card.py            数据卡片（六项自动验收 + 两张分布图）
scripts/evaluate.py                  按任务评测（含各任务自己的多数类基线）
data/processed/baseline_10k.jsonl    10,000 条
data/processed/splits.json           train 8989 / val 489 / test 522
report/data_stats.md                 数据卡片
report/figures/w4_data_overview_baseline_10k.png
report/figures/w4_answer_dist_baseline_10k.png
notes/week4_data_engineering_notes.md
```

### 5.4 数据规模消融结论（1k vs 10k）

| 指标 | 1k | 10k | 变化 |
| --- | ---: | ---: | ---: |
| 生成精确匹配 | 30.0% | 32.0% | **+2.0%** |
| 最优 val_loss | 0.1126 | 0.1051 | −0.008 |

**数据量 10 倍，精确匹配只涨 2 个百分点 → 瓶颈不在数据量。**
假设瓶颈在**冻结 LLM 的表达能力**，由 Week 6 的 LoRA/QLoRA 实验验证。

---

## 六、Week 5：组装 Mini-VLM + Baseline

- [x] `models/multimodal_model.py` 完成（组装 + forward + 自回归 generate）
- [x] `training/trainer.py` 完成（优化器分组、schedule、grad clip、日志、checkpoint）
- [x] `training/losses.py` 完成（labels 构造 + loss mask）
- [x] **100 样本过拟合测试通过**：loss **3.6226 → 0.0043**，答案 token 准确率 **100%**
- [x] 换图输出会变（真实模型中验证：**6/8**）
- [x] 本地 1k 数据跑通，loss 正常下降
- [x] **本地 1k Baseline 跑完**（EXP-001），耗时 **302 s**，checkpoint 已保存
- [ ] 本地 10k Baseline（Week 4 数据就绪后再跑）
- [ ] 首次云端训练完成，实际花费 ¥ ______（1k 本地已出结果，可延后）
- [x] 评测完成：生成精确匹配 **30%** / 前缀匹配 **40%**
- [x] 与"多数类基线"对比：模型 **30.0%** vs 基线 **5.0%**（**6.0×**）

### 6.1 EXP-001 结果摘要（本地 1k Projector-only Baseline）

| 指标 | 值 |
| --- | --- |
| 最优 val_loss | **0.1126**（step 360 / 400） |
| val 答案 token 准确率 | **96.6%** |
| 生成精确匹配 | **30.0%**（40 条验证样本） |
| 生成前缀匹配 | **40.0%** |
| 多数类基线 | **5.0%** → 模型是它的 **6.0×** |
| 训练耗时 / 显存峰值 | **302 s** / **3651 MB** |
| 花费 | **¥0**（全本地） |

**核心结论**：只训练 **0.58%** 的参数（3.41M / 584.9M），就验证了 Experiment A ——
冻结的 LLM 确实能用上 Projector 送进来的视觉信息。

**失败案例分析**：几乎全是"**先答对，再续写废话**"
（`图中有猫中有猫。猫。`、`图片主要是 picture。主要是人.`）——
说明视觉条件与解码路径都正确，差的是"什么时候停"，属于**欠拟合**。
下一步应**加数据**，而不是改结构。

**本周新增交付物**

```text
training/trainer.py                训练循环（特征预编码 + 优化器分组 + best 选择）
datasets/make_splits.py            按 image_id 划分（含泄漏自检）
scripts/train.py                   训练入口（训练 + 验证 + 生成评测 + 出图）
data/processed/splits.json         固定划分（train 900 / val 49 / test 51）
outputs/baseline_1k/               checkpoint_best / checkpoint_last / 训练曲线 / 评测结果
report/figures/train_curve_baseline_1k.png
report/experiments.md              EXP-001 完整记录
```

**本轮又修掉一个 bug**：`Trainer.make_batch` 直接把**原始问题**传给 collator，
绕过了 `JsonlVLDataset.build_question`（它负责插入 `<|image_pad|>`）→
前向时报"占位符 0 个，视觉 token 49 个"。
已修正，并在 collator 里加了**快速失败**校验（错误信息直接指出根因）。

---

## 七、Week 6：LoRA / QLoRA

- [x] LoRA 注入脚本 `models/lora.py` 完成
- [x] 两组 lr 分别设置（projector 1e-3 / lora 2e-4）已确认生效（参数组打印可见）
- [x] 实验 A（Projector-only）结果归档：**32.0%** 精确 / 51.0% 前缀
- [x] **实验 B（+LoRA bf16）完成**（EXP-003），实际花费 **¥0**
- [ ] 实验 C（+QLoRA 4-bit）：**本机未装 bitsandbytes，且 B 已是否定结果 → 优先级下调**
- [x] 对照表填写完整（显存 / 可训练参数 / 单步耗时 / 成本 / 指标）
- [x] 两组训练曲线保存到 `report/figures/`

### 7.1 EXP-003 结果（严格同口径：val 前 200 条）

| 指标 | A: Projector-only | **B: + LoRA** | 变化 |
| --- | ---: | ---: | ---: |
| 总体精确匹配 | **32.0%** | 28.5% | **−3.5%** |
| 总体前缀匹配 | **51.0%** | 50.0% | −1.0% |
| 最优 val_loss | 0.1051 | **0.0949** | −0.010 |
| val token 准确率 | 95.5% | **96.0%** | +0.5% |
| **视觉接地强度** | +3.0% | +3.5% | 基本持平 |
| 可训练参数 | 3.41M（0.58%） | 5.57M（0.95%） | — |
| 显存峰值 | 3586 MB | 3799 MB | +213 MB |
| 单步耗时 | 1548 ms | 2130 ms | +37% |
| 花费 | ¥0 | **¥0** | — |

| 任务 | A | B | 多数类基线 |
| --- | ---: | ---: | ---: |
| counting | 56.2% | 56.2% | 12.1% |
| existence | 20.9% | **25.6%** | 12.8% |
| listing | 33.3% | 33.3% | 10.6% |
| spatial | **48.6%** | 34.3% | 51.0% |
| attribute | **9.5%** | 0.0% | 22.8% |
| 超过基线数 | **3/5** | 3/5 | |

### 7.2 结论：**否定结果**（如实记录）

1. **LoRA 在这套默认配置下没有带来可确认的收益** —— val_loss 略降，
   但 test 生成质量与视觉接地都未提升。
2. **val 上的提升没有兑现**：existence 在 val 上 12.5%→25.6%，
   test 上却是 11.5%（与 A 持平）→ **LoRA 版过拟合了验证集**。
3. **容量不是瓶颈**：0.58% 参数（A）优于 0.95% 参数（B）。
   限制表现的是**输入信息质量**，不是可训练参数量。

**诚实局限**：LoRA 超参未调（lr / target_modules / r / alpha）。
准确表述是"这套配置下无明显收益"，不是"LoRA 无效"。

### 7.3 本周修掉一个危险 bug：静默的部分加载

`scripts/evaluate.py` 从**配置文件**读 `strategy`（= projector_only），
用它加载 LoRA checkpoint 时 LoRA 权重**被静默丢弃** →
评测从 50% 掉到 **0.8%，脚本还打印"已加载"**。

**修复**：
1. `Trainer.load_checkpoint(strict=True)` —— checkpoint 有、模型没有的参数**直接报错**，
   并提示"常见原因：用 projector_only 配置加载 LoRA checkpoint"
2. `scripts/evaluate.py` —— 从 checkpoint 的 `config` 字段读回真正用过的 strategy

> **教训**：静默的部分加载比直接报错危险得多。加载权重应默认 `strict=True` 并报告张量数。

### 7.4 关于 QLoRA（Experiment C）

本机未安装 bitsandbytes（计划书 R1，概率：高）。代码已做明确处理：
`prepare_qlora` 导入失败时抛出**带解决方案**的错误（WSL2 / 退回 bf16 LoRA / 云端 Linux），
而不是训练到一半才崩。

**优先级下调理由**：LoRA 与 QLoRA 的差别只是"基础模型是否量化"。
既然 LoRA 无收益，QLoRA 也不会改变结论 —— 除非先确认 LoRA 是训练不足。
**建议把云预算留给 Week 7 的视觉 Token 消融。**

### 7.5 交付物

```text
models/lora.py                       LoRA/QLoRA 注入 + 自检 + QLoRA 明确报错
models/multimodal_model.py           strategy 支持（projector_only/lora/qlora）
training/trainer.py                  checkpoint 存/取 LoRA + strict 校验
scripts/train.py                     --strategy / --lora-lr / --lora-r / --grad-ckpt
scripts/evaluate.py                  从 checkpoint 读回 strategy（防静默丢权重）
outputs/lora_10k/                    checkpoint（21MB，含 192 个 LoRA 张量）
report/figures/train_curve_lora_10k.png
notes/week6_lora_notes.md            完整记录（含机制解释与局限说明）
```

---

## 八、Week 7：正式实验与消融

> 2026-09-23 先导状态：已修复批量解码并全量重评 10k 权重；已审计空间歧义、生成 9,333 条清理数据；B/32 清理版通过 dry-run、100 样本和 1k smoke，并在本地完成 1,200 步训练及 489 条 test/错图评测。空间题原图、错图、无图都是 31/76，尚无空间视觉收益证据。详见 `report/experiments.md` 的 EVAL-006/EXP-007。下列正式 Week 7 消融任务尚未勾选。

> 2026-09-23 更新：B/16 官方权重已固定版本，196 token 的 dry-run、100 样本过拟合、1k/50 步 smoke 均通过；已生成可移植云端输入包和 Linux 启动脚本。**B/16 全量 1,200 步和 test 对照仍未执行**，因此“视觉 token 数量消融完成”暂不勾选。详见 `report/experiments.md` 的 EXP-008。

> 2026-09-24 更新：B/16 已在 AutoDL RTX 4090 完成 1,200 步和 489 条 test/错图评测；总体精确匹配 76.5%，空间原图 37/76、错图 46/76。视觉 token 数量消融已执行，但未获得可信的空间方位收益。最佳 checkpoint 已归档；本次费用 ¥3.82；水平翻转对照仅 3/76 道改变左右答案，见 EXP-009/EXP-010。

- [ ] 实验一（Projector 结构）完成
- [x] 实验二（visual token 数量）完成
- [ ] 实验三（Frozen / LoRA / QLoRA，含 r 值对比）完成
- [ ] 实验四（数据规模 1k / 10k / 50k）完成
- [ ] 补充实验：幻觉率 = ______%（不存在的物体误报率）
- [ ] 四张表全部填满真实数字
- [ ] 至少 3 个定性案例（2 成功 + 1 失败，失败含原因分析）
- [ ] `report/experiments.md` 每条实验记录完整（含花费与结论）
- [ ] 数据规模曲线出图

---

## 九、Week 8：Demo + GitHub + 报告

- [ ] `demo/app.py` 完成：上传图 + 提问 → 回答
- [ ] Demo 含预设问题按钮、生成参数调节、推理耗时显示
- [ ] Demo 含失败案例免责说明
- [ ] `scripts/inference.py` 一条命令可复现推理（干净环境验证过）
- [ ] README 十节齐全（含 Windows / WSL2 两种安装路径）
- [ ] 技术报告完成（≥ 4 表 + ≥ 3 图）
- [ ] `report/figures/` 图表齐全
- [ ] `.gitignore` 排除 data / models / outputs
- [ ] 仓库推送完成，含 1 张 Demo 截图
- [ ] 简历条目定稿（所有数字可追溯）
- [ ] 录一段 1–2 分钟 Demo 演示视频（可选但推荐）

---

## 十、成本台账（每次云端运行后立刻填写）

| 日期 | 实验名 | 卡型 | 时长(h) | 单价(¥/h) | 花费(¥) | 累计(¥) | 结果 | 是否达预期 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-09-24 | B/16 视觉 token 消融 | RTX 4090 | 待账单核对 | 待账单核对 | 3.82 | 3.82 | 1,200 步 + 489 条 test/错图完成 | 空间题未达预期 |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

**分周预算对照**

| 阶段 | 周 | 预算 | 实际 | 状态 |
| --- | --- | ---: | ---: | --- |
| 阶段 1 | W1–W4 | ¥0 | ¥0 | |
| 阶段 2 | W5 | ¥100–150 | | |
| 阶段 3 | W6 | ¥200–300 | | |
| 阶段 4 | W7 | ¥200–300 | | |
| 阶段 5 | W8 | ¥0–100 | | |
| **合计** | | **≤ ¥1000** | | |

---

## 十一、上云门禁（每次租卡前逐项确认，缺一不可）

```text
[ ] 代码 diff 已 review，无临时调试代码
[ ] config 路径 / 数据量 / epoch / lr 已核对
[ ] 100 样本过拟合测试通过（loss < 0.1 或 token 准确率 > 95%）
[ ] 1000 样本跑通，loss 正常下降
[ ] dry-run 打印的 batch shape 与可训练参数统计正确
[ ] 本次预估花费 = ¥______（单次不超过 ¥60，单次时长不超过 4h）
[ ] 已准备日志落盘 + checkpoint 下载方案
[ ] 已设置关机提醒
```

---

## 十二、实验记录模板（`report/experiments.md` 每条一行块）

```markdown
### EXP-000 | 日期 | 实验名
- config      : configs/train_xxx.yaml
- 数据         : baseline_10k.jsonl (train 9000 / val 500 / test 500)
- 硬件         : RTX 4090 24G (云) / RTX 4050 6G (本地)
- 可训练参数    : x.xx M (占比 x.xx%)
- 关键超参      : lr=..., bs=..., accum=..., epoch=..., max_len=...
- 耗时 / 花费   : xx min / ¥x.xx
- 结果         : VQA=..., Caption BLEU-4=..., Counting=..., val_loss=...
- 结论         : （一句话，说明这组实验支持什么判断）
- 异常/失败原因 : （若有，如实记录）
```

---

## 十三、周记模板（每周日 10 分钟）

```markdown
## Week N 周记
- 本周目标：
- 实际完成：
- 卡住的问题 & 解决方式：
- 云 GPU 花费：
- 验收清单：□ 全部通过  □ 部分通过（未通过项：______）
- 下周调整：
```

---

## 十四、红线（违反即暂停本周云端实验）

1. 未通过 100 样本过拟合测试就上云。
2. 单次云端训练无 checkpoint 保存点且超过 4 小时。
3. 训练结束未立即下载 checkpoint / 日志。
4. 忘记关机导致空转计费。
5. 修改了 `splits.json` 或评测集后，与历史实验直接横向比较。


### 2026-09-24 补充进展：空间诊断（EXP-011 / EXP-012）

- 冻结 CLIP 的左右分区特征可被小判别器利用：test 原图 59/76；此结果不是 VLM 成绩。
- 本地完成原图/镜像两组 Projector 续训，各 64 步，验证配对正确数仅从 62/124 变为 64/124，同时答对从 4/62 降为 3/62，未通过本轮门槛。
- 未运行新的 test 评测，未替换云端最佳权重，新增云费用 0 元，累计 3.82 元。
- 下一项：32 对训练样本的镜像过拟合诊断，先确认能否学会训练关系，再决定结构改动；暂不扩大云训练。


### 2026-09-24 补充进展：32 对镜像过拟合（EXP-013）

- 本地第192步（24轮）达到原图32/32、镜像32/32、同时答对32/32，按协议提前停止。
- 这是训练样本拟合成功，不是泛化提升；未评估本轮权重的val/test。
- 诊断权重单独保存，云端最佳模型未覆盖，新增云费用0元，累计3.82元。
- 下一项：固定step192，评估62对空间验证图、错图对照及其他任务退化，再决定是否扩大训练。


### 2026-09-24 补充进展：过拟合权重验证（EXP-014）

- 固定step192验证：空间配对60/124（原62/124），同时答对5/62，未出现有效泛化收益。
- 其他任务现有评分324/399→93/399；其中186条无法解析，包含同义措辞变化，不能全部算语义错误。
- 保留云端原模型；本轮未训练、未运行test，新增云费用0元。
- 下一项：补齐明确同义表达评分，保留旧结果并对两份固定输出同时重算，再决定混合任务训练方案。


### 2026-09-24 补充进展：评分修复与混合训练（EXP-015/016）

- 内容评分v2支持明确正数量修饰词，27项测试通过；旧输出重算：32对过拟合模型其他任务93/399→148/399，仍低于原模型324/399。
- 完成512张空间原图+镜像、1024条其他任务的混合训练，512步/2轮，本地完成。
- 最终训练权重验证：空间66/124、同时答对4/62；其他任务总分324/399，但计数退步6题。未通过空间门槛。
- 原模型验证loss更低，继续保留原模型；无test评测、无新增云支出，累计3.82元。
- 下一项建议：保留回放，增加方向专项监督和成对验证选点，设置匹配训练对照；暂不追加同方案云训练。


### 2026-09-26 补充进展：方向专项监督匹配对照（EXP-017）

- 完成普通监督/方向token权重6两组，各256步，完全相同的数据与样本顺序，31项测试通过。
- 普通/方向组验证空间配对62/124、66/124；同时答对2/62、4/62；其他任务332/399、333/399。
- 方向组未通过空间门槛，正式模型仍保留原云端最佳权重；无test评测，累计云费仍3.82元。
- 下一项建议：显式空间读出分支的小规模对照，继续保留回放；不再仅增加方向词权重。


### 2026-09-27 补充进展：独立四任务考卷封存（DATA-018）

- [x] 空间位置任务标记为当前版本不支持可靠判断，本轮暂不提升，也不参与扩展验收。
- [x] 排除旧问答/划分以及 Week 2 已触碰图，建立官方 VOC2007 test 来源的 400 图独立集，四任务各100题、每图一题；去除完全重复和近重复。
- [x] 固定内容评分v2、四任务宏平均、分层指标、成对置信区间和通过阈值；协议与题目文件写入哈希。
- [x] 离线完整性/标注规则核验通过；本阶段没有在新集上运行模型，云费累计仍3.82元。
- [ ] 下一项：仅用旧训练/验证集锁定一个四任务候选方案及确定性推理导出，再让旧基线与候选在新考卷上各评一次；不得按新考卷结果反复选点。

详见 `data/independent_eval_v1/README.md` 和 `report/experiments.md` 的 DATA-018。此前 EXP-017 关于继续空间结构试验的“下一项建议”由本轮用户决定暂缓。


### 2026-09-27 补充进展：四任务候选旧验证门禁（EXP-019）

- [x] 原云端 B/16 权重上，按预定分层配额取旧 train 四任务2048题，本地仅续训 Projector；旧 val 399题选点，空间题未参与。
- [x] 旧 val 原模型324/399、宏平均81.32%；step256为332/399、83.39%，只提高2.07点；step512为318/399、79.91%。预设门槛为宏平均至少提高5点且每项退步不超过3点，两候选均未通过。
- [x] 逐题诊断：多实例计数原模型4/29、step256为6/29；多类别列举18/30→16/30。继续同配方加步数没有依据。
- [x] 未生成候选冻结文件，未在独立400题上推理或评分；原权重不变。导出程序已有封存门禁与同环境检查。无新增云费，累计3.82元。
- [ ] 下一项：只用本项目旧 train 图与 VOC XML，设计物体级视觉监督/读出的本地小实验，先看多实例计数和多类别列举能否在旧 val 改善，同时守住其他两任务；不复用在 VOC2007 test 上调过的相邻 YOLO 权重。若旧验证门禁通过，再冻结候选并一次性评独立考卷。

详见 `report/experiments.md` 的 EXP-019 与 `outputs/b16_four_task_pilot/results.json`。


### 2026-09-27 补充进展：本地 demo 初版（DEMO-020）

- [x] 用户决定暂缓 EXP-019 后续物体级读出和其他性能提升，先交付本地上传图片的初版；沿用原 B/16 云端最佳权重，不使用未过门槛的续训权重。
- [x] 支持是否存在、数量、属性（主要物体类别）、物体列举，以及带明显“不可靠”提示的空间位置；类别限定为训练涉及的 VOC 20 类。
- [x] 页面注明实验性结果、空间位置暂不支持可靠判断、多个同类物体计数易错；明确旧验证324/399只是开发成绩，独立400题未评测，不代表任意照片准确率。
- [x] 本地127.0.0.1网页服务与真实旧验证图片的HTTP推理一致性检查通过；42项项目测试和JS语法检查通过。新独立集保持封存，无新增云费，累计3.82元。
- [ ] 下一项：实际打开本地页面试用几张自选照片，优先修正交互或稳定性问题；暂不追加模型训练。浏览器自动截图工具本轮异常重启，截图级视觉检查尚未完成。

启动与限制见 `demo/README.md`，实现与验证见 `report/experiments.md` 的 DEMO-020。上节 EXP-019 所列的“下一项”已由本轮用户决定暂缓。
