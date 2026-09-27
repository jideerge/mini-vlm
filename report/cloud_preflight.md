# B/16 上云训练准备（2026-09-23）

> 2026-09-24 已在 AutoDL RTX 4090 完成 1,200 步训练及 489 条 test/错图评测；
> 结果见 `report/experiments.md` 的 EXP-009。本页其余费用和时长文字是上云前估算。

目标是回答：把 CLIP 的视觉 patch 从 B/32 的 49 个增加到 B/16 的 196 个，
是否能提升方向明确的 76 道空间题，并使原图成绩高于错图和无图对照。
B/32 清理版在三种对照中均为 31/76；这是正式消融的起点，不是 B/16 的结果。

| 门禁 | 当前状态 |
| --- | --- |
| 固定权重 | [官方 CLIP B/16](https://huggingface.co/openai/clip-vit-base-patch16) 的 revision `57c216476eefef5ab752ec549e440a49ae4ae5f3` 已下载，并通过 SHA-256 与离线加载核验；Qwen 0.5B 沿用本地权重 |
| 配置/真实 batch | 196 个视觉 token；dry-run 的文本 `(2,240)`、视觉 `(2,196,768)`，有限 loss 1.3195，Projector 可训练参数 3,410,816 |
| 100 样本门禁 | 300 步，loss 1.4932→0.0724，token 准确率 99%，自由生成 7/8，换图变化 3/8，8 项断言全过；峰值显存 5,368 MB |
| 1k 短训练 | 50 步验证损失 0.4558→0.3707，峰值显存 5,310 MB，训练 222.6 秒 |
| 输入传输 | 已生成 `outputs/cloud_b16_spatial_clean_input.zip`（约 2.17 GB）及同名 `.sha256`；含 6,400 张无文件名冲突的图片、9,333 条清理数据、原始 image_id 划分、固定权重、最小代码与 Linux 启动脚本；ZIP CRC 已逐项验证；ZIP SHA-256 `af5345c400f63620b129ba4fe8d1a4987ac83466a95853718823406b23154ca7` |
| 正式 B/16 训练与 test | **已完成**。云端 1,200 步，最佳 val_loss 0.1034；test 374/489，空间原图 37/76、错图 46/76；详见 EXP-009 |

本机 6 GB 显存的余量较小；当前约 6.5 GB 空闲内存，B/16 的 6,400 张预编码视觉特征
按 float32 估算约 3.8 GB。按本机 50 步的 4.45 秒/步线性外推，正式实验把累积从 4 提到 8、
训练 1,200 步约需 3 小时，尚不含预编码和评测。这只是本机粗估，不能直接预测云端速度。
本机没有 WSL2，Linux 环境尚未实跑；云端脚本因此先检查全部输入并做 dry-run、50 步 smoke，
验证损失不下降就停止，不直接消耗完整训练时长。正式训练最多运行 210 分钟，失败或中断也会
把已有日志与 checkpoint 复制到指定的持久目录。

## 平台建议与费用

优先建议 AutoDL **1 张 RTX 4090 24 GB、按量计费**，选择系统内存至少 16 GB、
可用磁盘至少 20 GB 的 Linux 实例。[官网当前标价](https://www.autodl.com/)为
**¥1.88/小时**；若按计划书的单次 4 小时目标关机，GPU 费用约 **¥7.52**，
还应以创建实例页面的现价、机器库存及磁盘费用为准。AutoDL 的
[计费说明](https://www.autodl.com/docs/price/)写明开机开始计费、关机结束计费；
[文件存储说明](https://www.autodl.com/docs/fs/)写明 20 GB 以下免费，足够预先上传训练包，
避免在付费 GPU 开机期间等待 2.17 GB 上传。与美元计费的 Runpod 相比，
本项目首次上云采用国内平台、人民币计费和现成文件存储更直接；
[Runpod 官方价目](https://www.runpod.io/pricing)可作备选。

2026-09-24 已完成一次按量计费 RTX 4090 训练，实际费用 ¥3.82，已人工关机；
任务退出码为 0 后实例并未自动关机，后续必须在控制台确认关机状态。

## AutoDL 执行顺序（准备好账号后）

1. 在选定地区初始化文件存储，关机状态下上传本地的 ZIP 与 `.sha256`；
   [官方上传指引](https://www.autodl.com/docs/scp/)也支持 SCP 或 FileZilla。
2. 创建 1 张 RTX 4090 的按量实例，核对 24 GB 显存、至少 16 GB 内存、至少 20 GB 空闲盘，
   选兼容 CUDA 12.4 且可用 Python 3.12 的 Linux 环境。训练中记下实例实际单价与开机时间。
3. 将训练包从文件存储复制到实例本地盘，运行 `sha256sum -c`，再解压。创建 Python 虚拟环境。
   按 [PyTorch 官方旧版安装命令](https://docs.pytorch.org/get-started/previous-versions/)安装
   `torch==2.6.0 torchvision==0.21.0` 的 CUDA 12.4 wheel，然后安装 `requirements-cloud.txt`。
4. 设 `PYTHON_BIN` 为该虚拟环境的 Python，设 `MINIVLM_PERSIST_ROOT` 为文件存储中的结果目录，
   运行 `bash scripts/run_cloud_b16.sh`。脚本会离线校验输入、运行 25 项单测、真实 dry-run、
   50 步 smoke，并在验证损失下降后才开始完整训练，最后评测全部 489 条 test 和错图对照。
5. 核对 `outputs/b16_spatial_clean_cloud_eval/eval_test_per_task.json`、日志、
   `checkpoint_best.pt`、`checkpoint_last.pt`，从文件存储下载到本地并记录实际花费，然后关机。

解压后在项目根目录执行的命令示例（文件存储地区和实例路径以实际页面为准）：

```bash
sha256sum -c cloud_b16_spatial_clean_input.zip.sha256
unzip -q cloud_b16_spatial_clean_input.zip
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-cloud.txt
export PYTHON_BIN="$PWD/.venv/bin/python"
export MINIVLM_PERSIST_ROOT=/root/autodl-fs/mini-vlm-b16-results
bash scripts/run_cloud_b16.sh
```

该环境配方来自 Windows 本地通过门禁的版本组合和官方 Linux wheel 安装命令；
只有云端 dry-run 与 smoke 实际通过后，才算 Linux 环境验证完成。
