# Mini-VLM：CLIP B/16 + Qwen2.5-0.5B 的实验性图片问答

这是一个从数据生成、训练、诊断到本地演示的学习项目。冻结的 [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16) 提取 196 个视觉 token；训练得到的两层 Projector 将它们映射到冻结的 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。本仓库仅包含 **VLM 项目**的代码、实验记录和约 13 MB 的 Projector checkpoint，不包含相邻项目、VOC 图片、原始训练输出或约 1.3 GB 的上游模型权重。

项目训练的问题范围是 PASCAL VOC 2007 的 20 个物体类别：是否存在、数量、属性（训练时实际问“图片主要是什么？”，即主要物体类别）、物体列举和左右空间位置。**所有 demo 输出都是实验性结果。空间位置暂不支持可靠判断；多个同类物体的计数尤其容易出错。** 旧验证集四项任务的 324/399 是开发阶段成绩；独立 400 题已封存但尚未运行模型，不能将 324/399 当作任意照片的准确率。

## 在本机运行 demo

建议使用 Python 3.12。先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装适合本机 CPU/CUDA 的 `torch` 与 `torchvision`，再在项目根目录运行：

```powershell
python -m pip install -r requirements-demo.txt
python -m scripts.download_demo_models
python demo/app.py
```

打开 <http://127.0.0.1:7860>。首次下载固定版本的 CLIP 和 Qwen 权重需要约 1.3 GB 磁盘与网络流量，脚本会检查权重 SHA-256；以后的本地推理不上传图片。网页服务只监听 `127.0.0.1`，上传图片仅在内存中处理。停止服务按 `Ctrl+C`。已有本项目虚拟环境时可直接用 `..\venv\Scripts\python.exe -B demo\app.py`。详细操作与各任务原问句见 [demo/README.md](demo/README.md)。

`demo/assets/checkpoint_best.pt` 是唯一打包的训练结果，SHA-256 为 `2a4ce406e54cdb4798b3afa1af3ef2dd7ed59ebd62a7b65b5921e7f155c6d575`，训练步数 840。公开 demo 使用这份原始 B/16 最佳权重；后续四任务续训没有通过事先设定的旧验证门槛，因此未替换它。

## 仓库内容

| 路径 | 作用 |
| --- | --- |
| `models/`、`training/`、`datasets/`、`evaluation/` | 模型、训练循环、VOC 问答生成与评分 |
| `demo/` | 本地上传图片网页、离线推理、Projector checkpoint |
| `scripts/`、`configs/`、`tests/` | 实验脚本、模型配置与测试 |
| `report/`、`notes/` | 实验记录与学习笔记 |
| `data/`、`outputs/`、`checkpoints/` | 本地大文件目录，均不提交到 GitHub |

训练数据使用 [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)；需自行按数据集条款取得。历史数据脚本可用 `MINIVLM_VOC_ROOT` 指向含 `VOCtrainval_06-Nov-2007`、`VOCtest_06-Nov-2007` 的目录。仓库中的实验记录保留原开发环境路径作为历史信息，复现时应改为本机路径。代码与成绩的细节见 [实验记录](report/experiments.md)及[执行追踪表](Mini-VLM执行追踪表.md)。

公开网页的部署文件位于 `deploy/hf_space/`。网站运行模型需要计算资源；GitHub Pages 只能托管静态页面，无法直接运行该 Python 推理服务。部署计划与当前状态见 [部署说明](deploy/hf_space/README.md)。
