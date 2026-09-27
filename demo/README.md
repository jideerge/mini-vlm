# Mini-VLM 本地实验 demo

从项目目录使用已有虚拟环境启动：

```powershell
..\venv\Scripts\python.exe -B demo\app.py
```

在浏览器打开 <http://127.0.0.1:7860>。首次提问时会加载本地 B/16 原始最佳权重，通常比后续提问慢。使用 `Ctrl+C` 停止。若 7860 端口已占用，可加 `--port 7861`；默认自动使用本地 CUDA，必要时可加 `--device cpu`（会明显变慢）。服务器只监听 `127.0.0.1`，上传的图片仅在内存中处理，不保存、不传到云端，也不调用独立 400 题评测集。

从公开仓库克隆后，按根目录 `README.md` 安装 PyTorch 和 `requirements-demo.txt`，再运行 `python -m scripts.download_demo_models` 获取并校验固定版本的 CLIP 与 Qwen。Projector 权重已打包在 `demo/assets/checkpoint_best.pt`；旧本地目录 `outputs/b16_spatial_clean_cloud/checkpoint_best.pt` 仅作为兼容回退。该 demo 使用 Python 标准库提供网页服务；不必另外安装 Gradio。

上传 JPEG、PNG 或 WebP 图片，选择一项任务：

| 界面任务 | 模型实际收到的问句 | 类别选择 |
| --- | --- | --- |
| 是否存在 | `图中是否有{类别}？` | 一个 VOC 类别 |
| 数量 | `图中一共有几个{类别}？` | 一个 VOC 类别 |
| 属性（主要物体类别） | `图片主要是什么？` | 无；不支持颜色、材质等属性 |
| 物体列举 | `图中有什么物体？` | 无；按训练涉及的 VOC 20 类理解 |
| 空间位置 | `{主体}在{参照物}的左边还是右边？` | 两个不同 VOC 类别 |

输出是原模型的贪心生成文本（最多 24 token），不附加未经校准的置信度。**所有结果均为实验性结果。空间位置暂不支持可靠判断；多个同类物体的计数尤其容易出错。** 旧验证集 **324/399** 仅为开发阶段成绩；独立 400 题尚未评测，不能将其视为任意上传照片的准确率。请自行核对输出。

本 demo 不使用后续四任务续训权重，因为该续训未达到预先设定的旧验证门槛。见 `report/experiments.md` 的 EXP-019。
