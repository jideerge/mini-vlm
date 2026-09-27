# 公开推理网站部署

本目录是 Hugging Face Gradio/ZeroGPU Space 的源文件。`scripts/build_hf_space_bundle.py` 从项目中只挑出推理必需的文件，生成忽略 Git 的 `dist/hf_space/`；不会打包 VOC 图片、训练中间产物、独立评测集或上游大模型权重。Space 首次启动时下载固定版本的官方 CLIP/Qwen 文件并核对权重散列。

先在项目根目录运行：

```powershell
python -m scripts.build_hf_space_bundle
```

拥有符合 ZeroGPU 条件的 Hugging Face 账号后，在本机运行 `hf auth login`，然后执行：

```powershell
python -m scripts.publish_hf_space
```

脚本只请求免费 ZeroGPU 硬件；若账号不符合资格或没有容量，它会失败，不会自动换成付费 GPU。创建完成后查看 Space 构建日志，确认状态为 Running，再上传一张非敏感测试图片核对真实推理。首次启动会下载约 1.3 GB 的官方权重。免费 GPU 配额、排队与冷启动可能限制访客体验。

Hugging Face 当前要求免费个人账号验证邮箱且账号注册超过 30 天，才可创建免费 ZeroGPU Space（最多两个）；PRO 账号也可创建。CPU Basic 虽然没有每小时费用，但新建 Gradio/Docker 计算型 Space 同样需要付费计划。若创建返回 HTTP 402，当前账号不具备免费资格；请等待资格生效、申请社区资助或自行决定是否订阅 PRO，不要在脚本中切换到按小时付费 GPU。

公开网站保留本地 demo 的任务范围与风险提示。网站收到图片进行推理，和本机 demo 的“图片不离开本机”不同；请勿上传敏感图片。独立 400 题未评测，旧验证成绩不是网站对任意照片的准确率。
