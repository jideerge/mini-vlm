---
title: Mini-VLM 实验性图片问答
emoji: 🖼️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
python_version: 3.12
pinned: false
---

# Mini-VLM 实验性图片问答

本 Space 展示一个 CLIP ViT-B/16 + Qwen2.5-0.5B-Instruct 的学习项目。训练得到的 Projector 随代码提供；官方底座权重在首次启动时从固定版本下载并核对 SHA-256。

仅供实验观察。空间位置暂不支持可靠判断，多个同类物体的计数容易出错。旧验证集的 324/399 是开发阶段成绩；独立 400 题尚未评测，不代表任意图片的准确率。上传图片会发送到托管服务进行推理，请勿上传敏感图片。

项目源码与方法说明见 GitHub：<https://github.com/jideerge/mini-vlm>。
