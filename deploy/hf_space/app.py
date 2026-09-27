"""Public Gradio/ZeroGPU variant of the local Mini-VLM demo.

This file is copied to the root of a Hugging Face Space by
``scripts/build_hf_space_bundle.py``. The same frozen projector, task prompts,
and upstream model hashes are used as in the local demo.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

try:
    import spaces  # Import before torch for Hugging Face ZeroGPU.
except ImportError:  # Local CPU preview outside Spaces.
    class _SpacesFallback:
        @staticmethod
        def GPU(**_kwargs):  # noqa: N802
            return lambda function: function

    spaces = _SpacesFallback()

import gradio as gr
import torch
from PIL import Image

FILE_ROOT = Path(__file__).resolve().parent
ROOT = FILE_ROOT if (FILE_ROOT / "models").is_dir() else FILE_ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_demo_models import ensure_pretrained  # noqa: E402

# Download only pinned official files when the Space starts. The 13 MB learned
# projector is part of the Space bundle and verified by LocalMiniVLM.
ensure_pretrained()

from demo.inference import CLASSES, LocalMiniVLM, build_question  # noqa: E402

MODEL: LocalMiniVLM | None = None
MODEL_LOCK = threading.Lock()
CLASS_CHOICES = [(item["label"], item["value"]) for item in CLASSES]
TASK_CHOICES = [
    ("是否存在", "existence"),
    ("数量", "counting"),
    ("属性（主要物体类别）", "attribute"),
    ("物体列举", "listing"),
    ("空间位置（不可靠）", "spatial"),
]
TASK_NOTES = {
    "existence": "选择一个训练涉及的 VOC 类别。",
    "counting": "多个同类物体的计数尤其容易出错，请自行核对。",
    "attribute": "这里的“属性”仅指图片主要物体的 VOC 类别，不判断颜色、材质。",
    "listing": "结果应仅按训练涉及的 VOC 20 类理解；范围外物体不保证识别。",
    "spatial": "空间位置暂不支持可靠判断；输出仅供观察，不能用于实际判断。",
}


def update_task(task: str):
    need_category = task in ("existence", "counting", "spatial")
    need_other = task == "spatial"
    return (
        gr.update(visible=need_category, value=None),
        gr.update(visible=need_other, value=None),
        TASK_NOTES.get(task, ""),
    )


@spaces.GPU(duration=120)
def answer(image: Image.Image | None, task: str, category: str | None,
           other_category: str | None) -> tuple[str, str, str]:
    if image is None:
        raise gr.Error("请先上传一张图片。")
    if not isinstance(image, Image.Image):
        raise gr.Error("无法读取图片。")
    if (image.width < 32 or image.height < 32 or
            image.width * image.height > 20_000_000):
        raise gr.Error("图片尺寸应至少为 32×32，且不超过 2000 万像素。")
    category = category if task in ("existence", "counting", "spatial") else None
    other_category = other_category if task == "spatial" else None
    try:
        question = build_question(task, category, other_category)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    # The Gradio runtime may keep a temporary upload during the request; do not
    # claim that a public upload stays in browser memory. Never train on it.
    image = image.copy()
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            MODEL = LocalMiniVLM(device="cpu")
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            MODEL.model.to("cuda")
            MODEL.model.device_ = torch.device("cuda")
        try:
            result = MODEL.predict(image, question).strip()
        finally:
            if use_cuda:
                MODEL.model.to("cpu")
                MODEL.model.device_ = torch.device("cpu")
                torch.cuda.empty_cache()
            image.close()
    return question, result or "模型未生成可展示的答案。", TASK_NOTES[task]


CSS = """
.gradio-container { max-width: 1050px !important; margin: auto !important; }
.hero { padding: 22px 25px; border-radius: 18px; background: #e9f3ed; }
.hero h1 { color: #116e60; margin-bottom: 8px; }
.warning { border: 1px solid #e6c897; border-radius: 12px; padding: 14px 18px;
           background: #fff8ea; color: #704b1b; }
.privacy { font-size: 0.9em; color: #596b60; }
"""


def build_demo():
    with gr.Blocks(title="Mini-VLM 实验性图片问答") as demo:
        gr.HTML("""<div class='hero'><h1>Mini-VLM · 实验性图片问答</h1>
        <p>上传一张图片，选择训练涉及的 VOC 类别与任务，查看模型的原始回答。</p></div>""")
        gr.HTML("""<div class='warning'><strong>实验性结果</strong><br>
        空间位置暂不支持可靠判断；多个同类物体的计数尤其容易出错。
        旧验证集 324/399 仅为开发阶段成绩，独立 400 题尚未评测，
        不能将其理解为任意照片的准确率。</div>""")
        gr.Markdown("公开网站会将上传图片发送至托管服务进行推理；请勿上传敏感图片。模型不会用上传图片继续训练。",
                    elem_classes="privacy")
        with gr.Row():
            with gr.Column(scale=1):
                picture = gr.Image(type="pil", label="上传图片（JPEG / PNG / WebP）",
                                   sources=["upload"])
                task = gr.Radio(choices=TASK_CHOICES, value="existence", label="问题类型")
                category = gr.Dropdown(choices=CLASS_CHOICES, value=None, label="目标类别")
                other = gr.Dropdown(choices=CLASS_CHOICES, value=None,
                                    label="参照类别", visible=False)
                note = gr.Markdown(TASK_NOTES["existence"])
                run = gr.Button("生成回答", variant="primary")
            with gr.Column(scale=1):
                question = gr.Textbox(label="实际提问", interactive=False)
                response = gr.Textbox(label="模型回答 · 实验性结果", interactive=False, lines=4)
                caution = gr.Textbox(label="此任务的限制", interactive=False, lines=3)
        task.change(update_task, inputs=task, outputs=[category, other, note])
        run.click(answer, inputs=[picture, task, category, other],
                  outputs=[question, response, caution], concurrency_limit=1)
        gr.Markdown("模型仅支持 VOC 20 类范围内的实验性问答；请结合原图核对。")
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(max_size=10).launch(css=CSS)
