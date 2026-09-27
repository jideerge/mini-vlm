"""Offline, single-image inference for the frozen B/16 Mini-VLM checkpoint.

Keep the training prompt, square image padding, CLIP processor, expanded image
placeholders, and answer-start position identical to the evaluated model path.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datasets.build_voc_vqa import CLASS_ZH  # noqa: E402
from datasets.voc_source import VOC_CLASSES  # noqa: E402


CLASSES = [{"value": name, "label": CLASS_ZH[name]} for name in VOC_CLASSES]
BUNDLED_CHECKPOINT = ROOT / "demo/assets/checkpoint_best.pt"
LOCAL_CHECKPOINT = ROOT / "outputs/b16_spatial_clean_cloud/checkpoint_best.pt"
CHECKPOINT = BUNDLED_CHECKPOINT if BUNDLED_CHECKPOINT.is_file() else LOCAL_CHECKPOINT
CHECKPOINT_SHA256 = "2a4ce406e54cdb4798b3afa1af3ef2dd7ed59ebd62a7b65b5921e7f155c6d575"
CONFIG = ROOT / "configs/model_clip_b16_qwen05.yaml"
MAX_NEW_TOKENS = 24


def build_question(task: str, category: str | None = "", other_category: str | None = "") -> str:
    """Use the exact Chinese task prompts used to train the B/16 checkpoint."""
    if task in ("existence", "counting"):
        if category not in VOC_CLASSES or other_category:
            raise ValueError("该任务需选择一个 VOC 类别。")
        name = CLASS_ZH[category]
        return (f"图中是否有{name}？" if task == "existence"
                else f"图中一共有几个{name}？")
    if task in ("attribute", "listing"):
        if category or other_category:
            raise ValueError("该任务不需要选择类别。")
        return "图片主要是什么？" if task == "attribute" else "图中有什么物体？"
    if task == "spatial":
        if (category not in VOC_CLASSES or other_category not in VOC_CLASSES
                or category == other_category):
            raise ValueError("空间问题需选择两个不同的 VOC 类别。")
        return f"{CLASS_ZH[category]}在{CLASS_ZH[other_category]}的左边还是右边？"
    raise ValueError("不支持的任务。")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalMiniVLM:
    """Hold one offline B/16 baseline in memory for repeated local predictions."""

    def __init__(self, device: str = "auto") -> None:
        import torch

        from datasets.collator import MultipleImageCollator
        from models.multimodal_model import MiniVLM, config_from_yaml

        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 只能是 auto、cpu 或 cuda。")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 推理，但本机没有可用 CUDA GPU。")
        resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"缺少 B/16 基线权重：{CHECKPOINT}")
        if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
            raise ValueError("B/16 基线权重 SHA-256 不匹配，拒绝加载。")

        cfg = config_from_yaml(CONFIG)
        if (cfg.strategy != "projector_only" or cfg.num_image_token != 196
                or cfg.projector_depth != 2 or cfg.projector_hidden != 2048
                or not cfg.freeze_vision or not cfg.freeze_llm):
            raise ValueError("当前配置不是已验证的 B/16 projector-only 架构。")
        cfg.device = resolved
        self.device = resolved
        self.model = MiniVLM(cfg)
        if self.model.vision is None or self.model.processor is None:
            raise RuntimeError("B/16 视觉编码器或 CLIP 预处理器未加载。")

        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("projector"), dict):
            raise ValueError("B/16 checkpoint 缺少 Projector 权重。")
        if payload.get("global_step") != 840:
            raise ValueError("B/16 checkpoint 的训练步数不符。")
        if payload.get("lora") or payload.get("trainable_extra"):
            raise ValueError("B/16 checkpoint 含非基线权重，拒绝部分加载。")
        self.model.projector.load_state_dict(payload["projector"], strict=True)
        self.model.eval()
        self.collator = MultipleImageCollator(
            processor=self.model.processor,
            tokenizer=self.model.tokenizer,
            max_length=cfg.max_seq_len,
            num_image_token=self.model.num_image_token,
        )

    def predict(self, image: Image.Image, question: str) -> str:
        """Generate one experimental answer; ``question`` is the raw task prompt."""
        import torch

        from datasets.dataset import JsonlVLDataset

        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL 图片。")
        if not isinstance(question, str) or not question.strip() or "<|image_pad|>" in question:
            raise ValueError("question 必须是有效的原始任务问句。")
        sample = {
            "image": image.convert("RGB"),
            "question": JsonlVLDataset.build_question(question),
            "answer": "",
        }
        with torch.inference_mode():
            batch = self.collator([sample])
            pixels = batch["pixel_values"].to(
                self.model.device_, dtype=next(self.model.vision.parameters()).dtype
            )
            features = self.model.vision(pixels)
            answer_start = torch.tensor(batch["answer_starts"], device=self.model.device_)
            answers = self.model.generate_text(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                visual_features=features,
                answer_start=answer_start,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                micro_batch=1,
            )
        if len(answers) != 1:
            raise RuntimeError("模型没有返回一条完整答案。")
        return answers[0]
