import torch.nn as nn
from transformers import CLIPVisionModel
from transformers.utils import logging as hf_logging

# CLIP 官方仓库同时包含 text / tf / flax 权重，视觉塔只需要 vision_model.* 部分，
# 因此每次加载都会打印一大张 "UNEXPECTED" 报告。它只是噪音、不是错误，
# 但会把训练日志淹没，所以这里把 transformers 日志级别压到 error。
hf_logging.set_verbosity_error()


class VisionEncoder(nn.Module):
    """VLM 的视觉塔（默认冻结）。

    输入: pixel_values [B, 3, H, W]
    输出: features     [B, N, hidden_size]

    关于 CLS token（本地实测结论，见 scripts/verify_offline.py）:
        CLIPVisionModel 的 last_hidden_state 形状为 [B, 1 + num_patches, D]，
        第 0 个 token 是 CLS（它再经过 pooler 得到 pooler_output）。
        因此丢弃 CLS 必须显式切片 out[:, 1:, :]。
        这里做自动探测，避免"以为要丢 / 以为不用丢"造成的 off-by-one：
        只有当 token 数确实多出 1 个时才切片，否则原样返回。

    off-by-one（49/50）是 VLM 里最隐蔽的 bug 来源，
    所以 num_tokens 必须与 forward 的实际输出保持一致。
    """

    def __init__(self, model_name_or_path: str, freeze: bool = True, drop_cls: bool = True):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name_or_path)
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        cfg = self.model.config
        self.hidden_size = cfg.hidden_size                            # CLIP ViT-B/32 -> 768
        self.drop_cls = drop_cls
        self.patch_tokens = (cfg.image_size // cfg.patch_size) ** 2   # 224/32 -> 49
        self.num_tokens = self.patch_tokens + (0 if drop_cls else 1)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values).last_hidden_state
        # 自动探测：只有确实多出 1 个 token（CLS）时才丢弃第 0 个
        if self.drop_cls and out.shape[1] == self.patch_tokens + 1:
            out = out[:, 1:, :]
        return out
