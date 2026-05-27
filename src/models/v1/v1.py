from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.vit import VisionTransformer, _build_vit_backbone


def build_student_teacher(
    model_cfg: DictConfig,
) -> tuple[VisionTransformer, VisionTransformer]:
    student = _build_vit_backbone(model_cfg)
    teacher = _build_vit_backbone(model_cfg)

    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


_HF_TO_TORCHHUB: dict[str, str] = {
    "facebook/dino-vits16": "dino_vits16",
    "facebook/dino-vits8": "dino_vits8",
    "facebook/dino-vitb16": "dino_vitb16",
    "facebook/dino-vitb8": "dino_vitb8",
}


def _hf_to_torchhub(hub_model: str) -> str:
    return _HF_TO_TORCHHUB.get(hub_model, "dino_vits16")


def load_dinov1(cfg: DictConfig, device: torch.device) -> nn.Module:
    hub_model = cfg.inference.hub_model
    # weights_path = cfg.inference.get("weights", None)

    # if weights_path:
    #     model = _build_vit_backbone(cfg.model)
    #     state = torch.load(weights_path, map_location="cpu")

    #     state = state.get("teacher", state.get("student", state))
    #     state = {k.replace("backbone.", ""): v for k, v in state.items()}
    #     model.load_state_dict(state, strict=False)
    # else:
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(hub_model, attn_implementation="eager")
    except Exception:
        hub_name = _hf_to_torchhub(hub_model)
        model = torch.hub.load("facebookresearch/dino:main", hub_name, pretrained=True)
    return model.to(device).eval()
