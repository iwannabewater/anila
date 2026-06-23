from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from anila.checkpoint import load_checkpoint_payload
from anila.config import DistillConfig, LoRAConfig, ModelConfig, OPDConfig
from anila.data import IGNORE_INDEX
from anila.model import AnilaLM
from anila.peft import apply_lora


@dataclass
class DistillationLoss:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    kl_loss: torch.Tensor


def soft_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    config: DistillConfig,
) -> DistillationLoss:
    cfg = config.validated()
    return _masked_distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=cfg.temperature,
        kl_weight=cfg.kl_weight,
        ce_weight=cfg.ce_weight,
    )


def on_policy_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    config: OPDConfig,
) -> DistillationLoss:
    cfg = config.validated()
    return _masked_distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=cfg.distill_temperature,
        kl_weight=cfg.kl_weight,
        ce_weight=cfg.ce_weight,
    )


def _masked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    kl_weight: float,
    ce_weight: float,
) -> DistillationLoss:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student and teacher logits must have identical shape, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    mask = labels.ne(IGNORE_INDEX)
    valid = mask.sum()
    if int(valid.item()) == 0:
        raise ValueError("distillation batch has no trainable labels")

    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    per_token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temperature**2)
    kl_loss = per_token_kl.masked_select(mask).mean()
    loss = (ce_weight * ce_loss) + (kl_weight * kl_loss)
    return DistillationLoss(loss=loss, ce_loss=ce_loss, kl_loss=kl_loss)


def load_teacher_model(checkpoint: str | Path, device: torch.device) -> AnilaLM:
    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config"))
    model = AnilaLM(_model_config_from_payload(payload))
    lora_config = payload.get("lora_config")
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        apply_lora(model, LoRAConfig(**lora_config).validated())
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _model_config_from_payload(payload: dict) -> ModelConfig:
    values = payload["model_config"]
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed}).validated()
