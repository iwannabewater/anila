from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from anila.config import DPOConfig
from anila.data import IGNORE_INDEX


@dataclass
class DPOLoss:
    loss: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    preference_margin: torch.Tensor


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits and labels must share batch/sequence shape, got {logits.shape[:2]} and {labels.shape}")
    mask = labels.ne(IGNORE_INDEX)
    if mask.ndim != 2:
        raise ValueError("labels must have shape [batch, seq]")
    safe_labels = labels.masked_fill(~mask, 0)
    token_logprobs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.masked_fill(~mask, 0.0).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    config: DPOConfig,
) -> DPOLoss:
    cfg = config.validated()
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = policy_logratios - reference_logratios
    losses = -F.logsigmoid(cfg.beta * logits)
    chosen_rewards = cfg.beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = cfg.beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return DPOLoss(
        loss=losses.mean(),
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        preference_margin=(chosen_rewards - rejected_rewards).mean(),
    )
