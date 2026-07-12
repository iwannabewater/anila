from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anila.config import PPOConfig
from anila.data import IGNORE_INDEX
from anila.model import AnilaLM


@dataclass
class PolicyValueOutput:
    logits: torch.Tensor
    values: torch.Tensor
    aux_loss: torch.Tensor | None = None


@dataclass
class PPOLoss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


class PolicyValueModel(nn.Module):
    def __init__(self, policy: AnilaLM):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Linear(policy.config.n_embd, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids: torch.Tensor) -> PolicyValueOutput:
        output = self.policy(input_ids, return_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("policy did not return hidden states")
        return PolicyValueOutput(
            logits=output.logits,
            values=self.value_head(output.hidden_states).squeeze(-1),
            aux_loss=output.aux_loss,
        )

    def generate(self, *args, **kwargs) -> torch.Tensor:
        return self.policy.generate(*args, **kwargs)


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits and labels must share batch/sequence shape, got {logits.shape[:2]} and {labels.shape}")
    mask = labels.ne(IGNORE_INDEX)
    safe_labels = labels.masked_fill(~mask, 0)
    gathered = F.log_softmax(logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return gathered.masked_fill(~mask, 0.0)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have matching shapes, got {values.shape} and {mask.shape}")
    mask_f = mask.to(dtype=values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(eps)


def normalize_masked(values: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    count = mask_f.sum()
    if count <= 1:
        return values.masked_fill(~mask, 0.0)
    mean = (values * mask_f).sum() / count
    var = ((values - mean).pow(2) * mask_f).sum() / count
    return ((values - mean) / torch.sqrt(var + eps)).masked_fill(~mask, 0.0)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (rewards.shape == values.shape == mask.shape):
        raise ValueError("rewards, values, and mask must have matching shapes")
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.size(0), dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.size(1) - 1, -1, -1):
        current_mask = mask[:, index].to(dtype=rewards.dtype)
        if index + 1 < rewards.size(1):
            next_mask = mask[:, index + 1].to(dtype=rewards.dtype)
            next_values = values[:, index + 1] * next_mask
        else:
            next_mask = torch.zeros_like(current_mask)
            next_values = torch.zeros_like(current_mask)
        delta = rewards[:, index] + gamma * next_values - values[:, index]
        last_advantage = (delta + gamma * gae_lambda * next_mask * last_advantage) * current_mask
        advantages[:, index] = last_advantage
    returns = (advantages + values).masked_fill(~mask, 0.0)
    return advantages.masked_fill(~mask, 0.0), returns


def ppo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    mask: torch.Tensor,
    config: PPOConfig,
) -> PPOLoss:
    cfg = config.validated()
    for name, tensor in {
        "policy_logprobs": policy_logprobs,
        "old_logprobs": old_logprobs,
        "values": values,
        "old_values": old_values,
        "returns": returns,
        "advantages": advantages,
        "entropy": entropy,
    }.items():
        if tensor.shape != mask.shape:
            raise ValueError(f"{name} and mask must have matching shapes")

    ratio = torch.exp(policy_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    policy_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -masked_mean(policy_objective, mask)

    clipped_values = old_values + (values - old_values).clamp(-cfg.value_clip_range, cfg.value_clip_range)
    value_loss_unclipped = (values - returns).pow(2)
    value_loss_clipped = (clipped_values - returns).pow(2)
    value_loss = 0.5 * masked_mean(torch.maximum(value_loss_unclipped, value_loss_clipped), mask)

    mean_entropy = masked_mean(entropy, mask)
    approx_kl = 0.5 * masked_mean((policy_logprobs - old_logprobs).pow(2), mask)
    clip_fraction = masked_mean((ratio.sub(1.0).abs() > cfg.clip_range).to(dtype=policy_logprobs.dtype), mask)
    loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * mean_entropy
    return PPOLoss(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=mean_entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )
