from __future__ import annotations

from dataclasses import dataclass

import torch

from anila.config import GRPOConfig
from anila.reward import score_response as score_response


@dataclass
class GRPOLoss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    mean_reward: torch.Tensor
    reward_std: torch.Tensor


def group_advantages(rewards: torch.Tensor, *, normalize: bool = True, eps: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [batch, group], got {tuple(rewards.shape)}")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not normalize:
        return centered
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    scale = torch.where(std > eps, std, torch.ones_like(std))
    return centered / scale


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    config: GRPOConfig,
) -> GRPOLoss:
    cfg = config.validated()
    for name, tensor in {
        "policy_logps": policy_logps,
        "old_logps": old_logps,
        "reference_logps": reference_logps,
        "advantages": advantages,
    }.items():
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape [batch * group], got {tuple(tensor.shape)}")
    if not (policy_logps.shape == old_logps.shape == reference_logps.shape == advantages.shape):
        raise ValueError("policy, old, reference logprobs, and advantages must have matching shapes")

    ratio = torch.exp(policy_logps - old_logps)
    clipped_ratio = ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()

    reference_policy_logratio = reference_logps - policy_logps
    kl = torch.exp(reference_policy_logratio) - 1.0 - reference_policy_logratio
    kl_loss = kl.mean()
    loss = policy_loss + cfg.beta * kl_loss

    return GRPOLoss(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_reward=rewards.mean(),
        reward_std=rewards.std(unbiased=False),
    )
