from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from anila.checkpoint import load_checkpoint_payload
from anila.config import LoRAConfig, ModelConfig, RewardConfig
from anila.data import IGNORE_INDEX
from anila.model import AnilaLM
from anila.peft import apply_lora


@dataclass
class RewardModelOutput:
    scores: torch.Tensor


@dataclass
class RewardModelLoss:
    loss: torch.Tensor
    chosen_scores: torch.Tensor
    rejected_scores: torch.Tensor
    margin: torch.Tensor
    accuracy: torch.Tensor


class RewardModel(nn.Module):
    def __init__(self, backbone: AnilaLM):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.config.n_embd, 1)
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> RewardModelOutput:
        output = self.backbone(input_ids, return_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError("backbone did not return hidden states")
        pooled = last_response_hidden_state(output.hidden_states, labels)
        return RewardModelOutput(scores=self.reward_head(pooled).squeeze(-1))


class RewardScorer(Protocol):
    def score(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        responses: list[str],
        targets: list[str | None],
    ) -> torch.Tensor: ...


class RuleRewardScorer:
    def __init__(self, reward_type: str):
        self.reward_type = reward_type

    def score(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        responses: list[str],
        targets: list[str | None],
    ) -> torch.Tensor:
        del input_ids, labels
        if len(responses) != len(targets):
            raise ValueError("responses and targets must have matching lengths")
        rewards = []
        for response, target in zip(responses, targets, strict=True):
            if target is None:
                raise ValueError("rule reward requires an expected target in each prompt record")
            rewards.append(score_response(response, target, self.reward_type))
        return torch.tensor(rewards, dtype=torch.float32)


class LearnedRewardScorer:
    def __init__(self, checkpoint: str | Path, device: torch.device, *, scale: float = 1.0, bias: float = 0.0):
        self.device = device
        self.scale = scale
        self.bias = bias
        self.model = load_reward_model(checkpoint, device)

    @torch.no_grad()
    def score(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        responses: list[str],
        targets: list[str | None],
    ) -> torch.Tensor:
        del responses, targets
        was_training = self.model.training
        self.model.eval()
        try:
            scores = self.model(input_ids.to(self.device), labels.to(self.device)).scores.float()
        finally:
            if was_training:
                self.model.train()
        return scores.mul(self.scale).add(self.bias)


def score_response(response: str, expected: str, reward_type: str) -> float:
    if reward_type == "contains":
        return float(expected.strip().casefold() in response.casefold())
    if reward_type == "exact_match":
        return float(response.strip().casefold() == expected.strip().casefold())
    raise ValueError("reward_type must be contains or exact_match")


def last_response_hidden_state(hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [batch, seq, dim], got {tuple(hidden_states.shape)}")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError(
            f"hidden_states and labels must share batch/sequence shape, got {hidden_states.shape[:2]} and {labels.shape}"
        )
    mask = labels.ne(IGNORE_INDEX)
    if mask.ndim != 2:
        raise ValueError("labels must have shape [batch, seq]")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("reward model batch contains a sequence with no trainable response tokens")
    positions = mask.size(1) - 1 - mask.flip(dims=[1]).to(dtype=torch.long).argmax(dim=1)
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), positions]


def reward_model_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> RewardModelLoss:
    if chosen_scores.shape != rejected_scores.shape:
        raise ValueError("chosen_scores and rejected_scores must have matching shapes")
    margin = chosen_scores - rejected_scores
    loss = -F.logsigmoid(margin).mean()
    return RewardModelLoss(
        loss=loss,
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        margin=margin.mean(),
        accuracy=(margin > 0).to(dtype=chosen_scores.dtype).mean(),
    )


def build_reward_scorer(config: RewardConfig, *, reward_type: str, device: torch.device) -> RewardScorer:
    cfg = config.validated()
    if cfg.scorer == "rule":
        return RuleRewardScorer(reward_type)
    if cfg.checkpoint is None:
        raise ValueError("reward.checkpoint is required when reward.scorer is model")
    return LearnedRewardScorer(cfg.checkpoint, device, scale=cfg.scale, bias=cfg.bias)


def load_reward_model(checkpoint: str | Path, device: torch.device) -> RewardModel:
    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config", "reward_head"))
    if payload["reward_head"] is None:
        raise ValueError(f"Reward checkpoint is missing reward model payload: {checkpoint}")
    backbone = AnilaLM(_model_config_from_payload(payload))
    lora_config = payload.get("lora_config")
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        apply_lora(backbone, LoRAConfig(**lora_config).validated())
    model = RewardModel(backbone)
    model.backbone.load_state_dict(payload["model"])
    model.reward_head.load_state_dict(payload["reward_head"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _model_config_from_payload(payload: dict) -> ModelConfig:
    values = payload["model_config"]
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed}).validated()
