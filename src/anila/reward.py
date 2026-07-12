from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from anila._json import loads_strict_json
from anila.chat import parse_assistant_message
from anila.checkpoint import load_checkpoint_payload
from anila.config import LoRAConfig, ModelConfig, RewardConfig, SFTConfig
from anila.data import IGNORE_INDEX
from anila.model import AnilaLM
from anila.peft import apply_lora


@dataclass
class RewardModelOutput:
    scores: torch.Tensor
    aux_loss: torch.Tensor | None = None


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
        return RewardModelOutput(scores=self.reward_head(pooled).squeeze(-1), aux_loss=output.aux_loss)


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
    def __init__(self, reward_type: str, sft_config: SFTConfig | None = None):
        self.reward_type = reward_type
        self.sft_config = (sft_config or SFTConfig()).validated()

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
            rewards.append(score_response(response, target, self.reward_type, sft_config=self.sft_config))
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


def score_response(response: str, expected: str, reward_type: str, *, sft_config: SFTConfig | None = None) -> float:
    if reward_type == "contains":
        return float(expected.strip().casefold() in response.casefold())
    if reward_type == "exact_match":
        return float(response.strip().casefold() == expected.strip().casefold())
    if reward_type == "tool_call":
        return score_tool_call_response(response, expected, sft_config=sft_config)
    raise ValueError("reward_type must be contains, exact_match, or tool_call")


def score_tool_call_response(response: str, expected: str, *, sft_config: SFTConfig | None = None) -> float:
    cfg = (sft_config or SFTConfig()).validated()
    spec = _tool_reward_spec(expected)
    parsed = _parse_tool_call_response(response, cfg)
    calls = list(parsed.tool_calls)
    score = 0.0

    tag_gap = abs(response.count(cfg.tool_call_start) - response.count(cfg.tool_call_end))
    score -= 0.5 * tag_gap
    score -= 0.5 * len(parsed.invalid_tool_calls)

    thinking = parsed.reasoning_content
    if thinking is not None:
        score += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        score += 0.25 if response.count(cfg.thinking_end) == 1 else -0.25

    expected_tools = spec["tools"]
    missing_tools, extra_tools, malformed_tools, unexpected_tools = _tool_match_counts(calls, expected_tools)
    score -= 0.5 * malformed_tools
    if expected_tools:
        tool_gap = missing_tools + extra_tools + unexpected_tools
        score += 0.5 if tool_gap == 0 and malformed_tools == 0 else -0.5 * tool_gap
    elif calls:
        tool_gap = unexpected_tools
        score -= 0.5 * tool_gap
    else:
        tool_gap = 0

    final_text = _final_answer_text(response, parsed.content, config=cfg)
    answers = spec["answers"]
    if answers:
        verified = sum(_answer_in_text(answer, final_text) for answer in answers)
        missing_answers = len(answers) - verified
        score += 2.5 * verified / len(answers)
        if missing_answers:
            score -= missing_answers / len(answers)
    else:
        score += 0.5 if 5 <= len(final_text.strip()) <= 800 else -0.5
    score -= _repetition_penalty(final_text or response)
    if tag_gap or parsed.invalid_tool_calls or malformed_tools or tool_gap:
        score = min(score, 0.0)
    return max(min(score, 3.0), -3.0)


def tool_call_response_succeeds(response: str, expected: str, *, sft_config: SFTConfig | None = None) -> bool:
    cfg = (sft_config or SFTConfig()).validated()
    spec = _tool_reward_spec(expected)
    parsed = _parse_tool_call_response(response, cfg)
    if response.count(cfg.tool_call_start) != response.count(cfg.tool_call_end) or parsed.invalid_tool_calls:
        return False
    if any(_tool_match_counts(parsed.tool_calls, spec["tools"])):
        return False
    answers = spec["answers"]
    if not answers:
        return True
    final_text = _final_answer_text(response, parsed.content, config=cfg)
    return all(_answer_in_text(answer, final_text) for answer in answers)


def _parse_tool_call_response(response: str, config: SFTConfig):
    started_thinking = config.thinking_end in response and config.thinking_start not in response
    return parse_assistant_message(response, started_thinking=started_thinking, config=config)


def _tool_reward_spec(expected: str) -> dict[str, tuple[str, ...]]:
    expected = expected.strip()
    try:
        parsed = loads_strict_json(expected)
    except ValueError:
        return {"answers": (expected,), "tools": ()}
    if isinstance(parsed, dict):
        answers = _string_tuple(
            parsed.get("answers", parsed.get("answer", parsed.get("expected", parsed.get("targets", ()))))
        )
        tools = _string_tuple(parsed.get("tools", parsed.get("tool_names", parsed.get("required_tools", ()))))
        return {"answers": answers, "tools": tools}
    if isinstance(parsed, list):
        return {"answers": _string_tuple(parsed), "tools": ()}
    return {"answers": _string_tuple(parsed), "tools": ()}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list | tuple) else [value]
    out: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


def _tool_match_counts(calls: list[dict[str, Any]] | tuple[dict[str, Any], ...], expected_tools: tuple[str, ...]) -> tuple[int, int, int, int]:
    expected_counts = Counter(expected_tools)
    actual_counts: Counter[str] = Counter()
    malformed_count = 0
    unexpected_count = 0
    for call in calls:
        name, arguments, argument_error = _tool_call_name_and_arguments(call)
        if argument_error is not None or not name or not isinstance(arguments, dict):
            malformed_count += 1
            continue
        if not expected_counts:
            unexpected_count += 1
        elif name in expected_counts:
            actual_counts[name] += 1
        else:
            unexpected_count += 1
    missing_count = sum((expected_counts - actual_counts).values())
    extra_count = sum((actual_counts - expected_counts).values())
    return missing_count, extra_count, malformed_count, unexpected_count


def _tool_call_name_and_arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    if not isinstance(call, dict):
        return "", {}, "tool call must be an object"
    if isinstance(call.get("function"), dict):
        function = call["function"]
        raw_name = function.get("name", "")
        raw_arguments = function.get("arguments", {})
    else:
        raw_name = call.get("name", "")
        raw_arguments = call.get("arguments", {})
    name = raw_name if isinstance(raw_name, str) else ""
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = loads_strict_json(raw_arguments) if raw_arguments else {}
        except ValueError:
            return name, {}, "tool call arguments must be valid JSON"
    if not isinstance(raw_arguments, dict):
        return name, {}, "tool call arguments must be a JSON object"
    return name, raw_arguments, None


def _final_answer_text(response: str, parsed_content: str, *, config: SFTConfig) -> str:
    if config.tool_call_end in response:
        return response.rsplit(config.tool_call_end, maxsplit=1)[-1].strip()
    return parsed_content.strip()


def _answer_in_text(answer: str, text: str) -> bool:
    answer = answer.strip()
    if not answer:
        return False
    normalized_number = answer.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized_number):
        target = float(normalized_number)
        for match in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w)", text.replace(",", "")):
            if abs(float(match) - target) < 1e-6:
                return True
        return False
    return answer.casefold() in text.casefold()


def _repetition_penalty(text: str, *, n: int = 3, cap: float = 0.5) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


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


def build_reward_scorer(
    config: RewardConfig,
    *,
    reward_type: str,
    device: torch.device,
    sft_config: SFTConfig | None = None,
) -> RewardScorer:
    cfg = config.validated()
    if cfg.scorer == "rule":
        return RuleRewardScorer(reward_type, sft_config=sft_config)
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
