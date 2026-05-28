from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from anila.checkpoint import checkpoint_model_state, load_checkpoint_payload
from anila.config import DataConfig, DPOConfig, LoRAConfig, ModelConfig, RewardConfig, SFTConfig
from anila.data import IGNORE_INDEX, create_dataloader
from anila.dpo import sequence_logprobs
from anila.model import AnilaLM
from anila.peft import apply_lora
from anila.reward import RewardModel
from anila.tokenization import AnilaTokenizer
from anila.training import resolve_device


def evaluate_lm_checkpoint(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    dataset_path: str | list[str],
    objective: str = "pretrain",
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "auto",
    sft_config: SFTConfig | None = None,
    data_config: DataConfig | None = None,
    use_ema: bool = False,
) -> dict[str, Any]:
    if objective not in {"pretrain", "sft"}:
        raise ValueError("LM evaluation objective must be pretrain or sft")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _validate_max_batches(max_batches)

    runtime_device = resolve_device(device)
    tokenizer = AnilaTokenizer.load(tokenizer_path)
    model, payload = _load_policy_model(checkpoint, runtime_device, use_ema=use_ema)
    resolved_sft_config = sft_config or _validated_config_from_payload(payload, "sft_config", SFTConfig, SFTConfig())
    resolved_data_config = data_config or _validated_config_from_payload(payload, "data_config", DataConfig, DataConfig())
    loader = create_dataloader(
        dataset_path,
        tokenizer,
        context_length=model.config.context_length,
        batch_size=batch_size,
        objective=objective,
        sft_config=resolved_sft_config,
        data_config=resolved_data_config,
        shuffle=False,
        drop_last=False,
    )

    total_nll = 0.0
    total_tokens = 0
    num_batches = 0
    with torch.no_grad():
        for batch in _limited_batches(loader, max_batches):
            input_ids, labels = _move_batch_to_device(batch, runtime_device)
            output = model(input_ids)
            nll, tokens = _sum_nll(output.logits, labels)
            total_nll += float(nll.detach().cpu())
            total_tokens += tokens
            num_batches += 1
    if total_tokens == 0:
        raise ValueError("evaluation dataset produced no trainable tokens")
    loss = total_nll / total_tokens
    return {
        "task": "lm",
        "objective": objective,
        "checkpoint": str(checkpoint),
        "weights": "ema" if use_ema else "model",
        "dataset_path": dataset_path,
        "num_batches": num_batches,
        "num_tokens": total_tokens,
        "loss": loss,
        "perplexity": _safe_exp(loss),
    }


def evaluate_policy_preferences(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    dataset_path: str | list[str],
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "auto",
    dpo_config: DPOConfig | None = None,
    use_ema: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _validate_max_batches(max_batches)

    runtime_device = resolve_device(device)
    tokenizer = AnilaTokenizer.load(tokenizer_path)
    model, payload = _load_policy_model(checkpoint, runtime_device, use_ema=use_ema)
    resolved_dpo_config = dpo_config or _validated_config_from_payload(payload, "dpo_config", DPOConfig, DPOConfig())
    loader = create_dataloader(
        dataset_path,
        tokenizer,
        context_length=model.config.context_length,
        batch_size=batch_size,
        objective="dpo",
        dpo_config=resolved_dpo_config,
        shuffle=False,
        drop_last=False,
    )

    correct = 0
    total_pairs = 0
    margin_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in _limited_batches(loader, max_batches):
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _move_batch_to_device(
                batch, runtime_device
            )
            chosen_logps = sequence_logprobs(model(chosen_input_ids).logits, chosen_labels)
            rejected_logps = sequence_logprobs(model(rejected_input_ids).logits, rejected_labels)
            margin = chosen_logps - rejected_logps
            correct += int(margin.gt(0).sum().item())
            total_pairs += int(margin.numel())
            margin_sum += float(margin.sum().detach().cpu())
            num_batches += 1
    if total_pairs == 0:
        raise ValueError("preference evaluation dataset produced no pairs")
    return {
        "task": "preference",
        "checkpoint": str(checkpoint),
        "weights": "ema" if use_ema else "model",
        "dataset_path": dataset_path,
        "num_batches": num_batches,
        "num_pairs": total_pairs,
        "accuracy": correct / total_pairs,
        "mean_margin": margin_sum / total_pairs,
    }


def evaluate_reward_model(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    dataset_path: str | list[str],
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "auto",
    reward_config: RewardConfig | None = None,
    use_ema: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _validate_max_batches(max_batches)

    runtime_device = resolve_device(device)
    tokenizer = AnilaTokenizer.load(tokenizer_path)
    model, payload = _load_reward_model(checkpoint, runtime_device, use_ema=use_ema)
    resolved_reward_config = reward_config or _validated_config_from_payload(
        payload, "reward_config", RewardConfig, RewardConfig()
    )
    loader = create_dataloader(
        dataset_path,
        tokenizer,
        context_length=model.backbone.config.context_length,
        batch_size=batch_size,
        objective="reward_model",
        reward_config=resolved_reward_config,
        shuffle=False,
        drop_last=False,
    )

    correct = 0
    total_pairs = 0
    margin_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in _limited_batches(loader, max_batches):
            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _move_batch_to_device(
                batch, runtime_device
            )
            chosen_scores = model(chosen_input_ids, chosen_labels).scores
            rejected_scores = model(rejected_input_ids, rejected_labels).scores
            margin = chosen_scores - rejected_scores
            correct += int(margin.gt(0).sum().item())
            total_pairs += int(margin.numel())
            margin_sum += float(margin.sum().detach().cpu())
            num_batches += 1
    if total_pairs == 0:
        raise ValueError("reward evaluation dataset produced no pairs")
    return {
        "task": "reward",
        "checkpoint": str(checkpoint),
        "weights": "ema" if use_ema else "model",
        "dataset_path": dataset_path,
        "num_batches": num_batches,
        "num_pairs": total_pairs,
        "accuracy": correct / total_pairs,
        "mean_margin": margin_sum / total_pairs,
    }


def _load_policy_model(checkpoint: str | Path, device: torch.device, *, use_ema: bool = False) -> tuple[AnilaLM, dict[str, Any]]:
    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config"))
    model = AnilaLM(_model_config_from_payload(payload))
    lora_config = payload.get("lora_config")
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        apply_lora(model, LoRAConfig(**lora_config).validated())
    model.load_state_dict(checkpoint_model_state(payload, use_ema=use_ema))
    model.to(device).eval()
    return model, payload


def _load_reward_model(
    checkpoint: str | Path, device: torch.device, *, use_ema: bool = False
) -> tuple[RewardModel, dict[str, Any]]:
    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config"))
    if payload.get("reward_head") is None:
        raise ValueError(f"Checkpoint does not contain a reward head: {checkpoint}")
    backbone = AnilaLM(_model_config_from_payload(payload))
    lora_config = payload.get("lora_config")
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        apply_lora(backbone, LoRAConfig(**lora_config).validated())
    backbone.load_state_dict(checkpoint_model_state(payload, use_ema=use_ema))
    model = RewardModel(backbone)
    head_key = "ema_reward_head" if use_ema else "reward_head"
    head_state = payload.get(head_key)
    if head_state is None and use_ema:
        raise ValueError("Checkpoint does not contain EMA reward head weights")
    if not isinstance(head_state, dict):
        raise ValueError(f"Checkpoint {head_key} state must be a dictionary")
    model.reward_head.load_state_dict(head_state)
    model.to(device).eval()
    return model, payload


def _model_config_from_payload(payload: dict[str, Any]) -> ModelConfig:
    values = payload["model_config"]
    if not isinstance(values, dict):
        raise ValueError("checkpoint model_config must be an object")
    return _validated_config_from_payload(payload, "model_config", ModelConfig, ModelConfig())


def _validated_config_from_payload(payload: dict[str, Any], key: str, config_type: type[Any], fallback: Any) -> Any:
    values = payload.get(key)
    if not isinstance(values, dict):
        return fallback
    allowed = {field.name for field in fields(config_type)}
    return config_type(**{name: value for name, value in values.items() if name in allowed}).validated()


def _sum_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits and labels must share batch/sequence shape, got {logits.shape[:2]} and {labels.shape}")
    valid_tokens = int(labels.ne(IGNORE_INDEX).sum().item())
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return nll, valid_tokens


def _limited_batches(loader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(item, device) for item in batch]
    return batch


def _safe_exp(value: float) -> float:
    if value > 100:
        return float("inf")
    return math.exp(value)


def _validate_max_batches(max_batches: int | None) -> None:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
