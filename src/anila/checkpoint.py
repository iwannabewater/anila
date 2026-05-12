from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import torch

from anila.config import LoRAConfig, ModelConfig
from anila.model import AnilaLM
from anila.peft import apply_lora, merge_lora


def inspect_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary payload: {checkpoint}")

    train_config = payload.get("train_config")
    data_config = payload.get("data_config")
    model_config = payload.get("model_config")
    lora_config = payload.get("lora_config")
    summary: dict[str, Any] = {
        "schema_version": payload.get("schema_version"),
        "objective": payload.get("objective"),
        "step": payload.get("step"),
        "tokenizer_path": payload.get("tokenizer_path"),
        "has_model": "model" in payload,
        "has_optimizer": "optimizer" in payload,
        "has_lora": isinstance(lora_config, dict) and bool(lora_config.get("enabled", False)),
        "has_value_head": payload.get("value_head") is not None,
        "has_reward_head": payload.get("reward_head") is not None,
        "adapter_checkpoint": payload.get("adapter_checkpoint"),
        "is_merged_lora": "merged_lora_targets" in payload,
        "merged_lora_targets": payload.get("merged_lora_targets"),
    }
    if isinstance(model_config, dict):
        summary["model"] = {
            "vocab_size": model_config.get("vocab_size"),
            "context_length": model_config.get("context_length"),
            "n_layer": model_config.get("n_layer"),
            "n_head": model_config.get("n_head"),
            "n_kv_head": model_config.get("n_kv_head"),
            "n_embd": model_config.get("n_embd"),
        }
    if isinstance(train_config, dict):
        summary["train"] = {
            "objective": train_config.get("objective"),
            "dataset_path": train_config.get("dataset_path"),
            "out_dir": train_config.get("out_dir"),
            "max_steps": train_config.get("max_steps"),
        }
    if isinstance(data_config, dict):
        summary["data"] = {
            "pretrain_mode": data_config.get("pretrain_mode"),
            "sequence_stride": data_config.get("sequence_stride"),
        }
    return summary


def merge_lora_checkpoint(checkpoint: str | Path, out: str | Path) -> Path:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary payload: {checkpoint}")
    if "model" not in payload or "model_config" not in payload:
        raise ValueError(f"Checkpoint is missing model payload: {checkpoint}")

    lora_config = payload.get("lora_config")
    if not isinstance(lora_config, dict) or not lora_config.get("enabled", False):
        raise ValueError(f"Checkpoint does not contain enabled LoRA adapters: {checkpoint}")

    cfg = LoRAConfig(**lora_config).validated()
    model = AnilaLM(_model_config_from_payload(payload))
    apply_lora(model, cfg)
    model.load_state_dict(payload["model"])
    merged_targets = merge_lora(model)

    merged_payload = dict(payload)
    merged_payload["model"] = model.state_dict()
    merged_payload["lora_config"] = asdict(replace(cfg, enabled=False))
    merged_payload["lora_targets"] = []
    merged_payload["adapter_checkpoint"] = None
    merged_payload["merged_lora_targets"] = merged_targets
    merged_payload["merged_from_checkpoint"] = str(checkpoint)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_payload, out_path)
    return out_path


def _model_config_from_payload(payload: dict[str, Any]) -> ModelConfig:
    values = payload["model_config"]
    if not isinstance(values, dict):
        raise ValueError("checkpoint model_config must be an object")
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed}).validated()
