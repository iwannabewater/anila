from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def inspect_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary payload: {checkpoint}")

    train_config = payload.get("train_config")
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
    return summary
