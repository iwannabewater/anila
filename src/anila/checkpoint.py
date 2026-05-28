from __future__ import annotations

import json
import pickle
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import torch

from anila.config import LoRAConfig, ModelConfig
from anila.model import AnilaLM
from anila.peft import apply_lora, merge_lora


def load_checkpoint_payload(
    checkpoint: str | Path,
    *,
    required_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(f"Checkpoint cannot be loaded safely: {checkpoint}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary payload: {checkpoint}")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys {missing}: {checkpoint}")
    return payload


def inspect_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    payload = load_checkpoint_payload(checkpoint)

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
        "has_ema": payload.get("ema_model") is not None,
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
    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config"))

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
    if payload.get("ema_model") is not None:
        ema_model = AnilaLM(_model_config_from_payload(payload))
        apply_lora(ema_model, cfg)
        ema_model.load_state_dict(checkpoint_model_state(payload, use_ema=True))
        merge_lora(ema_model)
        merged_payload["ema_model"] = ema_model.state_dict()
    merged_payload["lora_config"] = asdict(replace(cfg, enabled=False))
    merged_payload["lora_targets"] = []
    merged_payload["adapter_checkpoint"] = None
    merged_payload["merged_lora_targets"] = merged_targets
    merged_payload["merged_from_checkpoint"] = str(checkpoint)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_payload, out_path)
    return out_path


def checkpoint_model_state(payload: dict[str, Any], *, use_ema: bool = False) -> dict[str, torch.Tensor]:
    key = "ema_model" if use_ema else "model"
    state = payload.get(key)
    if state is None and use_ema:
        raise ValueError("Checkpoint does not contain EMA model weights")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {key} state must be a dictionary")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint {key} state must contain only string tensor entries")
        if use_ema and not torch.is_floating_point(value):
            raise ValueError(f"Checkpoint {key} state must contain only floating point tensors")
    return state


def export_safetensors_checkpoint(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    weights_name: str = "model.safetensors",
) -> dict[str, Any]:
    save_file = _safetensors_save_file()
    weights_path_name = Path(weights_name)
    if weights_path_name.name != weights_name or not weights_name.endswith(".safetensors"):
        raise ValueError("weights_name must be a filename ending in .safetensors")

    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config"))
    tensors: dict[str, torch.Tensor] = {}
    tensor_groups: dict[str, int] = {}
    for group in ("model", "value_head", "reward_head", "ema_model", "ema_value_head", "ema_reward_head"):
        values = payload.get(group)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Checkpoint {group} state must be a dictionary to export safetensors")
        before = len(tensors)
        tensors.update(_prefixed_tensors(group, values))
        tensor_groups[group] = len(tensors) - before
    if not tensors:
        raise ValueError(f"Checkpoint does not contain exportable tensor state: {checkpoint}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / weights_name
    save_file(tensors, str(weights_path), metadata={"format": "anila", "source": str(checkpoint)})

    manifest = {
        "artifact": "anila_safetensors",
        "format": "safetensors",
        "weights": weights_path.name,
        "source_checkpoint": str(checkpoint),
        "schema_version": payload.get("schema_version"),
        "objective": payload.get("objective"),
        "step": payload.get("step"),
        "tokenizer_path": payload.get("tokenizer_path"),
        "model_config": payload.get("model_config"),
        "train_config": payload.get("train_config"),
        "data_config": payload.get("data_config"),
        "lora_config": payload.get("lora_config"),
        "tensor_groups": tensor_groups,
        "num_tensors": len(tensors),
        "num_parameters": sum(int(tensor.numel()) for tensor in tensors.values()),
    }
    manifest_path = output_dir / "anila_safetensors.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest_path": str(manifest_path), "weights_path": str(weights_path), **manifest}


def _model_config_from_payload(payload: dict[str, Any]) -> ModelConfig:
    values = payload["model_config"]
    if not isinstance(values, dict):
        raise ValueError("checkpoint model_config must be an object")
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed}).validated()


def _prefixed_tensors(prefix: str, state: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{prefix}.{name} is not a tensor and cannot be exported as safetensors")
        tensors[f"{prefix}.{name}"] = value.detach().cpu().contiguous().clone()
    return tensors


def _safetensors_save_file():
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise RuntimeError("safetensors export requires the optional dependency: uv sync --extra artifacts") from exc
    return save_file
