from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    context_length: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None
    n_embd: int = 256
    dropout: float = 0.0
    rope_base: float = 10000.0
    bias: bool = False
    tie_embeddings: bool = True

    def validated(self) -> ModelConfig:
        n_kv_head = self.n_kv_head or self.n_head
        cfg = replace(self, n_kv_head=n_kv_head)
        if cfg.vocab_size <= 0:
            raise ValueError("model.vocab_size must be positive")
        if cfg.context_length <= 0:
            raise ValueError("model.context_length must be positive")
        if cfg.n_layer <= 0:
            raise ValueError("model.n_layer must be positive")
        if cfg.n_head <= 0:
            raise ValueError("model.n_head must be positive")
        if cfg.n_kv_head is None or cfg.n_kv_head <= 0:
            raise ValueError("model.n_kv_head must be positive")
        if cfg.n_head % cfg.n_kv_head != 0:
            raise ValueError("model.n_head must be divisible by model.n_kv_head")
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if not 0.0 <= cfg.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if cfg.rope_base <= 0:
            raise ValueError("model.rope_base must be positive")
        return cfg


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: str
    tokenizer_path: str
    out_dir: str = "runs/default"
    val_dataset_path: str | None = None
    seed: int = 42
    batch_size: int = 16
    max_steps: int = 1000
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 10
    save_interval: int = 500
    log_interval: int = 10
    num_workers: int = 0
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    resume: str | None = None

    def validated(self) -> TrainConfig:
        if self.batch_size <= 0:
            raise ValueError("train.batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("train.max_steps must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("train.grad_accum_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("train.learning_rate must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("train.min_lr_ratio must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("train.warmup_steps cannot be negative")
        if self.weight_decay < 0:
            raise ValueError("train.weight_decay cannot be negative")
        if self.grad_clip < 0:
            raise ValueError("train.grad_clip cannot be negative")
        if self.eval_interval <= 0 or self.save_interval <= 0 or self.log_interval <= 0:
            raise ValueError("train intervals must be positive")
        if self.eval_batches <= 0:
            raise ValueError("train.eval_batches must be positive")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("train.dtype must be auto, float32, float16, or bfloat16")
        return self


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataclass_from_mapping(cls: type[T], values: dict[str, Any], *, section: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {section}: {', '.join(unknown)}")
    return cls(**values)


def load_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as f:
        if config_path.suffix == ".json":
            data = json.load(f)
        elif config_path.suffix in {".toml", ".tml"}:
            data = tomllib.load(f)
        else:
            raise ValueError("Config files must be .json or .toml")
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object")
    return data


def load_run_config(path: str | Path) -> RunConfig:
    data = load_mapping(path)
    unknown = sorted(set(data) - {"model", "train"})
    if unknown:
        raise ValueError(f"Unknown top-level config section(s): {', '.join(unknown)}")
    if "train" not in data:
        raise ValueError("Config requires a train section")

    model = _dataclass_from_mapping(ModelConfig, data.get("model", {}), section="model").validated()
    train = _dataclass_from_mapping(TrainConfig, data["train"], section="train").validated()
    return RunConfig(model=model, train=train)
