from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

SUPPORTED_OBJECTIVES = {"pretrain", "sft", "distill", "dpo", "grpo", "ppo", "reward_model"}
SUPPORTED_DATA_OBJECTIVES = {"pretrain", "sft"}
SUPPORTED_PRETRAIN_DATA_MODES = {"sliding_window", "packed", "streaming"}


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
        n_kv_head = self.n_head if self.n_kv_head is None else self.n_kv_head
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
    dataset_path: str | list[str]
    tokenizer_path: str
    out_dir: str = "runs/default"
    val_dataset_path: str | list[str] | None = None
    objective: str = "pretrain"
    init_from: str | None = None
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
    allow_tf32: bool = True
    gradient_checkpointing: bool = False
    fused_adamw: bool = False
    keep_last_checkpoints: int | None = None
    resume: str | None = None

    def validated(self) -> TrainConfig:
        _validate_path_input(self.dataset_path, "train.dataset_path")
        if self.val_dataset_path is not None:
            _validate_path_input(self.val_dataset_path, "train.val_dataset_path")
        if not isinstance(self.tokenizer_path, str) or not self.tokenizer_path:
            raise ValueError("train.tokenizer_path must be a non-empty string")
        if self.objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"train.objective must be one of: {', '.join(sorted(SUPPORTED_OBJECTIVES))}")
        if self.resume and self.init_from:
            raise ValueError("train.resume and train.init_from cannot both be set")
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
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("train.beta1 must be in [0, 1)")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("train.beta2 must be in [0, 1)")
        if self.grad_clip < 0:
            raise ValueError("train.grad_clip cannot be negative")
        if self.num_workers < 0:
            raise ValueError("train.num_workers cannot be negative")
        if not isinstance(self.out_dir, str) or not self.out_dir:
            raise ValueError("train.out_dir must be a non-empty string")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("train.device must be a non-empty string")
        if self.eval_interval <= 0 or self.save_interval <= 0 or self.log_interval <= 0:
            raise ValueError("train intervals must be positive")
        if self.eval_batches <= 0:
            raise ValueError("train.eval_batches must be positive")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("train.dtype must be auto, float32, float16, or bfloat16")
        for name in ("compile", "allow_tf32", "gradient_checkpointing", "fused_adamw"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"train.{name} must be a boolean")
        if self.keep_last_checkpoints is not None:
            if isinstance(self.keep_last_checkpoints, bool) or not isinstance(self.keep_last_checkpoints, int):
                raise ValueError("train.keep_last_checkpoints must be a positive integer when provided")
            if self.keep_last_checkpoints <= 0:
                raise ValueError("train.keep_last_checkpoints must be positive when provided")
        return self


@dataclass(frozen=True)
class DataConfig:
    pretrain_mode: str = "sliding_window"
    sequence_stride: int | None = None

    def validated(self) -> DataConfig:
        if self.pretrain_mode not in SUPPORTED_PRETRAIN_DATA_MODES:
            raise ValueError(
                f"data.pretrain_mode must be one of: {', '.join(sorted(SUPPORTED_PRETRAIN_DATA_MODES))}"
            )
        if self.sequence_stride is not None:
            if isinstance(self.sequence_stride, bool) or not isinstance(self.sequence_stride, int):
                raise ValueError("data.sequence_stride must be a positive integer when provided")
            if self.sequence_stride <= 0:
                raise ValueError("data.sequence_stride must be positive when provided")
            if self.pretrain_mode != "sliding_window":
                raise ValueError("data.sequence_stride is only supported when data.pretrain_mode is sliding_window")
        return self


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    train_base: bool = False
    train_bias: bool = False
    save_adapter: bool = True

    def validated(self) -> LoRAConfig:
        if self.rank <= 0:
            raise ValueError("lora.rank must be positive")
        if self.alpha <= 0:
            raise ValueError("lora.alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("lora.dropout must be in [0, 1)")
        if not isinstance(self.target_modules, list) or not self.target_modules:
            raise ValueError("lora.target_modules must be a non-empty list of strings")
        if any(not isinstance(name, str) or not name for name in self.target_modules):
            raise ValueError("lora.target_modules entries must be non-empty strings")
        return self


@dataclass(frozen=True)
class DistillConfig:
    mode: str = "hard"
    data_objective: str = "sft"
    teacher_checkpoint: str | None = None
    temperature: float = 2.0
    kl_weight: float = 0.0
    ce_weight: float = 1.0

    def validated(self) -> DistillConfig:
        if self.mode not in {"hard", "soft"}:
            raise ValueError("distill.mode must be hard or soft")
        if self.data_objective not in SUPPORTED_DATA_OBJECTIVES:
            raise ValueError(
                f"distill.data_objective must be one of: {', '.join(sorted(SUPPORTED_DATA_OBJECTIVES))}"
            )
        if self.mode == "soft" and not self.teacher_checkpoint:
            raise ValueError("distill.teacher_checkpoint is required when distill.mode is soft")
        if self.teacher_checkpoint is not None and (
            not isinstance(self.teacher_checkpoint, str) or not self.teacher_checkpoint
        ):
            raise ValueError("distill.teacher_checkpoint must be a non-empty string when provided")
        if self.temperature <= 0:
            raise ValueError("distill.temperature must be positive")
        if self.kl_weight < 0:
            raise ValueError("distill.kl_weight cannot be negative")
        if self.ce_weight < 0:
            raise ValueError("distill.ce_weight cannot be negative")
        if self.kl_weight == 0 and self.ce_weight == 0:
            raise ValueError("at least one of distill.kl_weight or distill.ce_weight must be positive")
        return self


@dataclass(frozen=True)
class DPOConfig:
    beta: float = 0.1
    reference_checkpoint: str | None = None
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    system_key: str = "system"

    def validated(self) -> DPOConfig:
        if self.beta <= 0:
            raise ValueError("dpo.beta must be positive")
        if self.reference_checkpoint is not None and (
            not isinstance(self.reference_checkpoint, str) or not self.reference_checkpoint
        ):
            raise ValueError("dpo.reference_checkpoint must be a non-empty string when provided")
        for config_field in fields(self):
            if config_field.name in {"beta", "reference_checkpoint"}:
                continue
            value = getattr(self, config_field.name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"dpo.{config_field.name} must be a non-empty string")
        return self


@dataclass(frozen=True)
class GRPOConfig:
    beta: float = 0.02
    reference_checkpoint: str | None = None
    prompt_key: str = "prompt"
    expected_key: str = "expected"
    system_key: str = "system"
    reward_type: str = "contains"
    num_generations: int = 4
    max_new_tokens: int = 32
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float = 1.0
    clip_range: float = 0.2
    normalize_advantages: bool = True

    def validated(self) -> GRPOConfig:
        if self.beta < 0:
            raise ValueError("grpo.beta cannot be negative")
        if self.reference_checkpoint is not None and (
            not isinstance(self.reference_checkpoint, str) or not self.reference_checkpoint
        ):
            raise ValueError("grpo.reference_checkpoint must be a non-empty string when provided")
        for name in ("prompt_key", "expected_key", "system_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"grpo.{name} must be a non-empty string")
        if self.reward_type not in {"contains", "exact_match"}:
            raise ValueError("grpo.reward_type must be contains or exact_match")
        if self.num_generations < 2:
            raise ValueError("grpo.num_generations must be at least 2")
        if self.max_new_tokens <= 0:
            raise ValueError("grpo.max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("grpo.temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("grpo.top_k must be positive when provided")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("grpo.top_p must be in (0, 1]")
        if self.clip_range < 0:
            raise ValueError("grpo.clip_range cannot be negative")
        return self


@dataclass(frozen=True)
class PPOConfig:
    beta: float = 0.02
    reference_checkpoint: str | None = None
    prompt_key: str = "prompt"
    expected_key: str = "expected"
    system_key: str = "system"
    reward_type: str = "contains"
    num_rollouts: int = 1
    max_new_tokens: int = 32
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float = 1.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    normalize_advantages: bool = True

    def validated(self) -> PPOConfig:
        if self.beta < 0:
            raise ValueError("ppo.beta cannot be negative")
        if self.reference_checkpoint is not None and (
            not isinstance(self.reference_checkpoint, str) or not self.reference_checkpoint
        ):
            raise ValueError("ppo.reference_checkpoint must be a non-empty string when provided")
        for name in ("prompt_key", "expected_key", "system_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ppo.{name} must be a non-empty string")
        if self.reward_type not in {"contains", "exact_match"}:
            raise ValueError("ppo.reward_type must be contains or exact_match")
        if self.num_rollouts <= 0:
            raise ValueError("ppo.num_rollouts must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("ppo.max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("ppo.temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("ppo.top_k must be positive when provided")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("ppo.top_p must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("ppo.gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("ppo.gae_lambda must be in [0, 1]")
        if self.clip_range < 0:
            raise ValueError("ppo.clip_range cannot be negative")
        if self.value_clip_range < 0:
            raise ValueError("ppo.value_clip_range cannot be negative")
        if self.value_loss_coef < 0:
            raise ValueError("ppo.value_loss_coef cannot be negative")
        if self.entropy_coef < 0:
            raise ValueError("ppo.entropy_coef cannot be negative")
        return self


@dataclass(frozen=True)
class RewardConfig:
    scorer: str = "rule"
    checkpoint: str | None = None
    scale: float = 1.0
    bias: float = 0.0
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    system_key: str = "system"

    def validated(self) -> RewardConfig:
        if not isinstance(self.scorer, str) or self.scorer not in {"rule", "model"}:
            raise ValueError("reward.scorer must be rule or model")
        if self.scorer == "model" and not self.checkpoint:
            raise ValueError("reward.checkpoint is required when reward.scorer is model")
        if self.checkpoint is not None and (not isinstance(self.checkpoint, str) or not self.checkpoint):
            raise ValueError("reward.checkpoint must be a non-empty string when provided")
        for name in ("scale", "bias"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"reward.{name} must be a number")
        if self.scale == 0:
            raise ValueError("reward.scale cannot be zero")
        for name in ("prompt_key", "chosen_key", "rejected_key", "system_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"reward.{name} must be a non-empty string")
        return self


@dataclass(frozen=True)
class SFTConfig:
    format: str = "auto"
    prompt_key: str = "prompt"
    response_key: str = "response"
    system_key: str = "system"
    messages_key: str = "messages"
    role_key: str = "role"
    content_key: str = "content"
    system_role: str = "system"
    user_role: str = "user"
    assistant_role: str = "assistant"
    system_prefix: str = "System:"
    user_prefix: str = "User:"
    assistant_prefix: str = "Assistant:"

    def validated(self) -> SFTConfig:
        if self.format not in {"auto", "prompt_response", "messages"}:
            raise ValueError("sft.format must be auto, prompt_response, or messages")
        for config_field in fields(self):
            value = getattr(self, config_field.name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"sft.{config_field.name} must be a non-empty string")
        return self


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_path_input(value: str | list[str], name: str) -> None:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} must be a non-empty string or non-empty list of strings")
        return
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty string or non-empty list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} entries must be non-empty strings")


def _dataclass_from_mapping(cls: type[T], values: dict[str, Any], *, section: str) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"Config section {section} must be an object")
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
    unknown = sorted(set(data) - {"model", "train", "data", "lora", "distill", "dpo", "grpo", "ppo", "reward", "sft"})
    if unknown:
        raise ValueError(f"Unknown top-level config section(s): {', '.join(unknown)}")
    if "train" not in data:
        raise ValueError("Config requires a train section")

    model = _dataclass_from_mapping(ModelConfig, data.get("model", {}), section="model").validated()
    train = _dataclass_from_mapping(TrainConfig, data["train"], section="train").validated()
    data_config = _dataclass_from_mapping(DataConfig, data.get("data", {}), section="data").validated()
    lora = _dataclass_from_mapping(LoRAConfig, data.get("lora", {}), section="lora").validated()
    distill = _dataclass_from_mapping(DistillConfig, data.get("distill", {}), section="distill").validated()
    dpo = _dataclass_from_mapping(DPOConfig, data.get("dpo", {}), section="dpo").validated()
    grpo = _dataclass_from_mapping(GRPOConfig, data.get("grpo", {}), section="grpo").validated()
    ppo = _dataclass_from_mapping(PPOConfig, data.get("ppo", {}), section="ppo").validated()
    reward = _dataclass_from_mapping(RewardConfig, data.get("reward", {}), section="reward").validated()
    sft = _dataclass_from_mapping(SFTConfig, data.get("sft", {}), section="sft").validated()
    return RunConfig(
        model=model,
        train=train,
        data=data_config,
        lora=lora,
        distill=distill,
        dpo=dpo,
        grpo=grpo,
        ppo=ppo,
        reward=reward,
        sft=sft,
    )
