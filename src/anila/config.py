from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

SUPPORTED_OBJECTIVES = {"pretrain", "sft", "distill", "dpo", "opd", "grpo", "ppo", "reward_model"}
SUPPORTED_DATA_OBJECTIVES = {"pretrain", "sft"}
SUPPORTED_PRETRAIN_DATA_MODES = {"sliding_window", "packed", "streaming"}


def _validate_positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive number")
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive")


def _validate_non_negative_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a non-negative number")
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def _validate_probability(value: object, name: str, *, include_zero: bool = True, include_one: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        interval = _probability_interval(include_zero=include_zero, include_one=include_one)
        raise ValueError(f"{name} must be finite and in {interval}")
    lower_ok = value >= 0.0 if include_zero else value > 0.0
    upper_ok = value <= 1.0 if include_one else value < 1.0
    if not (lower_ok and upper_ok):
        interval = _probability_interval(include_zero=include_zero, include_one=include_one)
        raise ValueError(f"{name} must be in {interval}")


def _probability_interval(*, include_zero: bool, include_one: bool) -> str:
    left = "[" if include_zero else "("
    right = "]" if include_one else ")"
    return f"{left}0, 1{right}"


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
    rope_scaling: str | None = None
    rope_scaling_factor: float = 1.0
    rope_original_context_length: int | None = None
    rope_yarn_beta_fast: float = 32.0
    rope_yarn_beta_slow: float = 1.0
    rope_yarn_attention_factor: float = 1.0
    bias: bool = False
    tie_embeddings: bool = True
    moe_num_experts: int = 0
    moe_top_k: int = 1
    moe_intermediate_size: int | None = None
    moe_normalize_top_k: bool = True
    moe_aux_loss_coef: float = 0.0

    def validated(self) -> ModelConfig:
        n_kv_head = self.n_head if self.n_kv_head is None else self.n_kv_head
        cfg = replace(self, n_kv_head=n_kv_head)
        _validate_positive_int(cfg.vocab_size, "model.vocab_size")
        _validate_positive_int(cfg.context_length, "model.context_length")
        _validate_positive_int(cfg.n_layer, "model.n_layer")
        _validate_positive_int(cfg.n_head, "model.n_head")
        _validate_positive_int(cfg.n_kv_head, "model.n_kv_head")
        if cfg.n_head % cfg.n_kv_head != 0:
            raise ValueError("model.n_head must be divisible by model.n_kv_head")
        _validate_positive_int(cfg.n_embd, "model.n_embd")
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if (cfg.n_embd // cfg.n_head) % 2 != 0:
            raise ValueError("model.n_embd divided by model.n_head must be even for RoPE")
        _validate_probability(cfg.dropout, "model.dropout", include_zero=True, include_one=False)
        if isinstance(cfg.rope_base, bool) or not isinstance(cfg.rope_base, int | float):
            raise ValueError("model.rope_base must be a positive number")
        if cfg.rope_base <= 0 or not math.isfinite(cfg.rope_base):
            raise ValueError("model.rope_base must be positive")
        for name in ("bias", "tie_embeddings"):
            if not isinstance(getattr(cfg, name), bool):
                raise ValueError(f"model.{name} must be a boolean")
        cfg = cfg._validated_rope_scaling()
        if isinstance(cfg.moe_num_experts, bool) or not isinstance(cfg.moe_num_experts, int):
            raise ValueError("model.moe_num_experts must be an integer")
        if cfg.moe_num_experts < 0:
            raise ValueError("model.moe_num_experts cannot be negative")
        if isinstance(cfg.moe_top_k, bool) or not isinstance(cfg.moe_top_k, int):
            raise ValueError("model.moe_top_k must be an integer")
        if cfg.moe_top_k <= 0:
            raise ValueError("model.moe_top_k must be positive")
        if cfg.moe_num_experts == 1:
            raise ValueError("model.moe_num_experts must be 0 for dense models or at least 2 for MoE")
        if cfg.moe_num_experts == 0 and cfg.moe_top_k != 1:
            raise ValueError("model.moe_top_k can only differ from 1 when MoE is enabled")
        if cfg.moe_num_experts > 0 and cfg.moe_top_k > cfg.moe_num_experts:
            raise ValueError("model.moe_top_k cannot exceed model.moe_num_experts")
        if cfg.moe_intermediate_size is not None:
            if isinstance(cfg.moe_intermediate_size, bool) or not isinstance(cfg.moe_intermediate_size, int):
                raise ValueError("model.moe_intermediate_size must be a positive integer when provided")
            if cfg.moe_intermediate_size <= 0:
                raise ValueError("model.moe_intermediate_size must be positive when provided")
        if not isinstance(cfg.moe_normalize_top_k, bool):
            raise ValueError("model.moe_normalize_top_k must be a boolean")
        _validate_non_negative_number(cfg.moe_aux_loss_coef, "model.moe_aux_loss_coef")
        return cfg

    def _validated_rope_scaling(self) -> ModelConfig:
        if self.rope_scaling not in (None, "yarn"):
            raise ValueError("model.rope_scaling must be null or 'yarn'")
        _validate_positive_number(self.rope_scaling_factor, "model.rope_scaling_factor")
        _validate_positive_number(self.rope_yarn_beta_fast, "model.rope_yarn_beta_fast")
        _validate_positive_number(self.rope_yarn_beta_slow, "model.rope_yarn_beta_slow")
        _validate_positive_number(self.rope_yarn_attention_factor, "model.rope_yarn_attention_factor")
        if self.rope_scaling is None:
            if self.rope_scaling_factor != 1.0:
                raise ValueError("model.rope_scaling_factor requires model.rope_scaling = 'yarn'")
            if self.rope_original_context_length is not None:
                raise ValueError("model.rope_original_context_length requires model.rope_scaling = 'yarn'")
            if self.rope_yarn_beta_fast != 32.0 or self.rope_yarn_beta_slow != 1.0:
                raise ValueError("model.rope_yarn_beta_* requires model.rope_scaling = 'yarn'")
            if self.rope_yarn_attention_factor != 1.0:
                raise ValueError("model.rope_yarn_attention_factor requires model.rope_scaling = 'yarn'")
            return self
        if self.rope_scaling_factor <= 1.0:
            raise ValueError("model.rope_scaling_factor must be greater than 1 when YaRN scaling is enabled")
        if self.rope_original_context_length is None:
            raise ValueError("model.rope_original_context_length is required when YaRN scaling is enabled")
        if isinstance(self.rope_original_context_length, bool) or not isinstance(self.rope_original_context_length, int):
            raise ValueError("model.rope_original_context_length must be a positive integer")
        if self.rope_original_context_length <= 0:
            raise ValueError("model.rope_original_context_length must be positive")
        if self.rope_original_context_length >= self.context_length:
            raise ValueError("model.rope_original_context_length must be less than model.context_length")
        if self.rope_yarn_beta_fast < self.rope_yarn_beta_slow:
            raise ValueError("model.rope_yarn_beta_fast must be greater than or equal to model.rope_yarn_beta_slow")
        return self


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
    ema_decay: float | None = None
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
        _validate_non_negative_int(self.seed, "train.seed")
        _validate_positive_int(self.batch_size, "train.batch_size")
        _validate_positive_int(self.max_steps, "train.max_steps")
        _validate_positive_int(self.grad_accum_steps, "train.grad_accum_steps")
        _validate_positive_number(self.learning_rate, "train.learning_rate")
        _validate_probability(self.min_lr_ratio, "train.min_lr_ratio")
        _validate_non_negative_int(self.warmup_steps, "train.warmup_steps")
        _validate_non_negative_number(self.weight_decay, "train.weight_decay")
        _validate_probability(self.beta1, "train.beta1", include_zero=True, include_one=False)
        _validate_probability(self.beta2, "train.beta2", include_zero=True, include_one=False)
        _validate_non_negative_number(self.grad_clip, "train.grad_clip")
        _validate_non_negative_int(self.num_workers, "train.num_workers")
        if not isinstance(self.out_dir, str) or not self.out_dir:
            raise ValueError("train.out_dir must be a non-empty string")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("train.device must be a non-empty string")
        _validate_positive_int(self.eval_interval, "train.eval_interval")
        _validate_positive_int(self.save_interval, "train.save_interval")
        _validate_positive_int(self.log_interval, "train.log_interval")
        _validate_positive_int(self.eval_batches, "train.eval_batches")
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
        if self.ema_decay is not None:
            _validate_probability(self.ema_decay, "train.ema_decay", include_zero=False, include_one=False)
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
        _validate_positive_int(self.rank, "lora.rank")
        _validate_positive_number(self.alpha, "lora.alpha")
        _validate_probability(self.dropout, "lora.dropout", include_zero=True, include_one=False)
        if not isinstance(self.target_modules, list) or not self.target_modules:
            raise ValueError("lora.target_modules must be a non-empty list of strings")
        if any(not isinstance(name, str) or not name for name in self.target_modules):
            raise ValueError("lora.target_modules entries must be non-empty strings")
        for name in ("enabled", "train_base", "train_bias", "save_adapter"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"lora.{name} must be a boolean")
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
        _validate_positive_number(self.temperature, "distill.temperature")
        _validate_non_negative_number(self.kl_weight, "distill.kl_weight")
        _validate_non_negative_number(self.ce_weight, "distill.ce_weight")
        if self.kl_weight == 0 and self.ce_weight == 0:
            raise ValueError("at least one of distill.kl_weight or distill.ce_weight must be positive")
        return self


@dataclass(frozen=True)
class OPDConfig:
    teacher_checkpoint: str | None = None
    prompt_key: str = "prompt"
    expected_key: str = "expected"
    system_key: str = "system"
    num_rollouts: int = 1
    max_new_tokens: int = 32
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float = 1.0
    distill_temperature: float = 2.0
    kl_weight: float = 1.0
    ce_weight: float = 0.0

    def validated(self) -> OPDConfig:
        if self.teacher_checkpoint is not None and (
            not isinstance(self.teacher_checkpoint, str) or not self.teacher_checkpoint
        ):
            raise ValueError("opd.teacher_checkpoint must be a non-empty string")
        for name in ("prompt_key", "expected_key", "system_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"opd.{name} must be a non-empty string")
        _validate_positive_int(self.num_rollouts, "opd.num_rollouts")
        _validate_positive_int(self.max_new_tokens, "opd.max_new_tokens")
        _validate_positive_number(self.temperature, "opd.temperature")
        if self.top_k is not None:
            _validate_positive_int(self.top_k, "opd.top_k")
        _validate_probability(self.top_p, "opd.top_p", include_zero=False)
        _validate_positive_number(self.distill_temperature, "opd.distill_temperature")
        _validate_non_negative_number(self.kl_weight, "opd.kl_weight")
        _validate_non_negative_number(self.ce_weight, "opd.ce_weight")
        if self.kl_weight == 0 and self.ce_weight == 0:
            raise ValueError("at least one of opd.kl_weight or opd.ce_weight must be positive")
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
        _validate_positive_number(self.beta, "dpo.beta")
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
    loss_type: str = "grpo"
    cispo_ratio_cap: float = 5.0

    def validated(self) -> GRPOConfig:
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, int | float)
            or self.beta < 0
            or not math.isfinite(self.beta)
        ):
            raise ValueError("grpo.beta must be finite and non-negative")
        if self.reference_checkpoint is not None and (
            not isinstance(self.reference_checkpoint, str) or not self.reference_checkpoint
        ):
            raise ValueError("grpo.reference_checkpoint must be a non-empty string when provided")
        for name in ("prompt_key", "expected_key", "system_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"grpo.{name} must be a non-empty string")
        if self.reward_type not in {"contains", "exact_match", "tool_call"}:
            raise ValueError("grpo.reward_type must be contains, exact_match, or tool_call")
        if self.loss_type not in {"grpo", "cispo"}:
            raise ValueError("grpo.loss_type must be grpo or cispo")
        _validate_positive_int(self.num_generations, "grpo.num_generations")
        if self.num_generations < 2:
            raise ValueError("grpo.num_generations must be at least 2")
        _validate_positive_int(self.max_new_tokens, "grpo.max_new_tokens")
        _validate_positive_number(self.temperature, "grpo.temperature")
        if self.top_k is not None:
            _validate_positive_int(self.top_k, "grpo.top_k")
        _validate_probability(self.top_p, "grpo.top_p", include_zero=False)
        _validate_non_negative_number(self.clip_range, "grpo.clip_range")
        _validate_positive_number(self.cispo_ratio_cap, "grpo.cispo_ratio_cap")
        if not isinstance(self.normalize_advantages, bool):
            raise ValueError("grpo.normalize_advantages must be a boolean")
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
        _validate_non_negative_number(self.beta, "ppo.beta")
        if self.reference_checkpoint is not None and (
            not isinstance(self.reference_checkpoint, str) or not self.reference_checkpoint
        ):
            raise ValueError("ppo.reference_checkpoint must be a non-empty string when provided")
        for name in ("prompt_key", "expected_key", "system_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ppo.{name} must be a non-empty string")
        if self.reward_type not in {"contains", "exact_match", "tool_call"}:
            raise ValueError("ppo.reward_type must be contains, exact_match, or tool_call")
        _validate_positive_int(self.num_rollouts, "ppo.num_rollouts")
        _validate_positive_int(self.max_new_tokens, "ppo.max_new_tokens")
        _validate_positive_number(self.temperature, "ppo.temperature")
        if self.top_k is not None:
            _validate_positive_int(self.top_k, "ppo.top_k")
        _validate_probability(self.top_p, "ppo.top_p", include_zero=False)
        _validate_probability(self.gamma, "ppo.gamma")
        _validate_probability(self.gae_lambda, "ppo.gae_lambda")
        _validate_non_negative_number(self.clip_range, "ppo.clip_range")
        _validate_non_negative_number(self.value_clip_range, "ppo.value_clip_range")
        _validate_non_negative_number(self.value_loss_coef, "ppo.value_loss_coef")
        _validate_non_negative_number(self.entropy_coef, "ppo.entropy_coef")
        if not isinstance(self.normalize_advantages, bool):
            raise ValueError("ppo.normalize_advantages must be a boolean")
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
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"reward.{name} must be a finite number")
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
    reasoning_key: str = "reasoning_content"
    tool_calls_key: str = "tool_calls"
    system_role: str = "system"
    user_role: str = "user"
    assistant_role: str = "assistant"
    tool_role: str = "tool"
    system_prefix: str = "System:"
    user_prefix: str = "User:"
    assistant_prefix: str = "Assistant:"
    tool_prefix: str = "Tool:"
    thinking_start: str = "<think>"
    thinking_end: str = "</think>"
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"

    def validated(self) -> SFTConfig:
        if self.format not in {"auto", "prompt_response", "messages"}:
            raise ValueError("sft.format must be auto, prompt_response, or messages")
        for config_field in fields(self):
            value = getattr(self, config_field.name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"sft.{config_field.name} must be a non-empty string")
        roles = [self.system_role, self.user_role, self.assistant_role, self.tool_role]
        if len(set(roles)) != len(roles):
            raise ValueError("sft role names must be distinct")
        delimiters = [
            self.thinking_start,
            self.thinking_end,
            self.tool_call_start,
            self.tool_call_end,
            self.tool_response_start,
            self.tool_response_end,
        ]
        if len(set(delimiters)) != len(delimiters):
            raise ValueError("sft tag delimiters must be distinct")
        return self


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    opd: OPDConfig = field(default_factory=OPDConfig)
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
    unknown = sorted(
        set(data) - {"model", "train", "data", "lora", "distill", "dpo", "opd", "grpo", "ppo", "reward", "sft"}
    )
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
    opd = _dataclass_from_mapping(OPDConfig, data.get("opd", {}), section="opd").validated()
    grpo = _dataclass_from_mapping(GRPOConfig, data.get("grpo", {}), section="grpo").validated()
    ppo = _dataclass_from_mapping(PPOConfig, data.get("ppo", {}), section="ppo").validated()
    reward = _dataclass_from_mapping(RewardConfig, data.get("reward", {}), section="reward").validated()
    sft = _dataclass_from_mapping(SFTConfig, data.get("sft", {}), section="sft").validated()
    if train.objective == "opd" and not opd.teacher_checkpoint:
        raise ValueError("opd.teacher_checkpoint is required when train.objective is opd")
    return RunConfig(
        model=model,
        train=train,
        data=data_config,
        lora=lora,
        distill=distill,
        dpo=dpo,
        opd=opd,
        grpo=grpo,
        ppo=ppo,
        reward=reward,
        sft=sft,
    )
