from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

from anila.checkpoint import load_checkpoint_payload
from anila.config import LoRAConfig, RunConfig, TrainConfig
from anila.data import IGNORE_INDEX, create_dataloader
from anila.distillation import load_teacher_model, on_policy_distillation_loss, soft_distillation_loss
from anila.dpo import dpo_loss, sequence_logprobs
from anila.grpo import group_advantages, grpo_loss
from anila.model import AnilaLM
from anila.peft import (
    adapt_state_dict_for_lora,
    apply_lora,
    lora_state_dict,
    mark_lora_trainable,
    trainable_parameter_count,
)
from anila.ppo import PolicyValueModel, compute_gae, normalize_masked, ppo_loss, token_entropy, token_logprobs
from anila.reward import RewardModel, RewardScorer, build_reward_scorer, reward_model_loss
from anila.tokenization import AnilaTokenizer

console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    if python_state is None or not isinstance(numpy_state, dict) or not isinstance(torch_state, torch.Tensor):
        raise ValueError("checkpoint rng_state is invalid")
    try:
        random.setstate(python_state)
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(torch_state)
        cuda_state = state.get("cuda")
        if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("checkpoint rng_state is invalid") from exc


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def configure_optimizer(model: torch.nn.Module, cfg: TrainConfig, device: torch.device | None = None) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    if not decay and not no_decay:
        raise ValueError("No trainable parameters found for optimizer")
    optimizer_kwargs: dict[str, Any] = {
        "lr": cfg.learning_rate,
        "betas": (cfg.beta1, cfg.beta2),
    }
    if cfg.fused_adamw and device is not None and device.type == "cuda" and "fused" in signature(torch.optim.AdamW).parameters:
        optimizer_kwargs["fused"] = True
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        **optimizer_kwargs,
    )


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def configure_torch_runtime(cfg: TrainConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32


class ResumableDataIterator(Iterator[tuple[Any, ...]]):
    def __init__(
        self,
        loader,
        generator: torch.Generator,
        state: dict[str, Any] | None = None,
    ):
        self.loader = loader
        self.generator = generator
        self.epoch = 0
        self.batches_in_epoch = 0
        self.total_batches = 0
        self.epoch_start_generator_state = generator.get_state().clone()
        self._iterator: Iterator[tuple[Any, ...]] | None = None
        if state is not None:
            self.load_state_dict(state)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "epoch": self.epoch,
            "batches_in_epoch": self.batches_in_epoch,
            "total_batches": self.total_batches,
            "epoch_start_generator_state": self.epoch_start_generator_state.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("checkpoint data_state schema_version is invalid")
        fields = (state.get("epoch"), state.get("batches_in_epoch"), state.get("total_batches"))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in fields):
            raise ValueError("checkpoint data_state counters are invalid")
        generator_state = state.get("epoch_start_generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("checkpoint data_state generator state is invalid")
        try:
            self.generator.set_state(generator_state)
        except RuntimeError as exc:
            raise ValueError("checkpoint data_state generator state is invalid") from exc
        self.epoch = fields[0]
        self.batches_in_epoch = fields[1]
        self.total_batches = fields[2]
        self.epoch_start_generator_state = generator_state.clone()
        self._iterator = None

    def __next__(self) -> tuple[Any, ...]:
        if self._iterator is None:
            self._open_current_epoch()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batches_in_epoch = 0
            self.epoch_start_generator_state = self.generator.get_state().clone()
            self._iterator = iter(self.loader)
            try:
                batch = next(self._iterator)
            except StopIteration as exc:
                raise ValueError("Dataloader produced no batches; reduce batch_size or add data") from exc
        self.batches_in_epoch += 1
        self.total_batches += 1
        return batch

    def _open_current_epoch(self) -> None:
        self.generator.set_state(self.epoch_start_generator_state)
        self._iterator = iter(self.loader)
        for _ in range(self.batches_in_epoch):
            try:
                next(self._iterator)
            except StopIteration as exc:
                raise ValueError("checkpoint data_state is not compatible with the current dataloader") from exc


def _dataloader_is_empty(loader) -> bool:
    try:
        return len(loader) == 0
    except TypeError:
        return False


class CheckpointManager:
    def __init__(self, out_dir: str | Path, *, keep_last: int | None = None):
        if keep_last is not None:
            if isinstance(keep_last, bool) or not isinstance(keep_last, int):
                raise ValueError("keep_last must be a positive integer when provided")
            if keep_last <= 0:
                raise ValueError("keep_last must be positive when provided")
        self.root = Path(out_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.pt"

    @property
    def latest_adapter_path(self) -> Path:
        return self.root / "adapters" / "latest.pt"

    def save(self, payload: dict[str, Any], *, step: int) -> Path:
        step_path = self.root / f"step_{step:08d}.pt"
        self._atomic_save(payload, step_path)
        self._atomic_save(payload, self.latest_path)
        self._prune_old_step_checkpoints(self.root)
        return step_path

    def save_adapter(self, payload: dict[str, Any], *, step: int) -> Path:
        adapter_root = self.root / "adapters"
        adapter_root.mkdir(parents=True, exist_ok=True)
        step_path = adapter_root / f"step_{step:08d}.pt"
        self._atomic_save(payload, step_path)
        self._atomic_save(payload, self.latest_adapter_path)
        self._prune_old_step_checkpoints(adapter_root)
        return step_path

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        load_path = Path(path) if path else self.latest_path
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        return load_checkpoint_payload(load_path)

    def _prune_old_step_checkpoints(self, root: Path) -> None:
        if self.keep_last is None:
            return
        step_paths = sorted(root.glob("step_*.pt"))
        for stale_path in step_paths[: max(0, len(step_paths) - self.keep_last)]:
            stale_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)


class RunRecorder:
    def __init__(self, out_dir: str | Path):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.root / "metrics.jsonl"
        self.config_path = self.root / "config.json"

    def write_config(self, config: RunConfig) -> None:
        payload = config.to_dict()
        tmp = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.config_path)

    def append(self, **record: Any) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


class Trainer:
    def __init__(self, config: RunConfig):
        self.config = config
        self.train_cfg = config.train
        self.device = resolve_device(self.train_cfg.device)
        self.dtype = resolve_dtype(self.train_cfg.dtype, self.device)
        configure_torch_runtime(self.train_cfg, self.device)
        set_seed(self.train_cfg.seed)

        self.tokenizer = AnilaTokenizer.load(self.train_cfg.tokenizer_path)
        self.model_cfg = replace(config.model, vocab_size=self.tokenizer.vocab_size).validated()
        self.model = AnilaLM(self.model_cfg)
        self.lora_targets: list[str] = []
        if self.config.lora.enabled:
            self.lora_targets = apply_lora(self.model, self.config.lora)
            mark_lora_trainable(
                self.model,
                train_base=self.config.lora.train_base,
                train_bias=self.config.lora.train_bias,
            )
        self.model.set_gradient_checkpointing(self.train_cfg.gradient_checkpointing)
        initial_payload = None
        if self.train_cfg.init_from:
            initial_payload = self._load_initial_weights(self.train_cfg.init_from)
        if self.train_cfg.objective == "ppo":
            self.model = PolicyValueModel(self.model)
            if initial_payload is not None and initial_payload.get("value_head") is not None:
                self._value_model().value_head.load_state_dict(initial_payload["value_head"])
        if self.train_cfg.objective == "reward_model":
            self.model = RewardModel(self.model)
            if initial_payload is not None and initial_payload.get("reward_head") is not None:
                self._reward_model().reward_head.load_state_dict(initial_payload["reward_head"])
        self.model = self.model.to(self.device)
        self.teacher_model: AnilaLM | None = None
        if self.train_cfg.objective == "distill" and self.config.distill.mode == "soft":
            if self.config.distill.teacher_checkpoint is None:
                raise ValueError("distill.teacher_checkpoint is required for soft distillation")
            self.teacher_model = load_teacher_model(self.config.distill.teacher_checkpoint, self.device)
        if self.train_cfg.objective == "opd":
            if self.config.opd.teacher_checkpoint is None:
                raise ValueError("opd.teacher_checkpoint is required for on-policy distillation")
            self.teacher_model = load_teacher_model(self.config.opd.teacher_checkpoint, self.device)
        self.reference_model: AnilaLM | None = None
        if self.train_cfg.objective in {"dpo", "grpo", "ppo"}:
            if self.train_cfg.init_from is None:
                raise ValueError(f"{self.train_cfg.objective.upper()} requires train.init_from to initialize the policy model")
            if self.train_cfg.objective == "dpo":
                reference_checkpoint = self.config.dpo.reference_checkpoint
            elif self.train_cfg.objective == "grpo":
                reference_checkpoint = self.config.grpo.reference_checkpoint
            else:
                reference_checkpoint = self.config.ppo.reference_checkpoint
            reference_checkpoint = reference_checkpoint or self.train_cfg.init_from
            self.reference_model = self._load_reference_model(reference_checkpoint)
        self.reward_scorer: RewardScorer | None = None
        if self.train_cfg.objective == "grpo":
            self.reward_scorer = build_reward_scorer(
                self.config.reward,
                reward_type=self.config.grpo.reward_type,
                device=self.device,
            )
        if self.train_cfg.objective == "ppo":
            self.reward_scorer = build_reward_scorer(
                self.config.reward,
                reward_type=self.config.ppo.reward_type,
                device=self.device,
            )
        if self.train_cfg.compile:
            self.model = torch.compile(self.model)

        self.ema_state: dict[str, torch.Tensor] | None = None
        if self.train_cfg.ema_decay is not None:
            self.ema_state = _clone_floating_state_dict(self._runtime_model())

        self.optimizer = configure_optimizer(self.model, self.train_cfg, self.device)
        self.recorder = RunRecorder(self.train_cfg.out_dir)
        self.recorder.write_config(self.config)
        self.checkpoints = CheckpointManager(self.train_cfg.out_dir, keep_last=self.train_cfg.keep_last_checkpoints)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and self.dtype == torch.float16)
        self.start_step = 0
        self._resume_data_state: dict[str, Any] | None = None
        if self.train_cfg.resume:
            self._restore(self.train_cfg.resume)

        data_objective = self._data_objective()
        self.train_data_generator = torch.Generator().manual_seed(self.train_cfg.seed)
        self.train_loader = create_dataloader(
            self.train_cfg.dataset_path,
            self.tokenizer,
            context_length=self.model_cfg.context_length,
            batch_size=self.train_cfg.batch_size,
            objective=data_objective,
            sft_config=self.config.sft,
            data_config=self.config.data,
            dpo_config=self.config.dpo,
            grpo_config=self.config.grpo,
            opd_config=self.config.opd,
            ppo_config=self.config.ppo,
            reward_config=self.config.reward,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            drop_last=data_objective == "pretrain",
            generator=self.train_data_generator,
        )
        eval_path = self.train_cfg.val_dataset_path or self.train_cfg.dataset_path
        self.eval_loader = create_dataloader(
            eval_path,
            self.tokenizer,
            context_length=self.model_cfg.context_length,
            batch_size=self.train_cfg.batch_size,
            objective=data_objective,
            sft_config=self.config.sft,
            data_config=self.config.data,
            dpo_config=self.config.dpo,
            grpo_config=self.config.grpo,
            opd_config=self.config.opd,
            ppo_config=self.config.ppo,
            reward_config=self.config.reward,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            drop_last=False,
        )
        if _dataloader_is_empty(self.train_loader):
            raise ValueError("Training dataloader is empty; reduce batch_size or add data")
        if _dataloader_is_empty(self.eval_loader):
            raise ValueError("Evaluation dataloader is empty; reduce batch_size or add validation data")
        if self._resume_data_state is not None and self._resume_data_state.get("contract") != self._data_resume_contract():
            raise ValueError("Resume checkpoint data contract does not match the current training data configuration")
        self.train_iterator = ResumableDataIterator(
            self.train_loader,
            self.train_data_generator,
            self._resume_data_state,
        )

    def train(self) -> None:
        runtime_model = self._runtime_model()
        console.print(
            f"[bold]Anila[/bold] training on {self.device} with dtype={self.dtype}, "
            f"params={format_parameter_count(sum(param.numel() for param in runtime_model.parameters()))}, "
            f"trainable={format_parameter_count(trainable_parameter_count(runtime_model))}, "
            f"grad_ckpt={self.train_cfg.gradient_checkpointing}, "
            f"ema={self.train_cfg.ema_decay if self.train_cfg.ema_decay is not None else 'off'}"
        )
        start_time = time.time()
        self.model.train()
        for step in range(self.start_step, self.train_cfg.max_steps):
            lr = cosine_lr(step, self.train_cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(self.train_cfg.grad_accum_steps):
                batch = self._move_batch_to_device(next(self.train_iterator))
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ):
                    loss = self._compute_batch_loss(batch)
                    loss = loss / self.train_cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                loss_value += float(loss.detach().cpu()) * self.train_cfg.grad_accum_steps

            self.scaler.unscale_(self.optimizer)
            if self.train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            scale_before_step = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self._optimizer_step_was_applied(scale_before_step):
                self._update_ema_state()

            completed_step = step + 1
            if completed_step % self.train_cfg.log_interval == 0:
                elapsed = time.time() - start_time
                self.recorder.append(
                    event="train",
                    step=completed_step,
                    loss=loss_value,
                    learning_rate=lr,
                    elapsed_seconds=elapsed,
                )
                console.print(
                    f"step={completed_step:6d}/{self.train_cfg.max_steps} "
                    f"loss={loss_value:.4f} lr={lr:.2e} elapsed={elapsed:.1f}s"
                )
            if completed_step % self.train_cfg.eval_interval == 0:
                eval_loss = self.evaluate()
                self.recorder.append(event="eval", step=completed_step, loss=eval_loss)
                console.print(f"eval step={completed_step:6d} loss={eval_loss:.4f}")
            if completed_step % self.train_cfg.save_interval == 0:
                self.save(completed_step)

        if self.train_cfg.max_steps % self.train_cfg.save_interval != 0:
            self.save(self.train_cfg.max_steps)

    @torch.no_grad()
    def evaluate(self) -> float:
        rng_state = capture_rng_state()
        was_training = self.model.training
        self.model.eval()
        losses = []
        try:
            with self._ema_weights_for_eval():
                for index, batch in enumerate(self.eval_loader):
                    if index >= self.train_cfg.eval_batches:
                        break
                    batch = self._move_batch_to_device(batch)
                    with torch.autocast(
                        device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
                    ):
                        loss = self._compute_batch_loss(batch)
                    losses.append(float(loss.detach().cpu()))
        finally:
            restore_rng_state(rng_state)
            if was_training:
                self.model.train()
        return sum(losses) / max(len(losses), 1)

    def save(self, step: int) -> Path:
        adapter_path = self._save_adapter(step)
        payload = {
            "schema_version": 1,
            "objective": self.train_cfg.objective,
            "model": self._raw_model().state_dict(),
            "model_config": asdict(self.model_cfg),
            "train_config": asdict(self.train_cfg),
            "data_config": asdict(self.config.data),
            "distill_config": asdict(self.config.distill),
            "lora_config": asdict(self.config.lora),
            "lora_targets": self.lora_targets,
            "adapter_checkpoint": str(adapter_path) if adapter_path else None,
            "dpo_config": asdict(self.config.dpo),
            "opd_config": asdict(self.config.opd),
            "grpo_config": asdict(self.config.grpo),
            "ppo_config": asdict(self.config.ppo),
            "reward_config": asdict(self.config.reward),
            "value_head": self._value_model().value_head.state_dict() if self.train_cfg.objective == "ppo" else None,
            "reward_head": self._reward_model().reward_head.state_dict()
            if self.train_cfg.objective == "reward_model"
            else None,
            "sft_config": asdict(self.config.sft),
            "tokenizer_path": self.train_cfg.tokenizer_path,
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "data_state": {
                **self.train_iterator.state_dict(),
                "contract": self._data_resume_contract(),
            },
        }
        payload.update(self._ema_payload_states())
        path = self.checkpoints.save(payload, step=step)
        self.recorder.append(event="checkpoint", step=step, path=str(path))
        console.print(f"saved checkpoint: {path}")
        return path

    def _restore(self, path: str) -> None:
        payload = self.checkpoints.load(path)
        self._raw_model().load_state_dict(payload["model"])
        if self.train_cfg.objective == "ppo" and payload.get("value_head") is not None:
            self._value_model().value_head.load_state_dict(payload["value_head"])
        if self.train_cfg.objective == "reward_model" and payload.get("reward_head") is not None:
            self._reward_model().reward_head.load_state_dict(payload["reward_head"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self._restore_ema_state(payload)
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        data_state = payload.get("data_state")
        if data_state is not None and not isinstance(data_state, dict):
            raise ValueError("checkpoint data_state must be an object")
        self._resume_data_state = data_state
        self.start_step = int(payload.get("step", 0))

    def _load_initial_weights(self, path: str) -> dict[str, Any]:
        payload = load_checkpoint_payload(path, required_keys=("model",))
        state_dict = payload["model"]
        if self.config.lora.enabled:
            state_dict = adapt_state_dict_for_lora(self.model, state_dict)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if ".lora_" not in key]
        missing = [
            key
            for key in missing
            if ".lora_" not in key
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Initial checkpoint is not compatible with the current model "
                f"(missing={missing}, unexpected={unexpected})"
            )
        return payload

    def _save_adapter(self, step: int) -> Path | None:
        if not self.config.lora.enabled or not self.config.lora.save_adapter:
            return None
        payload = {
            "schema_version": 1,
            "artifact": "lora_adapter",
            "objective": self.train_cfg.objective,
            "model_config": asdict(self.model_cfg),
            "train_config": asdict(self.train_cfg),
            "data_config": asdict(self.config.data),
            "lora_config": asdict(self.config.lora),
            "lora_targets": self.lora_targets,
            "adapter": lora_state_dict(self._raw_model()),
            "step": step,
        }
        return self.checkpoints.save_adapter(payload, step=step)

    def _optimizer_step_was_applied(self, scale_before_step: float) -> bool:
        if not self.scaler.is_enabled():
            return True
        return self.scaler.get_scale() >= scale_before_step

    def _update_ema_state(self) -> None:
        if self.ema_state is None or self.train_cfg.ema_decay is None:
            return
        decay = float(self.train_cfg.ema_decay)
        current = self._runtime_model().state_dict()
        for name, value in current.items():
            if not torch.is_floating_point(value):
                continue
            if name not in self.ema_state:
                self.ema_state[name] = value.detach().clone()
                continue
            self.ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    @contextmanager
    def _ema_weights_for_eval(self) -> Iterator[None]:
        if self.ema_state is None:
            yield
            return
        runtime_model = self._runtime_model()
        current = runtime_model.state_dict()
        backup = {name: current[name].detach().clone() for name in self.ema_state if name in current}
        runtime_model.load_state_dict(self.ema_state, strict=False)
        try:
            yield
        finally:
            runtime_model.load_state_dict(backup, strict=False)

    def _ema_payload_states(self) -> dict[str, Any]:
        if self.ema_state is None:
            return {}
        if self.train_cfg.objective == "ppo":
            payload: dict[str, Any] = {
                "ema_model": _state_to_cpu(_strip_state_prefix(self.ema_state, "policy.")),
                "ema_value_head": _state_to_cpu(_strip_state_prefix(self.ema_state, "value_head.")),
            }
        elif self.train_cfg.objective == "reward_model":
            payload = {
                "ema_model": _state_to_cpu(_strip_state_prefix(self.ema_state, "backbone.")),
                "ema_reward_head": _state_to_cpu(_strip_state_prefix(self.ema_state, "reward_head.")),
            }
        else:
            payload = {"ema_model": _state_to_cpu(self.ema_state)}
        payload["ema_decay"] = float(self.train_cfg.ema_decay) if self.train_cfg.ema_decay is not None else None
        return payload

    def _restore_ema_state(self, payload: dict[str, Any]) -> None:
        if self.train_cfg.ema_decay is None:
            self.ema_state = None
            return
        ema_model = payload.get("ema_model")
        if ema_model is None:
            self.ema_state = _clone_floating_state_dict(self._runtime_model())
            return
        if not isinstance(ema_model, dict):
            raise ValueError("checkpoint ema_model state must be a dictionary")
        if self.train_cfg.objective == "ppo":
            loaded: dict[str, Any] = _add_state_prefix(ema_model, "policy.")
            loaded.update(_add_state_prefix(_required_ema_head(payload, "ema_value_head"), "value_head."))
        elif self.train_cfg.objective == "reward_model":
            loaded = _add_state_prefix(ema_model, "backbone.")
            loaded.update(_add_state_prefix(_required_ema_head(payload, "ema_reward_head"), "reward_head."))
        else:
            loaded = dict(ema_model)
        self.ema_state = _load_ema_state_for_model(self._runtime_model(), loaded)

    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.train_cfg.objective == "distill" and self.config.distill.mode == "soft":
            if self.teacher_model is None:
                raise RuntimeError("teacher model is not loaded")
            student_logits = self.model(x).logits
            with torch.no_grad():
                teacher_logits = self.teacher_model(x).logits
            return soft_distillation_loss(student_logits, teacher_logits, y, self.config.distill).loss
        loss = self.model(x, targets=y).loss
        if loss is None:
            raise RuntimeError("model did not return a training loss")
        return loss

    def _data_objective(self) -> str:
        if self.train_cfg.objective == "distill":
            return self.config.distill.data_objective
        if self.train_cfg.objective in {"dpo", "opd", "grpo", "ppo", "reward_model"}:
            return self.train_cfg.objective
        return self.train_cfg.objective

    def _data_resume_contract(self) -> dict[str, Any]:
        contract = {
            "objective": self.train_cfg.objective,
            "dataset_path": self.train_cfg.dataset_path,
            "batch_size": self.train_cfg.batch_size,
            "grad_accum_steps": self.train_cfg.grad_accum_steps,
            "num_workers": self.train_cfg.num_workers,
            "context_length": self.model_cfg.context_length,
            "data_config": asdict(self.config.data),
            "sft_config": asdict(self.config.sft),
            "dpo_config": asdict(self.config.dpo),
            "grpo_config": asdict(self.config.grpo),
            "ppo_config": asdict(self.config.ppo),
            "reward_config": asdict(self.config.reward),
        }
        if self.train_cfg.objective == "opd":
            contract["opd_config"] = asdict(self.config.opd)
        return contract

    def _compute_batch_loss(self, batch: tuple[Any, ...]) -> torch.Tensor:
        if self.train_cfg.objective == "dpo":
            if len(batch) != 4:
                raise RuntimeError("DPO batch must contain chosen and rejected input/label tensors")
            return self._compute_dpo_loss(*batch)
        if self.train_cfg.objective == "reward_model":
            if len(batch) != 4:
                raise RuntimeError("reward_model batch must contain chosen and rejected input/label tensors")
            return self._compute_reward_model_loss(*batch)
        if self.train_cfg.objective == "opd":
            if len(batch) != 3:
                raise RuntimeError("OPD batch must contain prompt ids, prompt lengths, and optional expected strings")
            return self._compute_opd_loss(*batch)
        if self.train_cfg.objective == "grpo":
            if len(batch) != 3:
                raise RuntimeError("GRPO batch must contain prompt ids, prompt lengths, and expected responses")
            return self._compute_grpo_loss(*batch)
        if self.train_cfg.objective == "ppo":
            if len(batch) != 3:
                raise RuntimeError("PPO batch must contain prompt ids, prompt lengths, and expected responses")
            return self._compute_ppo_loss(*batch)
        if len(batch) != 2:
            raise RuntimeError("LM batch must contain input and label tensors")
        return self._compute_loss(batch[0], batch[1])

    def _compute_dpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.reference_model is None:
            raise RuntimeError("reference model is not loaded")
        policy_chosen_logps = sequence_logprobs(self.model(chosen_input_ids).logits, chosen_labels)
        policy_rejected_logps = sequence_logprobs(self.model(rejected_input_ids).logits, rejected_labels)
        with torch.no_grad():
            reference_chosen_logps = sequence_logprobs(self.reference_model(chosen_input_ids).logits, chosen_labels)
            reference_rejected_logps = sequence_logprobs(self.reference_model(rejected_input_ids).logits, rejected_labels)
        return dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            self.config.dpo,
        ).loss

    def _compute_reward_model_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> torch.Tensor:
        chosen_scores = self.model(chosen_input_ids, chosen_labels).scores
        rejected_scores = self.model(rejected_input_ids, rejected_labels).scores
        return reward_model_loss(chosen_scores, rejected_scores).loss

    def _compute_opd_loss(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        expected: list[str | None],
    ) -> torch.Tensor:
        if self.teacher_model is None:
            raise RuntimeError("teacher model is not loaded")
        cfg = self.config.opd.validated()
        rollout_input_ids, rollout_labels, _, _, _, _ = self._collect_rollout_batches(
            prompt_input_ids,
            prompt_lengths,
            expected,
            rollouts_per_prompt=cfg.num_rollouts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )
        student_logits = self.model(rollout_input_ids).logits
        with torch.no_grad():
            teacher_logits = self.teacher_model(rollout_input_ids).logits
        return on_policy_distillation_loss(student_logits, teacher_logits, rollout_labels, cfg).loss

    def _compute_grpo_loss(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        expected: list[str | None],
    ) -> torch.Tensor:
        if self.reference_model is None:
            raise RuntimeError("reference model is not loaded")
        rollout_input_ids, rollout_labels, rewards = self._build_grpo_rollouts(
            prompt_input_ids,
            prompt_lengths,
            expected,
        )
        policy_logps = sequence_logprobs(self.model(rollout_input_ids).logits, rollout_labels)
        with torch.no_grad():
            reference_logps = sequence_logprobs(self.reference_model(rollout_input_ids).logits, rollout_labels)
        reward_groups = rewards.view(prompt_input_ids.size(0), self.config.grpo.num_generations)
        advantages = group_advantages(
            reward_groups,
            normalize=self.config.grpo.normalize_advantages,
        ).reshape(-1)
        return grpo_loss(
            policy_logps,
            policy_logps.detach(),
            reference_logps,
            advantages,
            rewards,
            self.config.grpo,
        ).loss

    def _compute_ppo_loss(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        expected: list[str | None],
    ) -> torch.Tensor:
        if self.reference_model is None:
            raise RuntimeError("reference model is not loaded")
        cfg = self.config.ppo
        (
            rollout_input_ids,
            rollout_labels,
            old_logprobs,
            reference_logprobs,
            old_values,
            score_rewards,
        ) = self._build_ppo_rollouts(prompt_input_ids, prompt_lengths, expected)
        response_mask = rollout_labels.ne(IGNORE_INDEX)
        kl_rewards = -cfg.beta * (old_logprobs - reference_logprobs) * response_mask.to(dtype=old_logprobs.dtype)
        rewards = score_rewards + kl_rewards
        advantages, returns = compute_gae(
            rewards,
            old_values,
            response_mask,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        if cfg.normalize_advantages:
            advantages = normalize_masked(advantages, response_mask)

        policy_value_output = self.model(rollout_input_ids)
        policy_logprobs = token_logprobs(policy_value_output.logits, rollout_labels)
        entropy = token_entropy(policy_value_output.logits)
        return ppo_loss(
            policy_logprobs,
            old_logprobs,
            policy_value_output.values,
            old_values,
            returns,
            advantages,
            entropy,
            response_mask,
            cfg,
        ).loss

    def _build_ppo_rollouts(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        expected: list[str | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config.ppo.validated()
        (
            input_batch,
            label_batch,
            reward_input_batch,
            reward_label_batch,
            responses,
            targets,
        ) = self._collect_rollout_batches(
            prompt_input_ids,
            prompt_lengths,
            expected,
            rollouts_per_prompt=cfg.num_rollouts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )

        response_mask = label_batch.ne(IGNORE_INDEX)
        score_reward_batch = torch.zeros_like(label_batch, dtype=torch.float32)
        terminal_rewards = self._score_rollouts(reward_input_batch, reward_label_batch, responses, targets)
        for index, reward in enumerate(terminal_rewards):
            valid_positions = torch.nonzero(response_mask[index], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                raise RuntimeError("PPO rollout produced no trainable response tokens")
            score_reward_batch[index, valid_positions[-1]] = reward

        with torch.no_grad():
            was_training = self.model.training
            self.model.eval()
            try:
                policy_value_output = self.model(input_batch)
                old_logprobs = token_logprobs(policy_value_output.logits, label_batch).float()
                old_values = policy_value_output.values.float().masked_fill(~response_mask, 0.0)
                reference_logprobs = token_logprobs(self.reference_model(input_batch).logits, label_batch).float()
            finally:
                if was_training:
                    self.model.train()
        return input_batch, label_batch, old_logprobs, reference_logprobs, old_values, score_reward_batch

    def _build_grpo_rollouts(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        expected: list[str | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config.grpo.validated()
        (
            input_batch,
            label_batch,
            reward_input_batch,
            reward_label_batch,
            responses,
            targets,
        ) = self._collect_rollout_batches(
            prompt_input_ids,
            prompt_lengths,
            expected,
            rollouts_per_prompt=cfg.num_generations,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )
        reward_tensor = self._score_rollouts(reward_input_batch, reward_label_batch, responses, targets)
        return input_batch, label_batch, reward_tensor

    def _collect_rollout_batches(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        expected: list[str | None],
        *,
        rollouts_per_prompt: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str | None]]:
        was_training = self.model.training
        self.model.eval()
        rollout_inputs: list[torch.Tensor] = []
        rollout_labels: list[torch.Tensor] = []
        reward_inputs: list[torch.Tensor] = []
        reward_labels: list[torch.Tensor] = []
        responses: list[str] = []
        targets: list[str | None] = []
        try:
            for batch_index, target in enumerate(expected):
                prompt_len = int(prompt_lengths[batch_index].item())
                prompt = prompt_input_ids[batch_index, :prompt_len].unsqueeze(0)
                for _ in range(rollouts_per_prompt):
                    generated = self._raw_model().generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_id=self.tokenizer.eos_id,
                    )[0]
                    response_ids = generated[prompt_len:]
                    response = self.tokenizer.decode(response_ids.tolist())
                    responses.append(response)
                    targets.append(target)

                    input_ids = generated[:-1]
                    labels = generated[1:].clone()
                    labels[: max(prompt_len - 1, 0)] = IGNORE_INDEX
                    if input_ids.size(0) > self.model_cfg.context_length:
                        input_ids = input_ids[-self.model_cfg.context_length :]
                        labels = labels[-self.model_cfg.context_length :]
                    rollout_inputs.append(input_ids)
                    rollout_labels.append(labels)

                    reward_input_ids = generated
                    reward_label_ids = torch.full_like(generated, IGNORE_INDEX)
                    reward_label_ids[prompt_len:] = generated[prompt_len:]
                    if reward_input_ids.size(0) > self.model_cfg.context_length:
                        reward_input_ids = reward_input_ids[-self.model_cfg.context_length :]
                        reward_label_ids = reward_label_ids[-self.model_cfg.context_length :]
                    reward_inputs.append(reward_input_ids)
                    reward_labels.append(reward_label_ids)
        finally:
            if was_training:
                self.model.train()

        input_batch, label_batch = self._pad_rollout_pairs(rollout_inputs, rollout_labels)
        reward_input_batch, reward_label_batch = self._pad_rollout_pairs(reward_inputs, reward_labels)
        return input_batch, label_batch, reward_input_batch, reward_label_batch, responses, targets

    def _pad_rollout_pairs(
        self,
        inputs: list[torch.Tensor],
        labels: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(input_ids.size(0) for input_ids in inputs)
        input_batch = torch.full(
            (len(inputs), max_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        label_batch = torch.full(
            (len(labels), max_len),
            IGNORE_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        for index, (input_ids, label_ids) in enumerate(zip(inputs, labels, strict=True)):
            input_batch[index, : input_ids.size(0)] = input_ids
            label_batch[index, : label_ids.size(0)] = label_ids
        return input_batch, label_batch

    def _score_rollouts(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        responses: list[str],
        targets: list[str | None],
    ) -> torch.Tensor:
        if self.reward_scorer is None:
            raise RuntimeError("reward scorer is not initialized")
        rewards = self.reward_scorer.score(input_ids, labels, responses=responses, targets=targets)
        return rewards.to(device=self.device, dtype=torch.float32)

    def _move_batch_to_device(self, batch: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(self._move_item_to_device(item) for item in batch)

    def _move_item_to_device(self, item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return item.to(self.device, non_blocking=True)
        if isinstance(item, tuple):
            return tuple(self._move_item_to_device(value) for value in item)
        if isinstance(item, list):
            return item
        return item

    def _load_reference_model(self, checkpoint: str) -> AnilaLM:
        payload = load_checkpoint_payload(checkpoint, required_keys=("model",))
        reference = AnilaLM(self.model_cfg)
        lora_config = payload.get("lora_config")
        if isinstance(lora_config, dict) and lora_config.get("enabled", False):
            apply_lora(reference, LoRAConfig(**lora_config).validated())
        state_dict = payload["model"]
        missing, unexpected = reference.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Reference checkpoint is not compatible with the current model "
                f"(missing={missing}, unexpected={unexpected})"
            )
        reference.to(self.device).eval()
        for param in reference.parameters():
            param.requires_grad = False
        return reference

    def _runtime_model(self) -> torch.nn.Module:
        return getattr(self.model, "_orig_mod", self.model)

    def _value_model(self) -> PolicyValueModel:
        model = self._runtime_model()
        if not isinstance(model, PolicyValueModel):
            raise RuntimeError("policy-value model is not initialized")
        return model

    def _reward_model(self) -> RewardModel:
        model = self._runtime_model()
        if not isinstance(model, RewardModel):
            raise RuntimeError("reward model is not initialized")
        return model

    def _raw_model(self) -> AnilaLM:
        model = self._runtime_model()
        if isinstance(model, PolicyValueModel):
            return model.policy
        if isinstance(model, RewardModel):
            return model.backbone
        return model


def _clone_floating_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if torch.is_floating_point(tensor)
    }


def _state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _strip_state_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {name.removeprefix(prefix): tensor for name, tensor in state.items() if name.startswith(prefix)}


def _add_state_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{name}": tensor for name, tensor in state.items()}


def _required_ema_head(payload: dict[str, Any], key: str) -> dict[str, Any]:
    state = payload.get(key)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {key} state must be a dictionary")
    return state


def _load_ema_state_for_model(model: torch.nn.Module, loaded: dict[str, Any]) -> dict[str, torch.Tensor]:
    current = model.state_dict()
    expected = {name for name, tensor in current.items() if torch.is_floating_point(tensor)}
    missing = sorted(expected - set(loaded))
    unexpected = sorted(set(loaded) - expected)
    if missing or unexpected:
        raise ValueError(f"checkpoint EMA state is not compatible with the current model (missing={missing}, unexpected={unexpected})")

    ema_state: dict[str, torch.Tensor] = {}
    for name in sorted(expected):
        value = loaded[name]
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise ValueError("checkpoint EMA state must contain only floating point tensors")
        target = current[name]
        if value.shape != target.shape:
            raise ValueError(f"checkpoint EMA tensor shape does not match current model for {name}")
        ema_state[name] = value.detach().to(device=target.device, dtype=target.dtype).clone()
    return ema_state


def train(config: RunConfig) -> None:
    Trainer(config).train()
