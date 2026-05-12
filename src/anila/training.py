from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

from anila.config import RunConfig, TrainConfig
from anila.data import create_dataloader
from anila.model import AnilaLM
from anila.tokenization import AnilaTokenizer

console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def configure_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


def cycle(loader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


class CheckpointManager:
    def __init__(self, out_dir: str | Path):
        self.root = Path(out_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.pt"

    def save(self, payload: dict[str, Any], *, step: int) -> Path:
        step_path = self.root / f"step_{step:08d}.pt"
        self._atomic_save(payload, step_path)
        self._atomic_save(payload, self.latest_path)
        return step_path

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        load_path = Path(path) if path else self.latest_path
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        return torch.load(load_path, map_location="cpu")

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)


class Trainer:
    def __init__(self, config: RunConfig):
        self.config = config
        self.train_cfg = config.train
        self.device = resolve_device(self.train_cfg.device)
        self.dtype = resolve_dtype(self.train_cfg.dtype, self.device)
        set_seed(self.train_cfg.seed)

        self.tokenizer = AnilaTokenizer.load(self.train_cfg.tokenizer_path)
        self.model_cfg = replace(config.model, vocab_size=self.tokenizer.vocab_size).validated()
        self.model = AnilaLM(self.model_cfg).to(self.device)
        if self.train_cfg.compile:
            self.model = torch.compile(self.model)

        self.optimizer = configure_optimizer(self.model, self.train_cfg)
        self.checkpoints = CheckpointManager(self.train_cfg.out_dir)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and self.dtype == torch.float16)
        self.start_step = 0
        if self.train_cfg.resume:
            self._restore(self.train_cfg.resume)

        self.train_loader = create_dataloader(
            self.train_cfg.dataset_path,
            self.tokenizer,
            context_length=self.model_cfg.context_length,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
        )
        eval_path = self.train_cfg.val_dataset_path or self.train_cfg.dataset_path
        self.eval_loader = create_dataloader(
            eval_path,
            self.tokenizer,
            context_length=self.model_cfg.context_length,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
        )
        if len(self.train_loader) == 0:
            raise ValueError("Training dataloader is empty; reduce batch_size or add data")

    def train(self) -> None:
        console.print(
            f"[bold]Anila[/bold] training on {self.device} with dtype={self.dtype}, "
            f"params={self._raw_model().num_parameters() / 1e6:.2f}M"
        )
        iterator = cycle(self.train_loader)
        start_time = time.time()
        self.model.train()
        for step in range(self.start_step, self.train_cfg.max_steps):
            lr = cosine_lr(step, self.train_cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(self.train_cfg.grad_accum_steps):
                x, y = next(iterator)
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ):
                    loss = self.model(x, targets=y).loss
                    if loss is None:
                        raise RuntimeError("model did not return a training loss")
                    loss = loss / self.train_cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                loss_value += float(loss.detach().cpu()) * self.train_cfg.grad_accum_steps

            self.scaler.unscale_(self.optimizer)
            if self.train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            completed_step = step + 1
            if completed_step % self.train_cfg.log_interval == 0:
                elapsed = time.time() - start_time
                console.print(
                    f"step={completed_step:6d}/{self.train_cfg.max_steps} "
                    f"loss={loss_value:.4f} lr={lr:.2e} elapsed={elapsed:.1f}s"
                )
            if completed_step % self.train_cfg.eval_interval == 0:
                eval_loss = self.evaluate()
                console.print(f"eval step={completed_step:6d} loss={eval_loss:.4f}")
            if completed_step % self.train_cfg.save_interval == 0:
                self.save(completed_step)

        if self.train_cfg.max_steps % self.train_cfg.save_interval != 0:
            self.save(self.train_cfg.max_steps)

    @torch.no_grad()
    def evaluate(self) -> float:
        was_training = self.model.training
        self.model.eval()
        losses = []
        for index, (x, y) in enumerate(self.eval_loader):
            if index >= self.train_cfg.eval_batches:
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                loss = self.model(x, targets=y).loss
            if loss is not None:
                losses.append(float(loss.detach().cpu()))
        if was_training:
            self.model.train()
        return sum(losses) / max(len(losses), 1)

    def save(self, step: int) -> Path:
        payload = {
            "model": self._raw_model().state_dict(),
            "model_config": asdict(self.model_cfg),
            "train_config": asdict(self.train_cfg),
            "step": step,
            "optimizer": self.optimizer.state_dict(),
        }
        path = self.checkpoints.save(payload, step=step)
        console.print(f"saved checkpoint: {path}")
        return path

    def _restore(self, path: str) -> None:
        payload = self.checkpoints.load(path)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.start_step = int(payload.get("step", 0))

    def _raw_model(self) -> AnilaLM:
        return getattr(self.model, "_orig_mod", self.model)


def train(config: RunConfig) -> None:
    Trainer(config).train()
