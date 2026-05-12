from dataclasses import asdict
from pathlib import Path

import torch
from typer.testing import CliRunner

from anila.checkpoint import inspect_checkpoint
from anila.cli import app
from anila.config import ModelConfig, TrainConfig


def test_inspect_checkpoint_summarizes_native_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "reward_model",
            "step": 3,
            "model": {},
            "optimizer": {},
            "model_config": asdict(ModelConfig(vocab_size=128, context_length=64, n_layer=1, n_head=2, n_embd=32)),
            "train_config": asdict(
                TrainConfig(dataset_path="data.jsonl", tokenizer_path="tokenizer", objective="reward_model")
            ),
            "tokenizer_path": "tokenizer",
            "lora_config": {"enabled": False},
            "value_head": None,
            "reward_head": {"weight": torch.zeros(1, 32)},
            "adapter_checkpoint": None,
        },
        checkpoint,
    )

    summary = inspect_checkpoint(checkpoint)

    assert summary["objective"] == "reward_model"
    assert summary["step"] == 3
    assert summary["has_model"] is True
    assert summary["has_reward_head"] is True
    assert summary["model"]["context_length"] == 64
    assert summary["train"]["objective"] == "reward_model"


def test_inspect_checkpoint_cli_prints_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "pretrain",
            "step": 1,
            "model": {},
            "model_config": asdict(ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=2, n_embd=32)),
            "train_config": asdict(TrainConfig(dataset_path="data.txt", tokenizer_path="tokenizer")),
            "tokenizer_path": "tokenizer",
        },
        checkpoint,
    )

    result = CliRunner().invoke(app, ["inspect-checkpoint", "--checkpoint", str(checkpoint)])

    assert result.exit_code == 0
    assert '"objective": "pretrain"' in result.output
