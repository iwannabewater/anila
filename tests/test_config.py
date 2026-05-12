from pathlib import Path

import pytest

from anila.config import ModelConfig, RewardConfig, load_run_config


def test_model_config_fills_kv_heads() -> None:
    cfg = ModelConfig(n_embd=64, n_head=4, n_kv_head=None).validated()
    assert cfg.n_kv_head == 4


def test_invalid_attention_shape_fails() -> None:
    with pytest.raises(ValueError, match="n_embd"):
        ModelConfig(n_embd=65, n_head=4).validated()


def test_load_run_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """
        {
          "model": {"context_length": 8, "unknown": 1},
          "train": {"dataset_path": "x", "tokenizer_path": "y"}
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown config key"):
        load_run_config(path)


def test_load_sft_run_config(tmp_path: Path) -> None:
    path = tmp_path / "sft.json"
    path.write_text(
        """
        {
          "model": {"context_length": 32, "n_head": 2, "n_embd": 32},
          "train": {
            "objective": "sft",
            "dataset_path": ["train.jsonl", "more.jsonl"],
            "tokenizer_path": "tokenizer"
          },
          "lora": {"enabled": true, "rank": 4, "target_modules": ["q_proj", "v_proj"]},
          "distill": {"mode": "hard", "data_objective": "sft"},
          "dpo": {"beta": 0.2},
          "grpo": {"num_generations": 2, "max_new_tokens": 8},
          "ppo": {"num_rollouts": 2, "max_new_tokens": 8},
          "reward": {"scorer": "rule", "scale": 1.5},
          "sft": {"format": "auto"}
        }
        """,
        encoding="utf-8",
    )

    cfg = load_run_config(path)

    assert cfg.train.objective == "sft"
    assert cfg.train.dataset_path == ["train.jsonl", "more.jsonl"]
    assert cfg.lora.enabled is True
    assert cfg.lora.rank == 4
    assert cfg.distill.mode == "hard"
    assert cfg.distill.data_objective == "sft"
    assert cfg.dpo.beta == 0.2
    assert cfg.grpo.num_generations == 2
    assert cfg.grpo.max_new_tokens == 8
    assert cfg.ppo.num_rollouts == 2
    assert cfg.ppo.max_new_tokens == 8
    assert cfg.reward.scorer == "rule"
    assert cfg.reward.scale == 1.5
    assert cfg.train.allow_tf32 is True
    assert cfg.train.gradient_checkpointing is False
    assert cfg.train.fused_adamw is False
    assert cfg.sft.format == "auto"


def test_soft_distill_config_requires_teacher(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """
        {
          "train": {
            "objective": "distill",
            "dataset_path": "train.txt",
            "tokenizer_path": "tokenizer"
          },
          "distill": {"mode": "soft"}
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher_checkpoint"):
        load_run_config(path)


def test_reward_model_scorer_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="reward.checkpoint"):
        RewardConfig(scorer="model").validated()


def test_train_runtime_flags_must_be_booleans(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """
        {
          "train": {
            "dataset_path": "train.txt",
            "tokenizer_path": "tokenizer",
            "gradient_checkpointing": "yes"
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gradient_checkpointing"):
        load_run_config(path)


def test_train_optimizer_betas_must_be_valid(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """
        {
          "train": {
            "dataset_path": "train.txt",
            "tokenizer_path": "tokenizer",
            "beta2": 1.0
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="train.beta2"):
        load_run_config(path)
