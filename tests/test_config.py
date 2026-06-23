from pathlib import Path

import pytest

from anila.config import DataConfig, ModelConfig, OPDConfig, RewardConfig, TrainConfig, load_run_config


def test_model_config_fills_kv_heads() -> None:
    cfg = ModelConfig(n_embd=64, n_head=4, n_kv_head=None).validated()
    assert cfg.n_kv_head == 4


def test_model_config_rejects_zero_kv_heads() -> None:
    with pytest.raises(ValueError, match="n_kv_head"):
        ModelConfig(n_embd=64, n_head=4, n_kv_head=0).validated()


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


def test_data_config_validates_pretrain_mode() -> None:
    with pytest.raises(ValueError, match="data.pretrain_mode"):
        DataConfig(pretrain_mode="unknown").validated()


def test_data_config_rejects_stride_outside_sliding_window() -> None:
    with pytest.raises(ValueError, match="sequence_stride"):
        DataConfig(pretrain_mode="packed", sequence_stride=8).validated()


def test_quickstart_configs_are_loadable() -> None:
    config_dir = Path("configs/quickstart")
    names = sorted(path.name for path in config_dir.glob("*.json"))

    assert names == [
        "distill-hard-sft.json",
        "distill-soft-pretrain.json",
        "dpo.json",
        "grpo-learned-reward.json",
        "grpo-rule-reward.json",
        "lora-sft.json",
        "opd.json",
        "ppo-learned-reward.json",
        "ppo-rule-reward.json",
        "pretrain.json",
        "reward-model.json",
        "sft.json",
    ]
    for path in config_dir.glob("*.json"):
        assert load_run_config(path).train.out_dir.startswith("runs/quickstart/")


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
          "data": {"pretrain_mode": "packed"},
          "lora": {"enabled": true, "rank": 4, "target_modules": ["q_proj", "v_proj"]},
          "distill": {"mode": "hard", "data_objective": "sft"},
          "dpo": {"beta": 0.2},
          "grpo": {"num_generations": 2, "max_new_tokens": 8},
          "opd": {"teacher_checkpoint": "teacher.pt", "num_rollouts": 2, "max_new_tokens": 8},
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
    assert cfg.data.pretrain_mode == "packed"
    assert cfg.lora.enabled is True
    assert cfg.lora.rank == 4
    assert cfg.distill.mode == "hard"
    assert cfg.distill.data_objective == "sft"
    assert cfg.dpo.beta == 0.2
    assert cfg.grpo.num_generations == 2
    assert cfg.grpo.max_new_tokens == 8
    assert cfg.opd.teacher_checkpoint == "teacher.pt"
    assert cfg.opd.num_rollouts == 2
    assert cfg.opd.max_new_tokens == 8
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


def test_opd_config_requires_teacher_when_objective_is_opd(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """
        {
          "train": {
            "objective": "opd",
            "dataset_path": "prompts.jsonl",
            "tokenizer_path": "tokenizer"
          }
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="opd.teacher_checkpoint"):
        load_run_config(path)


def test_opd_config_validates_rollout_settings() -> None:
    with pytest.raises(ValueError, match="opd.num_rollouts"):
        OPDConfig(teacher_checkpoint="teacher.pt", num_rollouts=0).validated()


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


def test_train_config_validates_checkpoint_retention() -> None:
    cfg = TrainConfig(
        dataset_path="train.txt",
        tokenizer_path="tokenizer",
        keep_last_checkpoints=2,
    ).validated()
    assert cfg.keep_last_checkpoints == 2

    with pytest.raises(ValueError, match="keep_last_checkpoints"):
        TrainConfig(dataset_path="train.txt", tokenizer_path="tokenizer", keep_last_checkpoints=0).validated()
    with pytest.raises(ValueError, match="keep_last_checkpoints"):
        TrainConfig(dataset_path="train.txt", tokenizer_path="tokenizer", keep_last_checkpoints=True).validated()


def test_train_config_validates_ema_decay() -> None:
    cfg = TrainConfig(dataset_path="train.txt", tokenizer_path="tokenizer", ema_decay=0.999).validated()
    assert cfg.ema_decay == 0.999

    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(dataset_path="train.txt", tokenizer_path="tokenizer", ema_decay=1.0).validated()
    with pytest.raises(ValueError, match="ema_decay"):
        TrainConfig(dataset_path="train.txt", tokenizer_path="tokenizer", ema_decay=True).validated()
