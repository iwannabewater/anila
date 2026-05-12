from dataclasses import asdict
from pathlib import Path

import torch
from typer.testing import CliRunner

from anila.cli import app
from anila.config import LoRAConfig, ModelConfig, RewardConfig, SFTConfig
from anila.evaluation import evaluate_lm_checkpoint, evaluate_policy_preferences, evaluate_reward_model
from anila.model import AnilaLM
from anila.reward import RewardModel
from anila.tokenization import train_byte_bpe


def _tokenizer(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\n"
        "Assistant: Anila trains small language models.\n"
        "Checkpoints are saved atomically.\n"
        "Preference records compare chosen and rejected answers.\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    return train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1), tokenizer_dir


def _policy_checkpoint(tmp_path: Path, cfg: ModelConfig, extra_payload: dict[str, object] | None = None) -> Path:
    checkpoint = tmp_path / "policy.pt"
    model = AnilaLM(cfg)
    payload = {
        "schema_version": 1,
        "objective": "pretrain",
        "model": model.state_dict(),
        "model_config": asdict(cfg),
        "lora_config": asdict(LoRAConfig()),
        "tokenizer_path": "tokenizer",
        "step": 0,
    }
    if extra_payload:
        payload.update(extra_payload)
    torch.save(
        payload,
        checkpoint,
    )
    return checkpoint


def _preference_data(tmp_path: Path) -> Path:
    data = tmp_path / "prefs.jsonl"
    data.write_text(
        '{"prompt": "What does Anila train?", "chosen": "Small language models.", "rejected": "Image databases."}\n'
        '{"prompt": "How are checkpoints saved?", "chosen": "Atomically.", "rejected": "Never."}\n',
        encoding="utf-8",
    )
    return data


def test_evaluate_lm_checkpoint_reports_loss_and_perplexity(tmp_path: Path) -> None:
    _, tokenizer_dir = _tokenizer(tmp_path)
    data = tmp_path / "eval.txt"
    data.write_text("Anila trains small language models.\n" * 10, encoding="utf-8")
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _policy_checkpoint(tmp_path, cfg)

    metrics = evaluate_lm_checkpoint(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        dataset_path=str(data),
        objective="pretrain",
        batch_size=2,
        max_batches=1,
        device="cpu",
    )

    assert metrics["task"] == "lm"
    assert metrics["objective"] == "pretrain"
    assert metrics["num_tokens"] > 0
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 1


def test_evaluate_lm_checkpoint_uses_sft_config_from_checkpoint(tmp_path: Path) -> None:
    _, tokenizer_dir = _tokenizer(tmp_path)
    data = tmp_path / "sft.jsonl"
    data.write_text('{"instruction": "What does Anila train?", "completion": "Small language models."}\n', encoding="utf-8")
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    sft_config = SFTConfig(
        format="prompt_response",
        prompt_key="instruction",
        response_key="completion",
    ).validated()
    checkpoint = _policy_checkpoint(tmp_path, cfg, extra_payload={"sft_config": asdict(sft_config)})

    metrics = evaluate_lm_checkpoint(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        dataset_path=str(data),
        objective="sft",
        batch_size=1,
        device="cpu",
    )

    assert metrics["objective"] == "sft"
    assert metrics["num_tokens"] > 0


def test_evaluate_policy_preferences_reports_accuracy(tmp_path: Path) -> None:
    _, tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _policy_checkpoint(tmp_path, cfg)

    metrics = evaluate_policy_preferences(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        dataset_path=str(_preference_data(tmp_path)),
        batch_size=2,
        device="cpu",
    )

    assert metrics["task"] == "preference"
    assert metrics["num_pairs"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_reward_model_reports_accuracy(tmp_path: Path) -> None:
    _, tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    reward_model = RewardModel(AnilaLM(cfg))
    checkpoint = tmp_path / "reward.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "reward_model",
            "model": reward_model.backbone.state_dict(),
            "model_config": asdict(cfg),
            "lora_config": asdict(LoRAConfig()),
            "reward_config": asdict(RewardConfig()),
            "reward_head": reward_model.reward_head.state_dict(),
            "tokenizer_path": "tokenizer",
            "step": 0,
        },
        checkpoint,
    )

    metrics = evaluate_reward_model(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        dataset_path=str(_preference_data(tmp_path)),
        batch_size=2,
        device="cpu",
    )

    assert metrics["task"] == "reward"
    assert metrics["num_pairs"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_cli_prints_json_metrics(tmp_path: Path) -> None:
    _, tokenizer_dir = _tokenizer(tmp_path)
    data = tmp_path / "eval.txt"
    data.write_text("Anila trains small language models.\n" * 10, encoding="utf-8")
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _policy_checkpoint(tmp_path, cfg)

    result = CliRunner().invoke(
        app,
        [
            "model",
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--dataset",
            str(data),
            "--batch-size",
            "2",
            "--max-batches",
            "1",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert '"task": "lm"' in result.output
    assert '"perplexity"' in result.output
