import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from anila.benchmark import evaluate_benchmark_suite, load_benchmark_suite
from anila.cli import app
from anila.config import LoRAConfig, ModelConfig
from anila.model import AnilaLM
from anila.tokenization import train_byte_bpe


def _tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Anila trains compact language models.\n"
        "Preference records compare chosen and rejected answers.\n"
        "Checkpoints can be evaluated in benchmark suites.\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    return tokenizer_dir


def _checkpoint(tmp_path: Path, cfg: ModelConfig, *, include_ema: bool = False) -> Path:
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
    if include_ema:
        payload["ema_model"] = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        payload["ema_decay"] = 0.99
    torch.save(payload, checkpoint)
    return checkpoint


def _preference_data(tmp_path: Path) -> Path:
    data = tmp_path / "prefs.jsonl"
    data.write_text(
        '{"prompt": "What does Anila train?", "chosen": "Compact language models.", "rejected": "Image datasets."}\n',
        encoding="utf-8",
    )
    return data


def _suite(tmp_path: Path, lm_data: Path, preference_data: Path) -> Path:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tiny-suite",
                "tasks": [
                    {
                        "name": "tiny-lm",
                        "task": "lm",
                        "dataset_path": str(lm_data),
                        "objective": "pretrain",
                        "batch_size": 2,
                        "max_batches": 1,
                    },
                    {
                        "name": "tiny-preference",
                        "task": "preference",
                        "dataset_path": str(preference_data),
                        "batch_size": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return suite


def test_load_benchmark_suite_validates_tasks(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"name": "bad-suite", "tasks": [{"name": "x", "task": "lm", "dataset_path": "data.txt"}]}),
        encoding="utf-8",
    )

    loaded = load_benchmark_suite(suite)

    assert loaded.name == "bad-suite"
    assert loaded.tasks[0].name == "x"

    suite.write_text(
        json.dumps({"tasks": [{"name": "x", "task": "unknown", "dataset_path": "data.txt"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="benchmark task"):
        load_benchmark_suite(suite)


def test_evaluate_benchmark_suite_runs_lm_and_preference(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    lm_data = tmp_path / "eval.txt"
    lm_data.write_text("Anila trains compact language models.\n" * 8, encoding="utf-8")
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg, include_ema=True)
    suite = _suite(tmp_path, lm_data, _preference_data(tmp_path))

    metrics = evaluate_benchmark_suite(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        suite=suite,
        batch_size=2,
        max_batches=1,
        device="cpu",
        use_ema=True,
    )

    assert metrics["suite"] == "tiny-suite"
    assert metrics["weights"] == "ema"
    assert metrics["num_tasks"] == 2
    assert {result["name"] for result in metrics["results"]} == {"tiny-lm", "tiny-preference"}
    assert "lm_mean_perplexity" in metrics["summary"]
    assert "preference_mean_accuracy" in metrics["summary"]


def test_benchmark_cli_prints_json(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    lm_data = tmp_path / "eval.txt"
    lm_data.write_text("Anila trains compact language models.\n" * 8, encoding="utf-8")
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg, include_ema=True)
    suite = _suite(tmp_path, lm_data, _preference_data(tmp_path))

    result = CliRunner().invoke(
        app,
        [
            "model",
            "benchmark",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--suite",
            str(suite),
            "--max-batches",
            "1",
            "--device",
            "cpu",
            "--ema",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["suite"] == "tiny-suite"
    assert payload["weights"] == "ema"
    assert payload["num_tasks"] == 2
