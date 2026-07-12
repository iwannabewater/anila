import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from anila.benchmark import _iter_tool_call_records, evaluate_benchmark_suite, load_benchmark_suite
from anila.cli import app
from anila.config import LoRAConfig, ModelConfig, SFTConfig
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


def test_quickstart_benchmark_suite_includes_tool_call_task() -> None:
    suite = load_benchmark_suite(Path("configs/benchmarks/quickstart.json"))

    assert {task.task for task in suite.tasks} == {"lm", "preference", "tool_call"}
    assert any(task.name == "tiny-tool-call" for task in suite.tasks)


def test_load_benchmark_suite_validates_tool_call_generation_fields(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "tools",
                        "task": "tool_call",
                        "dataset_path": "tools.jsonl",
                        "max_new_tokens": 8,
                        "temperature": 0.7,
                        "top_k": 20,
                        "top_p": 0.9,
                        "do_sample": False,
                        "open_thinking": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_benchmark_suite(suite)

    assert loaded.tasks[0].task == "tool_call"
    assert loaded.tasks[0].max_new_tokens == 8

    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "lm",
                        "task": "lm",
                        "dataset_path": "data.txt",
                        "max_new_tokens": 8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation benchmark fields"):
        load_benchmark_suite(suite)

    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "tools",
                        "task": "tool_call",
                        "dataset_path": "tools.jsonl",
                        "temperature": math.nan,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="temperature"):
        load_benchmark_suite(suite)

    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "tools",
                        "task": "tool_call",
                        "dataset_path": "tools.jsonl",
                        "top_p": math.inf,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top_p"):
        load_benchmark_suite(suite)


def test_tool_call_benchmark_rejects_non_finite_json_records(tmp_path: Path) -> None:
    data = tmp_path / "tools.jsonl"
    data.write_text('{"prompt": "Use a tool.", "expected": {"answers": [NaN]}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON constant"):
        list(_iter_tool_call_records(str(data)))


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


def test_evaluate_benchmark_suite_runs_tool_call_task(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "tool_calls.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Use the calculator."}],
                "tools": [{"type": "function", "function": {"name": "calculate_math"}}],
                "expected": {"answers": ["4"], "tools": ["calculate_math"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "name": "tool-suite",
                "tasks": [
                    {
                        "name": "calculator",
                        "task": "tool_call",
                        "dataset_path": str(data),
                        "max_new_tokens": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_load_model_tokenizer_payload(**_kwargs):
        return object(), object(), torch.device("cpu"), {}

    def fake_generate_chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            assistant=SimpleNamespace(
                raw='<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call>\n4',
                tool_calls=({"name": "calculate_math", "arguments": {"expression": "2+2"}},),
                invalid_tool_calls=(),
            )
        )

    monkeypatch.setattr("anila.benchmark._load_model_tokenizer_payload", fake_load_model_tokenizer_payload)
    monkeypatch.setattr("anila.benchmark._generate_chat_with_model", fake_generate_chat)

    metrics = evaluate_benchmark_suite(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        suite=suite,
        device="cpu",
    )

    assert calls[0]["max_new_tokens"] == 12
    assert calls[0]["tools"] == [{"type": "function", "function": {"name": "calculate_math"}}]
    assert metrics["summary"]["tool_call_mean_accuracy"] == 1.0
    result = metrics["results"][0]
    assert result["task"] == "tool_call"
    assert result["primary_metric"] == "accuracy"
    assert result["metrics"]["count"] == 1
    assert result["metrics"]["tool_call_rate"] == 1.0
    assert result["primary_value"] == 1.0


def test_tool_call_benchmark_accuracy_requires_answers(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "tool_calls.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Use the calculator."}],
                "tools": [{"type": "function", "function": {"name": "calculate_math"}}],
                "expected": {"answers": ["4"], "tools": ["calculate_math"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "calculator",
                        "task": "tool_call",
                        "dataset_path": str(data),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_load_model_tokenizer_payload(**_kwargs):
        return object(), object(), torch.device("cpu"), {}

    def fake_generate_chat(**_kwargs):
        return SimpleNamespace(
            assistant=SimpleNamespace(
                raw='<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>',
                tool_calls=({"name": "calculate_math", "arguments": {}},),
                invalid_tool_calls=(),
            )
        )

    monkeypatch.setattr("anila.benchmark._load_model_tokenizer_payload", fake_load_model_tokenizer_payload)
    monkeypatch.setattr("anila.benchmark._generate_chat_with_model", fake_generate_chat)

    metrics = evaluate_benchmark_suite(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        suite=suite,
        device="cpu",
    )

    assert metrics["results"][0]["metrics"]["accuracy"] == 0.0
    assert metrics["results"][0]["primary_value"] == 0.0


def test_tool_call_benchmark_max_batches_stops_before_later_invalid_record(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "tool_calls.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Use the calculator."}],
                "tools": [{"type": "function", "function": {"name": "calculate_math"}}],
                "expected": {"answers": ["4"], "tools": ["calculate_math"]},
            }
        )
        + "\n"
        + "not json\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "calculator",
                        "task": "tool_call",
                        "dataset_path": str(data),
                        "max_batches": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_load_model_tokenizer_payload(**_kwargs):
        return object(), object(), torch.device("cpu"), {}

    def fake_generate_chat(**_kwargs):
        return SimpleNamespace(
            assistant=SimpleNamespace(
                raw='<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\n4',
                tool_calls=({"name": "calculate_math", "arguments": {}},),
                invalid_tool_calls=(),
            )
        )

    monkeypatch.setattr("anila.benchmark._load_model_tokenizer_payload", fake_load_model_tokenizer_payload)
    monkeypatch.setattr("anila.benchmark._generate_chat_with_model", fake_generate_chat)

    metrics = evaluate_benchmark_suite(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        suite=suite,
        device="cpu",
    )

    assert metrics["results"][0]["metrics"]["count"] == 1
    assert metrics["results"][0]["primary_value"] == 1.0


def test_tool_call_benchmark_uses_checkpoint_sft_config_for_scoring(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "tool_calls.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "Use the calculator."}],
                "expected": {"answers": ["4"], "tools": ["calculate_math"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"tasks": [{"name": "calculator", "task": "tool_call", "dataset_path": str(data)}]}),
        encoding="utf-8",
    )
    sft_config = SFTConfig(tool_call_start="<call>", tool_call_end="</call>").validated()
    load_calls = 0

    def fake_load_model_tokenizer_payload(**_kwargs):
        nonlocal load_calls
        load_calls += 1
        return object(), object(), torch.device("cpu"), {"sft_config": asdict(sft_config)}

    def fake_generate_chat(**kwargs):
        assert kwargs["cfg"].tool_call_start == "<call>"
        return SimpleNamespace(
            assistant=SimpleNamespace(
                raw='<call>{"name":"calculate_math","arguments":{}}</call>\n4',
                tool_calls=({"name": "calculate_math", "arguments": {}},),
                invalid_tool_calls=(),
            )
        )

    monkeypatch.setattr("anila.benchmark._load_model_tokenizer_payload", fake_load_model_tokenizer_payload)
    monkeypatch.setattr("anila.benchmark._generate_chat_with_model", fake_generate_chat)

    metrics = evaluate_benchmark_suite(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        suite=suite,
        device="cpu",
    )

    assert load_calls == 1
    assert metrics["results"][0]["primary_value"] == 1.0


def test_evaluate_tool_call_benchmark_runs_native_generation(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)
    data = tmp_path / "tool_calls.jsonl"
    data.write_text(
        json.dumps(
            {
                "prompt": "Use the calculator to compute 2+2.",
                "tools": [{"type": "function", "function": {"name": "calculate_math"}}],
                "expected": {"answers": ["4"], "tools": ["calculate_math"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "calculator",
                        "task": "tool_call",
                        "dataset_path": str(data),
                        "max_new_tokens": 1,
                        "do_sample": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_benchmark_suite(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        suite=suite,
        device="cpu",
    )

    task_metrics = metrics["results"][0]["metrics"]
    assert task_metrics["count"] == 1
    assert 0.0 <= task_metrics["accuracy"] <= 1.0
    assert 0.0 <= task_metrics["tool_call_rate"] <= 1.0
    assert 0.0 <= task_metrics["invalid_tool_call_rate"] <= 1.0


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
