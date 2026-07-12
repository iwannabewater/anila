from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from itertools import islice
from pathlib import Path
from typing import Any

from anila._json import dumps_strict_json, loads_strict_json
from anila.config import load_mapping
from anila.evaluation import evaluate_lm_checkpoint, evaluate_policy_preferences, evaluate_reward_model
from anila.reward import score_response, tool_call_response_succeeds
from anila.sampling import _generate_chat_with_model, _load_model_tokenizer_payload, _sft_config_from_payload

SUPPORTED_BENCHMARK_TASKS = {"lm", "preference", "reward", "tool_call"}
SUPPORTED_LM_OBJECTIVES = {"pretrain", "sft"}


@dataclass(frozen=True)
class BenchmarkTaskConfig:
    name: str
    task: str
    dataset_path: str | list[str]
    objective: str = "pretrain"
    batch_size: int | None = None
    max_batches: int | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    open_thinking: bool | None = None

    def validated(self) -> BenchmarkTaskConfig:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("benchmark task name must be a non-empty string")
        if self.task not in SUPPORTED_BENCHMARK_TASKS:
            raise ValueError(f"benchmark task must be one of: {', '.join(sorted(SUPPORTED_BENCHMARK_TASKS))}")
        _validate_dataset_path(self.dataset_path, f"benchmark task {self.name}.dataset_path")
        if self.task == "lm" and self.objective not in SUPPORTED_LM_OBJECTIVES:
            raise ValueError(f"LM benchmark objective must be one of: {', '.join(sorted(SUPPORTED_LM_OBJECTIVES))}")
        if self.task != "lm" and self.objective != "pretrain":
            raise ValueError("benchmark objective is only supported for LM tasks")
        generation_fields = ("max_new_tokens", "temperature", "top_k", "top_p", "do_sample", "open_thinking")
        if self.task != "tool_call" and any(getattr(self, name) is not None for name in generation_fields):
            raise ValueError("generation benchmark fields are only supported for tool_call tasks")
        if self.task == "tool_call":
            if self.batch_size is not None:
                raise ValueError("benchmark task batch_size is not supported for tool_call tasks")
            if self.max_new_tokens is not None:
                _validate_positive_int(self.max_new_tokens, f"benchmark task {self.name}.max_new_tokens")
            if self.temperature is not None and (
                isinstance(self.temperature, bool)
                or not isinstance(self.temperature, int | float)
                or self.temperature <= 0
                or not math.isfinite(self.temperature)
            ):
                raise ValueError(f"benchmark task {self.name}.temperature must be finite and positive")
            if self.top_k is not None:
                _validate_positive_int(self.top_k, f"benchmark task {self.name}.top_k")
            if self.top_p is not None and (
                isinstance(self.top_p, bool)
                or not isinstance(self.top_p, int | float)
                or not math.isfinite(self.top_p)
                or not 0.0 < self.top_p <= 1.0
            ):
                raise ValueError(f"benchmark task {self.name}.top_p must be finite and in (0, 1]")
            for name in ("do_sample", "open_thinking"):
                value = getattr(self, name)
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"benchmark task {self.name}.{name} must be a boolean")
        if self.batch_size is not None:
            _validate_positive_int(self.batch_size, f"benchmark task {self.name}.batch_size")
        if self.max_batches is not None:
            _validate_positive_int(self.max_batches, f"benchmark task {self.name}.max_batches")
        return self


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    name: str = "benchmark"
    tasks: list[BenchmarkTaskConfig] = field(default_factory=list)

    def validated(self) -> BenchmarkSuiteConfig:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("benchmark suite name must be a non-empty string")
        if not self.tasks:
            raise ValueError("benchmark suite requires at least one task")
        seen: set[str] = set()
        validated_tasks: list[BenchmarkTaskConfig] = []
        for task in self.tasks:
            validated = task.validated()
            if validated.name in seen:
                raise ValueError(f"duplicate benchmark task name: {validated.name}")
            seen.add(validated.name)
            validated_tasks.append(validated)
        return BenchmarkSuiteConfig(name=self.name, tasks=validated_tasks)


def load_benchmark_suite(path: str | Path) -> BenchmarkSuiteConfig:
    data = load_mapping(path)
    unknown = sorted(set(data) - {"name", "tasks"})
    if unknown:
        raise ValueError(f"Unknown benchmark suite key(s): {', '.join(unknown)}")
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Benchmark suite requires a tasks list")
    return BenchmarkSuiteConfig(
        name=data.get("name", "benchmark"),
        tasks=[_task_from_mapping(index, values) for index, values in enumerate(tasks)],
    ).validated()


def evaluate_benchmark_suite(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    suite: str | Path | BenchmarkSuiteConfig,
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "auto",
    use_ema: bool = False,
) -> dict[str, Any]:
    _validate_positive_int(batch_size, "batch_size")
    if max_batches is not None:
        _validate_positive_int(max_batches, "max_batches")
    suite_config = load_benchmark_suite(suite) if isinstance(suite, str | Path) else suite.validated()
    results = [
        _evaluate_task(
            task,
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            default_batch_size=batch_size,
            max_batches_override=max_batches,
            device=device,
            use_ema=use_ema,
        )
        for task in suite_config.tasks
    ]
    return {
        "suite": suite_config.name,
        "checkpoint": str(checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "weights": "ema" if use_ema else "model",
        "num_tasks": len(results),
        "summary": _summarize(results),
        "results": results,
    }


def _task_from_mapping(index: int, values: Any) -> BenchmarkTaskConfig:
    if not isinstance(values, dict):
        raise ValueError(f"Benchmark task at index {index} must be an object")
    allowed = {config_field.name for config_field in fields(BenchmarkTaskConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown benchmark task key(s) at index {index}: {', '.join(unknown)}")
    return BenchmarkTaskConfig(**values)


def _evaluate_task(
    task: BenchmarkTaskConfig,
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    default_batch_size: int,
    max_batches_override: int | None,
    device: str,
    use_ema: bool,
) -> dict[str, Any]:
    batch_size = task.batch_size if task.batch_size is not None else default_batch_size
    max_batches = max_batches_override if max_batches_override is not None else task.max_batches
    if task.task == "lm":
        metrics = evaluate_lm_checkpoint(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            dataset_path=task.dataset_path,
            objective=task.objective,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            use_ema=use_ema,
        )
        primary_metric = "perplexity"
    elif task.task == "preference":
        metrics = evaluate_policy_preferences(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            dataset_path=task.dataset_path,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            use_ema=use_ema,
        )
        primary_metric = "accuracy"
    elif task.task == "reward":
        metrics = evaluate_reward_model(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            dataset_path=task.dataset_path,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            use_ema=use_ema,
        )
        primary_metric = "accuracy"
    elif task.task == "tool_call":
        metrics = _evaluate_tool_call_task(
            task,
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_path,
            max_records=max_batches,
            device=device,
            use_ema=use_ema,
        )
        primary_metric = "accuracy"
    else:
        raise ValueError(f"unsupported benchmark task: {task.task}")
    return {
        "name": task.name,
        "task": task.task,
        "primary_metric": primary_metric,
        "primary_value": metrics[primary_metric],
        "metrics": metrics,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task_name, metric_name, output_name in (
        ("lm", "loss", "lm_mean_loss"),
        ("lm", "perplexity", "lm_mean_perplexity"),
        ("preference", "accuracy", "preference_mean_accuracy"),
        ("reward", "accuracy", "reward_mean_accuracy"),
        ("tool_call", "accuracy", "tool_call_mean_accuracy"),
        ("tool_call", "mean_reward", "tool_call_mean_reward"),
    ):
        values = [float(result["metrics"][metric_name]) for result in results if result["task"] == task_name]
        if values:
            summary[output_name] = sum(values) / len(values)
    return summary


def _evaluate_tool_call_task(
    task: BenchmarkTaskConfig,
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    max_records: int | None,
    device: str,
    use_ema: bool,
) -> dict[str, Any]:
    scores: list[float] = []
    successes = 0
    records_with_calls = 0
    records_with_invalid_calls = 0
    model, tokenizer, runtime_device, payload = _load_model_tokenizer_payload(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
        use_ema=use_ema,
    )
    cfg = _sft_config_from_payload(payload)
    records = _iter_tool_call_records(task.dataset_path)
    if max_records is not None:
        records = islice(records, max_records)
    for record in records:
        generated = _generate_chat_with_model(
            model=model,
            tokenizer=tokenizer,
            runtime_device=runtime_device,
            messages=record["messages"],
            tools=record["tools"],
            cfg=cfg,
            max_new_tokens=task.max_new_tokens or 64,
            temperature=task.temperature if task.temperature is not None else 0.8,
            top_k=task.top_k if task.top_k is not None else 50,
            top_p=task.top_p if task.top_p is not None else 1.0,
            do_sample=task.do_sample if task.do_sample is not None else False,
            open_thinking=task.open_thinking if task.open_thinking is not None else False,
        )
        assistant = generated.assistant
        scores.append(score_response(assistant.raw, record["expected"], "tool_call", sft_config=cfg))
        successes += int(tool_call_response_succeeds(assistant.raw, record["expected"], sft_config=cfg))
        records_with_calls += int(bool(assistant.tool_calls))
        records_with_invalid_calls += int(bool(assistant.invalid_tool_calls))
    if not scores:
        raise ValueError("tool_call benchmark dataset is empty")
    accuracy = successes / len(scores)
    return {
        "count": len(scores),
        "accuracy": accuracy,
        "mean_reward": sum(scores) / len(scores),
        "tool_call_rate": records_with_calls / len(scores),
        "invalid_tool_call_rate": records_with_invalid_calls / len(scores),
    }


def _iter_tool_call_records(dataset_path: str | list[str]) -> Iterator[dict[str, Any]]:
    paths = [dataset_path] if isinstance(dataset_path, str) else dataset_path
    for input_path in paths:
        path = Path(input_path)
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = loads_strict_json(line)
                except ValueError as exc:
                    detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {detail}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"tool_call benchmark record in {path}:{line_number} must be an object")
                yield _tool_call_record_from_mapping(record, path=path, line_number=line_number)


def _tool_call_record_from_mapping(record: dict[str, Any], *, path: Path, line_number: int) -> dict[str, Any]:
    messages = _tool_call_record_messages(record, path=path, line_number=line_number)
    tools = _tool_call_record_tools(record, path=path, line_number=line_number)
    expected = _tool_call_record_expected(record, path=path, line_number=line_number)
    return {"messages": messages, "tools": tools, "expected": expected}


def _tool_call_record_messages(record: dict[str, Any], *, path: Path, line_number: int) -> list[dict[str, Any]]:
    if "messages" in record:
        if "prompt" in record:
            raise ValueError(f"tool_call benchmark record in {path}:{line_number} cannot mix messages and prompt")
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"tool_call benchmark record in {path}:{line_number} requires a non-empty messages list")
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError(f"tool_call benchmark messages in {path}:{line_number} must be objects")
        return messages
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"tool_call benchmark record in {path}:{line_number} requires prompt or messages")
    messages = []
    if "system" in record:
        system = record["system"]
        if not isinstance(system, str) or not system:
            raise ValueError(f"tool_call benchmark record in {path}:{line_number} field 'system' must be a string")
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _tool_call_record_tools(record: dict[str, Any], *, path: Path, line_number: int) -> list[dict[str, Any]] | None:
    if "tools" not in record:
        return None
    tools = record["tools"]
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"tool_call benchmark record in {path}:{line_number} field 'tools' must be a non-empty list")
    if not all(isinstance(tool, dict) for tool in tools):
        raise ValueError(f"tool_call benchmark tools in {path}:{line_number} must be objects")
    return tools


def _tool_call_record_expected(record: dict[str, Any], *, path: Path, line_number: int) -> str:
    if "expected" not in record:
        raise ValueError(f"tool_call benchmark record in {path}:{line_number} requires expected")
    expected = record["expected"]
    if isinstance(expected, str):
        if not expected:
            raise ValueError(f"tool_call benchmark record in {path}:{line_number} expected cannot be empty")
        return expected
    if isinstance(expected, dict | list):
        if not expected:
            raise ValueError(f"tool_call benchmark record in {path}:{line_number} expected cannot be empty")
        try:
            return dumps_strict_json(expected, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tool_call benchmark expected in {path}:{line_number} must be JSON serializable with finite numbers"
            ) from exc
    raise ValueError(f"tool_call benchmark record in {path}:{line_number} expected must be a string, object, or list")


def _validate_dataset_path(value: str | list[str], name: str) -> None:
    if isinstance(value, str):
        if value:
            return
        raise ValueError(f"{name} must be a non-empty string or list of strings")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty string or list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} entries must be non-empty strings")


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
