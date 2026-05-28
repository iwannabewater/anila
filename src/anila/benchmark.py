from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from anila.config import load_mapping
from anila.evaluation import evaluate_lm_checkpoint, evaluate_policy_preferences, evaluate_reward_model

SUPPORTED_BENCHMARK_TASKS = {"lm", "preference", "reward"}
SUPPORTED_LM_OBJECTIVES = {"pretrain", "sft"}


@dataclass(frozen=True)
class BenchmarkTaskConfig:
    name: str
    task: str
    dataset_path: str | list[str]
    objective: str = "pretrain"
    batch_size: int | None = None
    max_batches: int | None = None

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
    ):
        values = [float(result["metrics"][metric_name]) for result in results if result["task"] == task_name]
        if values:
            summary[output_name] = sum(values) / len(values)
    return summary


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
