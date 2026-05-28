from __future__ import annotations

import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from anila._version import __version__
from anila.checkpoint import inspect_checkpoint, merge_lora_checkpoint
from anila.config import load_run_config
from anila.evaluation import evaluate_lm_checkpoint, evaluate_policy_preferences, evaluate_reward_model
from anila.sampling import generate_text, sample_text, stream_text
from anila.tokenization import train_byte_bpe
from anila.training import train

app = typer.Typer(help="Anila: from-scratch language-model training.", no_args_is_help=True)
tokenizer_app = typer.Typer(help="Tokenizer commands.")
model_app = typer.Typer(help="Model training and generation commands.")
checkpoint_app = typer.Typer(help="Checkpoint inspection and export commands.")

app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(model_app, name="model")
app.add_typer(checkpoint_app, name="checkpoint")


class EvaluationTask(StrEnum):
    lm = "lm"
    preference = "preference"
    reward = "reward"


class LanguageModelObjective(StrEnum):
    pretrain = "pretrain"
    sft = "sft"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"anila {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            help="Print the installed Anila version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Run Anila commands."""


@tokenizer_app.command("train")
@app.command("train-tokenizer", hidden=True)
def train_tokenizer(
    input: Annotated[list[Path], typer.Option("--input", "-i", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", "-o")],
    vocab_size: Annotated[int, typer.Option("--vocab-size")] = 8192,
    min_frequency: Annotated[int, typer.Option("--min-frequency")] = 2,
) -> None:
    """Train a byte-level BPE tokenizer."""
    tokenizer = train_byte_bpe(input, out, vocab_size=vocab_size, min_frequency=min_frequency)
    typer.echo(f"saved tokenizer to {out} ({tokenizer.vocab_size} tokens)")


@model_app.command("train")
@app.command("train", hidden=True)
def train_model(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)],
) -> None:
    """Train a causal language model from a JSON or TOML config."""
    train(load_run_config(config))


@model_app.command("generate")
@app.command(hidden=True)
def sample(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", exists=True, readable=True)],
    tokenizer: Annotated[Path, typer.Option("--tokenizer", "-t", exists=True, readable=True)],
    prompt: Annotated[str, typer.Option("--prompt", "-p")],
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 80,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.8,
    top_k: Annotated[int, typer.Option("--top-k", help="Use 0 to disable top-k filtering.")] = 50,
    top_p: Annotated[float, typer.Option("--top-p")] = 1.0,
    min_p: Annotated[float, typer.Option("--min-p")] = 0.0,
    repetition_penalty: Annotated[float, typer.Option("--repetition-penalty")] = 1.0,
    num_beams: Annotated[int, typer.Option("--num-beams", help="Use values above 1 for deterministic beam search.")] = 1,
    length_penalty: Annotated[float, typer.Option("--length-penalty")] = 1.0,
    device: Annotated[str, typer.Option("--device")] = "auto",
    do_sample: Annotated[bool, typer.Option("--sample/--greedy")] = True,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    return_full_text: Annotated[bool, typer.Option("--full-text/--completion-only")] = True,
    stop: Annotated[list[str] | None, typer.Option("--stop", help="Stop when this text appears in the completion.")] = None,
    return_logprobs: Annotated[bool, typer.Option("--logprobs", help="Include generated token logprobs in JSON output.")] = False,
    json_output: Annotated[bool, typer.Option("--json/--text", help="Print structured generation metadata as JSON.")] = False,
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="Stream generated text chunks as they are decoded.")] = False,
) -> None:
    """Generate text from a checkpoint."""
    if top_k < 0:
        raise typer.BadParameter("top_k must be non-negative; use 0 to disable top-k filtering", param_hint="--top-k")
    if num_beams <= 0:
        raise typer.BadParameter("num_beams must be positive", param_hint="--num-beams")
    if length_penalty < 0:
        raise typer.BadParameter("length_penalty cannot be negative", param_hint="--length-penalty")
    if stream and num_beams > 1:
        raise typer.BadParameter("streaming generation is only supported with --num-beams 1", param_hint="--stream")
    if stream and json_output:
        raise typer.BadParameter("--stream cannot be combined with --json", param_hint="--stream")
    if return_logprobs and not json_output:
        raise typer.BadParameter("--logprobs requires --json", param_hint="--logprobs")
    if stream:
        if return_full_text:
            typer.echo(prompt, nl=False)
        for chunk in stream_text(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None if top_k == 0 else top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            device=device,
            do_sample=do_sample,
            seed=seed,
            stop_strings=stop,
        ):
            typer.echo(chunk, nl=False)
        typer.echo()
        return
    if json_output:
        result = generate_text(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None if top_k == 0 else top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            length_penalty=length_penalty,
            device=device,
            do_sample=do_sample,
            seed=seed,
            return_full_text=return_full_text,
            stop_strings=stop,
            return_logprobs=return_logprobs,
        )
        typer.echo(json.dumps(asdict(result), indent=2, sort_keys=True))
        return
    text = sample_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=None if top_k == 0 else top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        length_penalty=length_penalty,
        device=device,
        do_sample=do_sample,
        seed=seed,
        return_full_text=return_full_text,
        stop_strings=stop,
    )
    typer.echo(text)


@model_app.command("evaluate")
def evaluate_model(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", exists=True, readable=True)],
    tokenizer: Annotated[Path, typer.Option("--tokenizer", "-t", exists=True, readable=True)],
    dataset: Annotated[Path, typer.Option("--dataset", "-d", exists=True, readable=True)],
    task: Annotated[EvaluationTask, typer.Option("--task")] = EvaluationTask.lm,
    objective: Annotated[LanguageModelObjective, typer.Option("--objective")] = LanguageModelObjective.pretrain,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 8,
    max_batches: Annotated[int | None, typer.Option("--max-batches")] = None,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    """Evaluate a checkpoint and print JSON metrics."""
    if task is EvaluationTask.lm:
        metrics = evaluate_lm_checkpoint(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            dataset_path=str(dataset),
            objective=objective.value,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
        )
    elif task is EvaluationTask.preference:
        metrics = evaluate_policy_preferences(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            dataset_path=str(dataset),
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
        )
    elif task is EvaluationTask.reward:
        metrics = evaluate_reward_model(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            dataset_path=str(dataset),
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
        )
    else:
        raise typer.BadParameter("task must be lm, preference, or reward", param_hint="--task")
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@checkpoint_app.command("inspect")
@app.command("inspect-checkpoint", hidden=True)
def inspect_checkpoint_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", exists=True, readable=True)],
) -> None:
    """Print a JSON summary of a native Anila checkpoint."""
    typer.echo(json.dumps(inspect_checkpoint(checkpoint), indent=2, sort_keys=True))


@checkpoint_app.command("merge-lora")
@app.command("merge-lora-checkpoint", hidden=True)
def merge_lora_checkpoint_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", "-o")],
) -> None:
    """Export a LoRA checkpoint as a merged full-model checkpoint."""
    path = merge_lora_checkpoint(checkpoint, out)
    typer.echo(f"saved merged checkpoint to {path}")


if __name__ == "__main__":
    app()
