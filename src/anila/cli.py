from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from anila.config import load_run_config
from anila.sampling import sample_text
from anila.tokenization import train_byte_bpe
from anila.training import train

app = typer.Typer(help="Anila: from-scratch language-model training.")


@app.command("train-tokenizer")
def train_tokenizer(
    input: Annotated[list[Path], typer.Option("--input", "-i", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", "-o")],
    vocab_size: Annotated[int, typer.Option("--vocab-size")] = 8192,
    min_frequency: Annotated[int, typer.Option("--min-frequency")] = 2,
) -> None:
    """Train a byte-level BPE tokenizer."""
    tokenizer = train_byte_bpe(input, out, vocab_size=vocab_size, min_frequency=min_frequency)
    typer.echo(f"saved tokenizer to {out} ({tokenizer.vocab_size} tokens)")


@app.command("train")
def train_model(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)],
) -> None:
    """Train a causal language model from a JSON or TOML config."""
    train(load_run_config(config))


@app.command()
def sample(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", exists=True, readable=True)],
    tokenizer: Annotated[Path, typer.Option("--tokenizer", "-t", exists=True, readable=True)],
    prompt: Annotated[str, typer.Option("--prompt", "-p")],
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 80,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.8,
    top_k: Annotated[int, typer.Option("--top-k")] = 50,
    top_p: Annotated[float, typer.Option("--top-p")] = 1.0,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    """Generate text from a checkpoint."""
    text = sample_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )
    typer.echo(text)


if __name__ == "__main__":
    app()
