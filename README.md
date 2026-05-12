# Anila

Anila is a compact, from-scratch language-model training repository built with PyTorch. It is designed to keep the important parts of a small LLM training pipeline visible: tokenizer training, data loading, model definition, optimization, checkpointing, and sampling.

The repository is intentionally small enough to study and modify, while still using production-shaped engineering practices: typed configuration, explicit validation, atomic checkpoints, deterministic setup, fast tests, and uv-managed environments.

## Features

- Byte-level BPE tokenizer training.
- GPT-style causal language model with RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings, and top-k/top-p sampling.
- Single-process trainer with gradient accumulation, mixed precision, cosine decay, validation, checkpointing, resume, and atomic saves.
- JSON/TOML run configs with strict validation and fail-fast errors.
- CLI commands for tokenizer training, model training, and sampling.
- Fast unit tests plus an end-to-end smoke test.

## Requirements

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- CPU works for tests and smoke runs. CUDA is used automatically when available.

## Quick Start

```bash
uv sync --group dev

uv run anila train-tokenizer \
  --input examples/tiny_corpus.txt \
  --out runs/tokenizer \
  --vocab-size 512 \
  --min-frequency 1

uv run anila train --config configs/smoke.json

uv run anila sample \
  --checkpoint runs/smoke/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is"
```

The smoke config writes checkpoints under `runs/smoke/`. Training outputs are ignored by Git.

## Quality Checks

```bash
uv run ruff check .
uv run pytest
```

## Repository Layout

```text
src/anila/
  config.py        typed JSON/TOML config loading and validation
  tokenization.py  byte-level BPE tokenizer training/loading
  data.py          tokenized text datasets and dataloaders
  model.py         native PyTorch GPT model
  training.py      trainer, schedule, checkpointing, resume
  sampling.py      checkpoint loading and text generation
  cli.py           command-line interface
configs/           runnable training configs
examples/          tiny local corpus for smoke tests
docs/              architecture and development notes
tests/             fast unit and smoke tests
```

## Configuration

Run configs live under `configs/` and contain two top-level sections:

- `model`: vocabulary-independent architecture settings.
- `train`: dataset, tokenizer, runtime, optimizer, evaluation, and checkpoint settings.

See `configs/smoke.json` for a minimal runnable example.

## Documentation

- [Architecture](docs/architecture.md)
- [Development](docs/development.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
