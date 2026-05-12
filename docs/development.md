# Development

## Environment

Anila uses uv for Python and dependency management:

```bash
uv sync --group dev
```

Run commands through uv so the local package and locked dependencies are used:

```bash
uv run anila --help
```

## Quality Gates

```bash
uv run ruff check .
uv run pytest
```

Run the end-to-end smoke path before larger changes:

```bash
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

## Checkpoint Contract

Training checkpoints are ordinary `torch.save` dictionaries:

- `model`: model state dict.
- `model_config`: model config as plain data.
- `train_config`: train config as plain data.
- `step`: completed optimizer step.
- `optimizer`: optimizer state dict.

`latest.pt` is written atomically and is safe to use for resume or sampling after a completed save.

## Release Hygiene

Do not commit generated training artifacts, checkpoints, caches, local virtual environments, or experiment logs. They are ignored by `.gitignore` and should remain reproducible from committed configs and source.
