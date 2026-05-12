# Architecture

Anila is organized as a small set of explicit modules with narrow contracts. The goal is to keep model and training logic easy to inspect while still separating concerns that change at different rates.

```text
corpus files
    |
    v
Tokenizer -----> TextTokenDataset
    |                  |
    v                  v
RunConfig --------> Trainer --------> CheckpointManager
                         |
                         v
                      AnilaLM
                         |
                         v
                      sampler
```

## Module Responsibilities

- `config`: parses JSON/TOML configs, applies defaults, and rejects unknown keys.
- `tokenization`: trains and loads byte-level BPE tokenizers.
- `data`: builds contiguous next-token prediction examples from text corpora.
- `model`: implements the causal language model and generation-time filtering.
- `training`: owns device/dtype selection, optimizer setup, learning-rate schedule, evaluation, checkpointing, and resume.
- `sampling`: restores checkpoints and exposes text generation.

## Runtime Contract

- Configs fail before training starts when shapes, intervals, or unsupported dtypes are invalid.
- Checkpoints are ordinary `torch.save` dictionaries with model state, model config, train config, optimizer state, and completed step.
- `latest.pt` and step checkpoints are written through a temporary file and atomically replaced.
- Tokenizer vocabulary size is loaded from the tokenizer artifact and becomes the model vocabulary at runtime.

## Extension Points

- Add dataset adapters under `data.py` or a `datasets/` package once there are multiple real formats.
- Add new model variants by keeping the `forward(input_ids, targets=None)` interface.
- Add supervised fine-tuning as a vertical slice: dataset adapter, loss masking, config section, tests, and CLI command.
- Add distributed training behind a separate runtime module after single-process coverage is stronger.
