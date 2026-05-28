# Anila

Anila is a compact, from-scratch language-model training repository built with PyTorch. It is designed to keep the important parts of a small LLM training pipeline visible: tokenizer training, data loading, model definition, optimization, checkpointing, and sampling.

The repository is intentionally small enough to study and modify, while still using production-shaped engineering practices: typed configuration, explicit validation, restricted checkpoint loading, atomic checkpoints, RNG-preserving evaluation, fast tests, and uv-managed environments.

## Features

- Byte-level BPE tokenizer training.
- GPT-style causal language model with RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings, KV-cache generation, streaming steps, top-k/top-p sampling, and native beam search.
- Single-process trainer with gradient accumulation, mixed precision, TF32 control, optional fused AdamW, optional activation checkpointing, cosine decay, RNG-preserving validation, checkpointed random state, resume, and atomic saves.
- Objective-aware training with plain-text pretraining, response-masked supervised fine-tuning, LoRA adapters, hard/soft distillation, DPO preference optimization, learned reward models, GRPO, and PPO with a value head.
- Pretraining data modes for dense sliding-window sampling, packed fixed-length blocks, and streaming local text files.
- JSON/TOML run configs and UTF-8 training inputs with strict validation, fail-fast errors, and optional checkpoint retention.
- Grouped CLI commands for tokenizer training, model training, evaluation, generation, checkpoint inspection, and LoRA checkpoint merge/export.
- A small top-level Python API for version checks, tokenizer training, native training, checkpoint evaluation, structured generation, streaming, and sampling.
- Fast unit tests plus end-to-end integration coverage.

## Requirements

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- CPU works for tests and quickstart runs. CUDA is used automatically when available.

## Quick Start

```bash
# Install the package, runtime dependencies, and development tools.
uv sync --group dev

# Confirm the installed CLI.
uv run anila --version

# Train a byte-level BPE tokenizer from the tiny local examples.
uv run anila tokenizer train \
  --input examples/tiny_corpus.txt \
  --input examples/tiny_sft.jsonl \
  --out runs/tokenizer \
  --vocab-size 512 \
  --min-frequency 1

# Run plain next-token pretraining.
uv run anila model train --config configs/quickstart/pretrain.json

# Run response-masked supervised fine-tuning.
uv run anila model train --config configs/quickstart/sft.json

# Fine-tune LoRA adapters from the pretraining checkpoint.
uv run anila model train --config configs/quickstart/lora-sft.json

# Train hard-label distillation on SFT-style data.
uv run anila model train --config configs/quickstart/distill-hard-sft.json

# Train soft-logit distillation from the pretraining checkpoint.
uv run anila model train --config configs/quickstart/distill-soft-pretrain.json

# Train DPO preference optimization from the SFT checkpoint.
uv run anila model train --config configs/quickstart/dpo.json

# Train a scalar reward model from chosen/rejected preference records.
uv run anila model train --config configs/quickstart/reward-model.json

# Train GRPO with a built-in rule reward.
uv run anila model train --config configs/quickstart/grpo-rule-reward.json

# Train PPO with a built-in rule reward.
uv run anila model train --config configs/quickstart/ppo-rule-reward.json

# Train GRPO with the learned reward-model checkpoint.
uv run anila model train --config configs/quickstart/grpo-learned-reward.json

# Train PPO with the learned reward-model checkpoint.
uv run anila model train --config configs/quickstart/ppo-learned-reward.json

# Export a LoRA checkpoint as a merged full-model checkpoint for plain inference.
uv run anila checkpoint merge-lora \
  --checkpoint runs/quickstart/lora-sft/checkpoints/latest.pt \
  --out runs/quickstart/lora-sft/checkpoints/merged.pt

# Generate text from a checkpoint.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is"

# Generate a deterministic beam-search continuation.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --num-beams 4 \
  --length-penalty 0.7 \
  --completion-only

# Print structured generation metadata with token logprobs.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 16 \
  --json \
  --logprobs

# Print a JSON checkpoint summary.
uv run anila checkpoint inspect \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt

# Measure pretraining loss/perplexity on a local evaluation file.
uv run anila model evaluate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_corpus.txt \
  --task lm \
  --objective pretrain
```

The pretraining quickstart config writes checkpoints under `runs/quickstart/pretrain/`. Each run also writes a reproducibility snapshot to `config.json` and structured metrics to `metrics.jsonl` under its output directory. Training outputs are ignored by Git.

The canonical CLI is grouped by resource: `anila tokenizer train`, `anila model train`, `anila model evaluate`, `anila model generate`, `anila checkpoint inspect`, and `anila checkpoint merge-lora`. `anila --version` prints the installed package version. Older flat commands (`train-tokenizer`, `train`, `sample`, `inspect-checkpoint`, `merge-lora-checkpoint`) remain available as compatibility aliases.

## Python API

Common entry points are exported from the package root for lightweight scripts and notebooks:

```python
from pathlib import Path

from anila import generate_text, load_run_config, train, train_byte_bpe

train_byte_bpe([Path("examples/tiny_corpus.txt")], Path("runs/tokenizer"), vocab_size=512, min_frequency=1)
train(load_run_config(Path("configs/quickstart/pretrain.json")))
result = generate_text(
    checkpoint=Path("runs/quickstart/pretrain/checkpoints/latest.pt"),
    tokenizer_path=Path("runs/tokenizer"),
    prompt="Anila is",
    max_new_tokens=32,
    stop_strings=["\n"],
    return_logprobs=True,
)
text = result.text
```

Lower-level modules remain importable when extending objectives, data adapters, or model internals. Checkpoint artifact reads in library code should go through Anila's checkpoint helpers rather than ad hoc `torch.load` calls.

## Quality Checks

```bash
# Check import order, lint rules, and common bug patterns.
uv run ruff check .

# Run the unit and integration test suite.
uv run pytest
```

## Repository Layout

```text
src/anila/
  config.py        typed JSON/TOML config loading and validation
  tokenization.py  byte-level BPE tokenizer training/loading
  data.py          tokenized text datasets and dataloaders
  distillation.py  teacher loading and soft-logit distillation loss
  dpo.py           DPO preference optimization
  grpo.py          GRPO reward utilities and loss
  model.py         native PyTorch GPT model
  peft.py          LoRA adapter injection and extraction
  ppo.py           PPO value head and loss utilities
  reward.py        reward model and reward scorer adapters
  training.py      trainer, schedule, checkpointing, resume
  evaluation.py    checkpoint evaluation metrics
  sampling.py      checkpoint loading and text generation
  cli.py           command-line interface
configs/           runnable training configs, including quickstart recipes
examples/          tiny local corpus for quickstart and integration tests
docs/              architecture and development notes
tests/             fast unit and integration tests
```

## Configuration

Run configs live under `configs/` and contain these top-level sections:

- `model`: vocabulary-independent architecture settings.
- `train`: objective, dataset, tokenizer, runtime, optimizer, evaluation, and checkpoint settings.
- `data`: pretraining data mode and sequence-window controls.
- `lora`: optional adapter configuration.
- `distill`: optional hard or soft distillation settings.
- `dpo`: optional Direct Preference Optimization settings.
- `grpo`: optional Group Relative Policy Optimization settings.
- `ppo`: optional Proximal Policy Optimization settings.
- `reward`: optional reward scorer and reward-model data settings.
- `sft`: supervised fine-tuning record formatting settings.

See `configs/quickstart/pretrain.json` for pretraining, `configs/quickstart/sft.json` for full-model supervised fine-tuning, `configs/quickstart/lora-sft.json` for LoRA SFT, `configs/quickstart/distill-hard-sft.json` for hard distillation, `configs/quickstart/distill-soft-pretrain.json` for soft-logit distillation, `configs/quickstart/dpo.json` for DPO, `configs/quickstart/reward-model.json` for reward model training, `configs/quickstart/grpo-rule-reward.json` and `configs/quickstart/ppo-rule-reward.json` for rule-reward RL, and `configs/quickstart/grpo-learned-reward.json` plus `configs/quickstart/ppo-learned-reward.json` for learned-reward RL.

Useful runtime flags in `train`:

- `allow_tf32`: enables CUDA TF32 matmul and cuDNN kernels when available.
- `gradient_checkpointing`: recomputes transformer blocks during backward to reduce activation memory.
- `fused_adamw`: requests PyTorch fused AdamW on CUDA and falls back to ordinary AdamW elsewhere.
- `keep_last_checkpoints`: when set, keeps only the most recent N step checkpoints plus `latest.pt` to limit local disk growth.

Generation uses a native KV cache by default, so sampling only evaluates the newest token after the initial prefill. Pass `use_cache=False` to `AnilaLM.generate` when comparing against the plain full-context path. The native generation path also supports greedy decoding, seeded sampling, top-k, top-p, min-p, repetition penalty, streaming single-path steps, and deterministic beam search through `num_beams`. In batched single-path generation, rows that emit `eos_id` remain terminal while unfinished rows continue. The sampling API layers text-level stop strings, structured generation metadata, optional token logprobs, and `stream_text` on top of the native model loop without changing the default `sample_text` string return.

## Data Modes

The `data` section controls how plain-text pretraining examples are produced:

- `sliding_window`: the default. Builds dense overlapping next-token windows and supports `sequence_stride` when less overlap is desired.
- `packed`: builds non-overlapping fixed-length blocks from a token stream, which is usually the practical default for larger local corpora.
- `streaming`: reads local text files through an iterable dataset and emits packed blocks without materializing the whole corpus as one tensor.

All tokenizer and dataset text inputs are decoded as strict UTF-8; malformed input fails before it can silently alter a training corpus.

```json
{
  "data": {
    "pretrain_mode": "packed"
  }
}
```

Use `streaming` when the corpus is too large to hold as one token tensor:

```json
{
  "data": {
    "pretrain_mode": "streaming"
  }
}
```

## Training Objectives

- `pretrain`: default objective. Trains next-token prediction on one or more plain-text files.
- `sft`: supervised fine-tuning objective. Reads JSONL records and masks prompt/user/system tokens so only assistant response tokens contribute to loss.
- `distill`: distillation objective. In `hard` mode, trains on teacher-generated hard labels through the configured data objective. In `soft` mode, loads a teacher checkpoint and trains with masked KL divergence, optionally mixed with CE loss.
- `dpo`: preference optimization objective. Reads JSONL prompt/chosen/rejected records, scores chosen and rejected response tokens under policy and reference models, and applies DPO loss.
- `reward_model`: trains a scalar reward model from prompt/chosen/rejected preference records with a pairwise Bradley-Terry loss.
- `grpo`: online policy optimization objective. Reads JSONL prompt records, samples a group of responses per prompt, scores them with a rule or learned reward scorer, normalizes rewards within each prompt group, and applies clipped GRPO loss with reference KL.
- `ppo`: online policy optimization objective. Reads JSONL prompt records, samples responses, applies reward scorer outputs plus reference KL penalties, computes GAE with a learned value head, and applies clipped PPO policy/value losses.

SFT records may use either prompt/response fields:

```json
{"prompt": "What does Anila train?", "response": "Small causal language models."}
```

or chat messages:

```json
{"messages": [{"role": "user", "content": "What is SFT?"}, {"role": "assistant", "content": "Supervised fine-tuning."}]}
```

## Evaluation

`anila model evaluate` restores a native checkpoint and prints JSON metrics. The current harness covers:

- `--task lm`: token-weighted negative log-likelihood and perplexity for `--objective pretrain` or `--objective sft`.
- `--task preference`: chosen-vs-rejected policy accuracy and mean log-probability margin on DPO-style records.
- `--task reward`: chosen-vs-rejected reward-model accuracy and mean score margin.

```bash
# Measure pretraining loss/perplexity.
uv run anila model evaluate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_corpus.txt \
  --task lm \
  --objective pretrain

# Measure policy preference accuracy.
uv run anila model evaluate \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_preferences.jsonl \
  --task preference

# Measure reward-model pairwise accuracy.
uv run anila model evaluate \
  --checkpoint runs/quickstart/reward-model/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_preferences.jsonl \
  --task reward
```

## LoRA

LoRA is enabled through the optional `lora` config section. When `train_base` is false, Anila freezes the base model and trains only adapter weights. Full checkpoints remain directly sampleable, and adapter-only artifacts are also saved under `checkpoints/adapters/`.

```json
{
  "train": {
    "objective": "sft",
    "init_from": "runs/quickstart/pretrain/checkpoints/latest.pt"
  },
  "lora": {
    "enabled": true,
    "rank": 4,
    "alpha": 8.0,
    "target_modules": ["q_proj", "v_proj"]
  }
}
```

Adapter checkpoints can be folded into a plain full-model checkpoint:

```bash
# Merge LoRA weights into base Linear weights and disable the lora_config flag.
uv run anila checkpoint merge-lora \
  --checkpoint runs/quickstart/lora-sft/checkpoints/latest.pt \
  --out runs/quickstart/lora-sft/checkpoints/merged.pt
```

## Distillation

Hard distillation uses teacher-generated data and ordinary hard-label loss:

```json
{
  "train": {"objective": "distill"},
  "distill": {"mode": "hard", "data_objective": "sft"}
}
```

Soft distillation loads a native Anila teacher checkpoint and matches its logits:

```json
{
  "train": {"objective": "distill"},
  "distill": {
    "mode": "soft",
    "data_objective": "pretrain",
    "teacher_checkpoint": "runs/quickstart/pretrain/checkpoints/latest.pt",
    "temperature": 2.0,
    "kl_weight": 1.0,
    "ce_weight": 0.5
  }
}
```

## DPO

DPO expects a policy initialized from a checkpoint and a frozen reference model. If `dpo.reference_checkpoint` is omitted, Anila uses `train.init_from` as the reference checkpoint.

```json
{
  "train": {
    "objective": "dpo",
    "init_from": "runs/quickstart/sft/checkpoints/latest.pt",
    "dataset_path": "examples/tiny_preferences.jsonl"
  },
  "dpo": {
    "beta": 0.1
  }
}
```

Preference records use:

```json
{"prompt": "What does Anila train?", "chosen": "Small causal language models.", "rejected": "Image databases."}
```

## Reward Models

Reward model training uses the same prompt/chosen/rejected preference format as DPO, but trains a scalar reward head instead of a policy objective. If `train.init_from` is set, the reward model starts from that native Anila checkpoint.

```json
{
  "train": {
    "objective": "reward_model",
    "init_from": "runs/quickstart/sft/checkpoints/latest.pt",
    "dataset_path": "examples/tiny_preferences.jsonl"
  }
}
```

GRPO and PPO can then load the reward checkpoint through the top-level `reward` section:

```json
{
  "reward": {
    "scorer": "model",
    "checkpoint": "runs/quickstart/reward-model/checkpoints/latest.pt"
  }
}
```

## GRPO

GRPO expects a policy initialized from a checkpoint and a frozen reference model. If `grpo.reference_checkpoint` is omitted, Anila uses `train.init_from` as the reference checkpoint.

```json
{
  "train": {
    "objective": "grpo",
    "init_from": "runs/quickstart/sft/checkpoints/latest.pt",
    "dataset_path": "examples/tiny_grpo_prompts.jsonl"
  },
  "grpo": {
    "num_generations": 4,
    "max_new_tokens": 24,
    "reward_type": "contains"
  }
}
```

Prompt reward records use:

```json
{"prompt": "What does Anila train?", "expected": "language models"}
```

When `reward.scorer` is `model`, prompt records may omit `expected` because the loaded reward model scores generated responses directly.

## PPO

PPO expects a policy initialized from a checkpoint and a frozen reference model. If `ppo.reference_checkpoint` is omitted, Anila uses `train.init_from` as the reference checkpoint. PPO checkpoints keep the base LM weights in `model` for direct sampling and store the value head separately in `value_head`.

```json
{
  "train": {
    "objective": "ppo",
    "init_from": "runs/quickstart/sft/checkpoints/latest.pt",
    "dataset_path": "examples/tiny_ppo_prompts.jsonl"
  },
  "ppo": {
    "num_rollouts": 1,
    "max_new_tokens": 24,
    "reward_type": "contains",
    "gae_lambda": 0.95
  }
}
```

## Documentation

- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Project Status](docs/status.md)
- [Changelog](CHANGELOG.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
