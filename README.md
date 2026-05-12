# Anila

Anila is a compact, from-scratch language-model training repository built with PyTorch. It is designed to keep the important parts of a small LLM training pipeline visible: tokenizer training, data loading, model definition, optimization, checkpointing, and sampling.

The repository is intentionally small enough to study and modify, while still using production-shaped engineering practices: typed configuration, explicit validation, atomic checkpoints, deterministic setup, fast tests, and uv-managed environments.

## Features

- Byte-level BPE tokenizer training.
- GPT-style causal language model with RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings, KV-cache generation, and top-k/top-p sampling.
- Single-process trainer with gradient accumulation, mixed precision, TF32 control, optional fused AdamW, optional activation checkpointing, cosine decay, validation, checkpointing, resume, and atomic saves.
- Objective-aware training with plain-text pretraining, response-masked supervised fine-tuning, LoRA adapters, hard/soft distillation, DPO preference optimization, learned reward models, GRPO, and PPO with a value head.
- JSON/TOML run configs with strict validation and fail-fast errors.
- Grouped CLI commands for tokenizer training, model training, generation, checkpoint inspection, and LoRA checkpoint merge/export.
- Fast unit tests plus an end-to-end smoke test.

## Requirements

- Python 3.11 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- CPU works for tests and smoke runs. CUDA is used automatically when available.

## Quick Start

```bash
# Install the package, runtime dependencies, and development tools.
uv sync --group dev

# Train a byte-level BPE tokenizer from the tiny local examples.
uv run anila tokenizer train \
  --input examples/tiny_corpus.txt \
  --input examples/tiny_sft.jsonl \
  --out runs/tokenizer \
  --vocab-size 512 \
  --min-frequency 1

# Run plain next-token pretraining.
uv run anila model train --config configs/smoke.json

# Run response-masked supervised fine-tuning.
uv run anila model train --config configs/sft-smoke.json

# Fine-tune LoRA adapters from the pretraining checkpoint.
uv run anila model train --config configs/lora-sft-smoke.json

# Train hard-label and soft-logit distillation runs.
uv run anila model train --config configs/distill-hard-sft-smoke.json
uv run anila model train --config configs/distill-soft-smoke.json

# Train preference, reward-model, and online RL smoke runs.
uv run anila model train --config configs/dpo-smoke.json
uv run anila model train --config configs/reward-model-smoke.json
uv run anila model train --config configs/grpo-smoke.json
uv run anila model train --config configs/ppo-smoke.json
uv run anila model train --config configs/grpo-learned-reward-smoke.json
uv run anila model train --config configs/ppo-learned-reward-smoke.json

# Export a LoRA checkpoint as a merged full-model checkpoint for plain inference.
uv run anila checkpoint merge-lora \
  --checkpoint runs/lora-sft-smoke/checkpoints/latest.pt \
  --out runs/lora-sft-smoke/checkpoints/merged.pt

# Generate text from a checkpoint.
uv run anila model generate \
  --checkpoint runs/ppo-smoke/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is"

# Print a JSON checkpoint summary.
uv run anila checkpoint inspect \
  --checkpoint runs/ppo-smoke/checkpoints/latest.pt
```

The smoke config writes checkpoints under `runs/smoke/`. Each run also writes a reproducibility snapshot to `config.json` and structured metrics to `metrics.jsonl` under its output directory. Training outputs are ignored by Git.

The canonical CLI is grouped by resource: `anila tokenizer train`, `anila model train`, `anila model generate`, `anila checkpoint inspect`, and `anila checkpoint merge-lora`. Older flat commands (`train-tokenizer`, `train`, `sample`, `inspect-checkpoint`, `merge-lora-checkpoint`) remain available as compatibility aliases.

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
  distillation.py  teacher loading and soft-logit distillation loss
  dpo.py           DPO preference optimization
  grpo.py          GRPO reward utilities and loss
  model.py         native PyTorch GPT model
  peft.py          LoRA adapter injection and extraction
  ppo.py           PPO value head and loss utilities
  reward.py        reward model and reward scorer adapters
  training.py      trainer, schedule, checkpointing, resume
  sampling.py      checkpoint loading and text generation
  cli.py           command-line interface
configs/           runnable training configs
examples/          tiny local corpus for smoke tests
docs/              architecture and development notes
tests/             fast unit and smoke tests
```

## Configuration

Run configs live under `configs/` and contain these top-level sections:

- `model`: vocabulary-independent architecture settings.
- `train`: objective, dataset, tokenizer, runtime, optimizer, evaluation, and checkpoint settings.
- `lora`: optional adapter configuration.
- `distill`: optional hard or soft distillation settings.
- `dpo`: optional Direct Preference Optimization settings.
- `grpo`: optional Group Relative Policy Optimization settings.
- `ppo`: optional Proximal Policy Optimization settings.
- `reward`: optional reward scorer and reward-model data settings.
- `sft`: supervised fine-tuning record formatting settings.

See `configs/smoke.json` for pretraining, `configs/sft-smoke.json` for full-model supervised fine-tuning, `configs/lora-sft-smoke.json` for LoRA SFT, `configs/distill-hard-sft-smoke.json` for hard distillation, `configs/distill-soft-smoke.json` for soft-logit distillation, `configs/dpo-smoke.json` for DPO, `configs/reward-model-smoke.json` for reward model training, `configs/grpo-smoke.json` and `configs/ppo-smoke.json` for rule-reward RL, and `configs/grpo-learned-reward-smoke.json` plus `configs/ppo-learned-reward-smoke.json` for learned-reward RL.

Useful runtime flags in `train`:

- `allow_tf32`: enables CUDA TF32 matmul and cuDNN kernels when available.
- `gradient_checkpointing`: recomputes transformer blocks during backward to reduce activation memory.
- `fused_adamw`: requests PyTorch fused AdamW on CUDA and falls back to ordinary AdamW elsewhere.

Generation uses a native KV cache by default, so sampling only evaluates the newest token after the initial prefill. Pass `use_cache=False` to `AnilaLM.generate` when comparing against the plain full-context path.

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

## LoRA

LoRA is enabled through the optional `lora` config section. When `train_base` is false, Anila freezes the base model and trains only adapter weights. Full checkpoints remain directly sampleable, and adapter-only artifacts are also saved under `checkpoints/adapters/`.

```json
{
  "train": {
    "objective": "sft",
    "init_from": "runs/smoke/checkpoints/latest.pt"
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
  --checkpoint runs/lora-sft-smoke/checkpoints/latest.pt \
  --out runs/lora-sft-smoke/checkpoints/merged.pt
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
    "teacher_checkpoint": "runs/smoke/checkpoints/latest.pt",
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
    "init_from": "runs/sft-smoke/checkpoints/latest.pt",
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
    "init_from": "runs/sft-smoke/checkpoints/latest.pt",
    "dataset_path": "examples/tiny_preferences.jsonl"
  }
}
```

GRPO and PPO can then load the reward checkpoint through the top-level `reward` section:

```json
{
  "reward": {
    "scorer": "model",
    "checkpoint": "runs/reward-model-smoke/checkpoints/latest.pt"
  }
}
```

## GRPO

GRPO expects a policy initialized from a checkpoint and a frozen reference model. If `grpo.reference_checkpoint` is omitted, Anila uses `train.init_from` as the reference checkpoint.

```json
{
  "train": {
    "objective": "grpo",
    "init_from": "runs/sft-smoke/checkpoints/latest.pt",
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
    "init_from": "runs/sft-smoke/checkpoints/latest.pt",
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
