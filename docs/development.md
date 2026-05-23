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

Run the end-to-end quickstart path before larger changes:

```bash
# Train the tiny tokenizer used by all quickstart configs.
uv run anila tokenizer train \
  --input examples/tiny_corpus.txt \
  --input examples/tiny_sft.jsonl \
  --out runs/tokenizer \
  --vocab-size 512 \
  --min-frequency 1

# Build the base pretraining checkpoint.
uv run anila model train --config configs/quickstart/pretrain.json

# Build SFT and LoRA checkpoints from the base path.
uv run anila model train --config configs/quickstart/sft.json
uv run anila model train --config configs/quickstart/lora-sft.json

# Exercise distillation, preference, reward, and online RL objectives.
uv run anila model train --config configs/quickstart/distill-hard-sft.json
uv run anila model train --config configs/quickstart/distill-soft-pretrain.json
uv run anila model train --config configs/quickstart/dpo.json
uv run anila model train --config configs/quickstart/reward-model.json
uv run anila model train --config configs/quickstart/grpo-rule-reward.json
uv run anila model train --config configs/quickstart/ppo-rule-reward.json
uv run anila model train --config configs/quickstart/grpo-learned-reward.json
uv run anila model train --config configs/quickstart/ppo-learned-reward.json

# Fold LoRA adapter weights into a plain native checkpoint.
uv run anila checkpoint merge-lora \
  --checkpoint runs/quickstart/lora-sft/checkpoints/latest.pt \
  --out runs/quickstart/lora-sft/checkpoints/merged.pt

# Generate a quick continuation from the PPO checkpoint.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is"

# Generate a deterministic beam-search completion from the same checkpoint.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --num-beams 4 \
  --length-penalty 0.7 \
  --completion-only

# Inspect checkpoint metadata as JSON.
uv run anila checkpoint inspect \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt

# Measure base language-model loss and perplexity.
uv run anila model evaluate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_corpus.txt \
  --task lm \
  --objective pretrain

# Measure DPO-style policy preference accuracy.
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

Each run writes `config.json` and `metrics.jsonl` under `train.out_dir`. These files are generated run artifacts, so they remain ignored with the rest of `runs/`.

Quickstart configs are intentionally tiny, readable recipes under `configs/quickstart/`. Test-only end-to-end checks live under `tests/` and use integration-test naming.

The pretraining quickstart uses `data.pretrain_mode = "packed"`. The default remains `sliding_window` for backward-compatible configs, and `streaming` is available for larger local text files that should not be materialized as one corpus tensor.

## Runtime Flags

The trainer keeps performance-oriented behavior explicit in `train` config:

- `allow_tf32`: toggles CUDA TF32 matmul/cuDNN usage.
- `gradient_checkpointing`: enables transformer block activation recomputation during backward.
- `fused_adamw`: requests PyTorch fused AdamW on CUDA and falls back to ordinary AdamW outside CUDA.
- `keep_last_checkpoints`: keeps only the most recent N numbered step checkpoints while preserving `latest.pt`.

## Inference Path

`AnilaLM.generate` enables `use_cache=True` by default. The first sampling step pre-fills the cache with the active context window, later steps feed only the newest token, and the cache is rebuilt from the most recent context window when it would exceed `model.context_length`.

The ordinary `forward(input_ids, targets=...)` training path remains cache-free. Cached continuations reject `targets` because cached loss computation would obscure label alignment.

The CLI generation path exposes sampling and deterministic modes through `--sample/--greedy`, optional `--seed`, `--top-k 0` to disable top-k, `--top-p`, `--min-p`, `--repetition-penalty`, `--num-beams`, `--length-penalty`, and `--full-text/--completion-only`.

## Checkpoint Contract

Training checkpoints are ordinary `torch.save` dictionaries:

- `schema_version`: checkpoint schema version.
- `objective`: training objective, currently `pretrain`, `sft`, `distill`, `dpo`, `reward_model`, `grpo`, or `ppo`.
- `model`: model state dict.
- `model_config`: model config as plain data.
- `train_config`: train config as plain data.
- `data_config`: pretraining data mode and sequence-window controls as plain data.
- `lora_config`: LoRA config as plain data.
- `lora_targets`: projection modules that were wrapped with LoRA.
- `adapter_checkpoint`: adapter-only checkpoint path when LoRA adapter saving is enabled.
- `distill_config`: distillation config as plain data.
- `dpo_config`: DPO config as plain data.
- `grpo_config`: GRPO config as plain data.
- `ppo_config`: PPO config as plain data.
- `reward_config`: reward scorer and reward model config as plain data.
- `value_head`: PPO value head state dict, or `None` for non-PPO objectives.
- `reward_head`: reward model head state dict, or `None` for non-reward-model objectives.
- `sft_config`: SFT formatting config as plain data.
- `tokenizer_path`: tokenizer artifact path used by the run.
- `step`: completed optimizer step.
- `optimizer`: optimizer state dict.
- `rng_state`: Python, NumPy, PyTorch, and available CUDA RNG states used to keep validation side-effect-free and continue stochastic streams on resume.
- `merged_lora_targets`: projection modules folded into base weights by `checkpoint merge-lora`, present only on merged exports.
- `merged_from_checkpoint`: source checkpoint path for merged LoRA exports, present only on merged exports.

`latest.pt` is written atomically and is safe to use for resume or sampling after a completed save from a trusted run. Library reads accept tensor/plain-data checkpoint payloads through PyTorch restricted weight loading, and checkpoints containing ordinary serialized Python objects are rejected. Do not treat checkpoint loading as a sandbox for arbitrary untrusted files.

`rng_state` restores random streams on resume. It does not record a partially consumed dataloader iterator, so exact mid-epoch batch-order replay is outside the current single-process checkpoint contract.

When LoRA is enabled, adapter-only checkpoints are also written under `checkpoints/adapters/`.

Merged LoRA exports add `merged_lora_targets` and `merged_from_checkpoint`, reset `lora_config.enabled` to false, clear `adapter_checkpoint`, and store ordinary base model keys under `model`.

## Release Hygiene

Do not commit generated training artifacts, checkpoints, caches, local virtual environments, or experiment logs. They are ignored by `.gitignore` and should remain reproducible from committed configs and source.
