# Development

## Environment

Anila uses uv for Python and dependency management:

```bash
uv sync --group dev
```

Run commands through uv so the local package and locked dependencies are used:

```bash
uv run anila --help
uv run anila --version
```

For the ordered beginner path through tokenizer training, every quickstart objective, evaluation, export, and inference, use [Full-Flow Quickstart](full-flow-quickstart.md). For the input file schemas behind those commands, use [Data Contracts](data-contracts.md).

## Quality Gates

```bash
bash scripts/verify.sh
```

The verification wrapper runs `uv lock --check`, `uv run ruff check .`, and `uv run pytest -q`, matching CI.

Before release-minded changes, use [Iteration Review Protocol](iteration-review.md) to check scope, evidence, architecture locality, and stop conditions.

Run the end-to-end quickstart path before larger changes:

```bash
bash scripts/quickstart-smoke.sh
```

The smoke script writes ignored training artifacts under `runs/` and compact JSON/text outputs under the OS temp directory. The expanded command sequence is:

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

# Optionally exercise the native routed-MoE model path.
uv run anila model train --config configs/quickstart/pretrain-moe.json

# Build SFT and LoRA checkpoints from the base path.
uv run anila model train --config configs/quickstart/sft.json
uv run anila model train --config configs/quickstart/lora-sft.json

# Exercise distillation, OPD, preference, reward, and online RL objectives.
uv run anila model train --config configs/quickstart/distill-hard-sft.json
uv run anila model train --config configs/quickstart/distill-soft-pretrain.json
uv run anila model train --config configs/quickstart/opd.json
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

# Export tensors and a native manifest for external artifact handling.
uv run anila checkpoint export-safetensors \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --out-dir runs/quickstart/sft/safetensors

# Generate a quick continuation from the PPO checkpoint.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is"

# Generate from EMA weights when the checkpoint includes train.ema_decay.
uv run anila model generate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --ema

# Generate a deterministic beam-search completion from the same checkpoint.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --num-beams 4 \
  --length-penalty 0.7 \
  --completion-only

# Inspect structured generation metadata and token logprobs.
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 16 \
  --json \
  --logprobs

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

# Run a small multi-task benchmark suite.
uv run anila model benchmark \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --suite configs/benchmarks/quickstart.json \
  --max-batches 1
```

Each run writes `config.json` and `metrics.jsonl` under `train.out_dir`. These files are generated run artifacts, so they remain ignored with the rest of `runs/`.

Quickstart configs are intentionally tiny, readable recipes under `configs/quickstart/`. Test-only end-to-end checks live under `tests/` and use integration-test naming.

The pretraining quickstart uses `data.pretrain_mode = "packed"`. The default remains `sliding_window` for backward-compatible configs, and `streaming` is available for larger local text files that should not be materialized as one corpus tensor.

The default model path uses ordinary RoPE. To exercise native YaRN-style RoPE scaling, set `model.rope_scaling = "yarn"`, `model.rope_scaling_factor`, and `model.rope_original_context_length`; validation rejects scaling settings that would be ignored or leave the configured context unextended.

The dense model path remains the default. `configs/quickstart/pretrain-moe.json` sets `model.moe_num_experts`, `model.moe_top_k`, `model.moe_intermediate_size`, and `model.moe_aux_loss_coef` to exercise the native routed-SwiGLU expert path without adding distributed runtime or fused-kernel dependencies.

## Python API

The package root intentionally exposes only the common convenience surface: `__version__`, config dataclasses and `load_run_config`, `AnilaLM`, `RewardModel`, `train`, `train_byte_bpe`, `sample_text`, `generate_text`, `generate_chat`, `generate_tool_chat`, `stream_text`, chat prompt/render helpers, checkpoint inspection/merge/export helpers, evaluation functions, and the lightweight benchmark suite runner. Use module-level imports such as `anila.data`, `anila.peft`, or `anila.training` when changing internals or adding optional adapters.

## Runtime Flags

The trainer keeps performance-oriented behavior explicit in `train` config:

- `allow_tf32`: toggles CUDA TF32 matmul/cuDNN usage.
- `gradient_checkpointing`: enables transformer block activation recomputation during backward.
- `fused_adamw`: requests PyTorch fused AdamW on CUDA and falls back to ordinary AdamW outside CUDA.
- `keep_last_checkpoints`: keeps only the most recent N numbered step checkpoints while preserving `latest.pt`.
- `ema_decay`: keeps exponential moving average weights for validation, checkpointing, and optional `--ema` inference/evaluation.

## Inference Path

`AnilaLM.generate` enables `use_cache=True` by default. The first sampling step pre-fills the cache with the active context window, later steps feed only the newest token, and the cache is rebuilt from the most recent context window when it would exceed `model.context_length`.

The ordinary `forward(input_ids, targets=...)` training path remains cache-free. Cached continuations reject `targets` because cached loss computation would obscure label alignment. For inference-style scoring, `forward(..., logits_to_keep=N)` materializes only the last N logit positions and keeps full hidden states available when `return_hidden_states=True`.

The CLI generation path exposes sampling and deterministic modes through `--sample/--greedy`, optional `--seed`, `--top-k 0` to disable top-k, `--top-p`, `--min-p`, `--repetition-penalty`, `--num-beams`, `--length-penalty`, and `--full-text/--completion-only`. It also supports repeated `--stop` strings, `--json` structured output, `--logprobs` for generated-token logprobs in JSON output, `--stream` for single-path streaming generation, and `--ema` for checkpoints saved with EMA weights.

`anila model chat` renders `System:` / `User:` / `Assistant:` prompts, optional JSON tool specs, and parsed `<think>` / `<tool_call>` metadata over the native generation path. When a checkpoint includes `sft_config`, chat generation uses those saved prefixes and tags for rendering and parsing. The Python `generate_tool_chat` helper can run caller-supplied callbacks and append JSON tool results for local tool-use loops bounded by rounds and per-turn call count. The CLI is a local inference helper, not an OpenAI API server or arbitrary tool executor.

## Checkpoint Contract

Training checkpoints are ordinary `torch.save` dictionaries:

- `schema_version`: checkpoint schema version.
- `objective`: training objective, currently `pretrain`, `sft`, `distill`, `opd`, `dpo`, `reward_model`, `grpo`, or `ppo`.
- `model`: model state dict.
- `model_config`: model config as plain data, including optional RoPE scaling and MoE settings when enabled.
- `train_config`: train config as plain data.
- `data_config`: pretraining data mode and sequence-window controls as plain data.
- `lora_config`: LoRA config as plain data.
- `lora_targets`: projection modules that were wrapped with LoRA.
- `adapter_checkpoint`: adapter-only checkpoint path when LoRA adapter saving is enabled.
- `distill_config`: distillation config as plain data.
- `dpo_config`: DPO config as plain data.
- `opd_config`: on-policy distillation config as plain data.
- `grpo_config`: GRPO config as plain data, including the optional CISPO loss type and rule reward type.
- `ppo_config`: PPO config as plain data, including the rule reward type.
- `reward_config`: reward scorer and reward model config as plain data.
- `value_head`: PPO value head state dict, or `None` for non-PPO objectives.
- `reward_head`: reward model head state dict, or `None` for non-reward-model objectives.
- `ema_model`: optional EMA model/backbone state dict when `train.ema_decay` is configured.
- `ema_value_head`: optional EMA PPO value head state dict.
- `ema_reward_head`: optional EMA reward model head state dict.
- `ema_decay`: EMA decay used by the run when EMA weights are present.
- `sft_config`: SFT formatting config as plain data.
- `tokenizer_path`: tokenizer artifact path used by the run.
- `step`: completed optimizer step.
- `optimizer`: optimizer state dict.
- `rng_state`: Python, NumPy, PyTorch, and available CUDA RNG states used to keep validation side-effect-free and continue stochastic streams on resume.
- `data_state`: deterministic training-loader generator epoch, batch cursor, and data-shaping contract used to continue built-in loader order on resume.
- `merged_lora_targets`: projection modules folded into base weights by `checkpoint merge-lora`, present only on merged exports.
- `merged_from_checkpoint`: source checkpoint path for merged LoRA exports, present only on merged exports.

`latest.pt` is written atomically and is safe to use for resume or sampling after a completed save from a trusted run. Library reads accept tensor/plain-data checkpoint payloads through PyTorch restricted weight loading, and checkpoints containing ordinary serialized Python objects are rejected. Do not treat checkpoint loading as a sandbox for arbitrary untrusted files.

`rng_state` and `data_state` restore random streams and the next built-in training batch on resume, including saves taken within an epoch. Exact replay requires unchanged dataset files and data-shaping configuration; resume rejects a changed recorded data contract. Older checkpoints without `data_state` remain loadable but begin a new deterministic training-loader sequence. Older checkpoints without EMA state also remain loadable when `train.ema_decay` is configured; the trainer initializes EMA from the restored model weights.

When LoRA is enabled, adapter-only checkpoints are also written under `checkpoints/adapters/`.

Merged LoRA exports add `merged_lora_targets` and `merged_from_checkpoint`, reset `lora_config.enabled` to false, clear `adapter_checkpoint`, and store ordinary base model keys under `model` and `ema_model` when EMA weights are present.

Safetensors exports are optional artifact adapters. They write namespaced tensors such as `model.embed.weight` and `ema_model.embed.weight` plus `anila_safetensors.json` metadata, but native checkpoint resume, sampling, and evaluation continue to use restricted `.pt` checkpoint payloads.

## Release Hygiene

Do not commit generated training artifacts, checkpoints, caches, local virtual environments, or experiment logs. They are ignored by `.gitignore` and should remain reproducible from committed configs and source.
