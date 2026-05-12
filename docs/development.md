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
  --input examples/tiny_sft.jsonl \
  --out runs/tokenizer \
  --vocab-size 512 \
  --min-frequency 1

uv run anila train --config configs/smoke.json

uv run anila train --config configs/sft_smoke.json

uv run anila train --config configs/lora_sft_smoke.json
uv run anila train --config configs/distill_hard_sft_smoke.json
uv run anila train --config configs/distill_soft_smoke.json
uv run anila train --config configs/dpo_smoke.json
uv run anila train --config configs/reward_model_smoke.json
uv run anila train --config configs/grpo_smoke.json
uv run anila train --config configs/ppo_smoke.json
uv run anila train --config configs/grpo_learned_reward_smoke.json
uv run anila train --config configs/ppo_learned_reward_smoke.json

uv run anila sample \
  --checkpoint runs/ppo-smoke/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is"

uv run anila inspect-checkpoint \
  --checkpoint runs/ppo-smoke/checkpoints/latest.pt
```

Each run writes `config.json` and `metrics.jsonl` under `train.out_dir`. These files are generated run artifacts, so they remain ignored with the rest of `runs/`.

## Runtime Flags

The trainer keeps performance-oriented behavior explicit in `train` config:

- `allow_tf32`: toggles CUDA TF32 matmul/cuDNN usage.
- `gradient_checkpointing`: enables transformer block activation recomputation during backward.
- `fused_adamw`: requests PyTorch fused AdamW on CUDA and falls back to ordinary AdamW outside CUDA.

## Inference Path

`AnilaLM.generate` enables `use_cache=True` by default. The first sampling step pre-fills the cache with the active context window, later steps feed only the newest token, and the cache is rebuilt from the most recent context window when it would exceed `model.context_length`.

The ordinary `forward(input_ids, targets=...)` training path remains cache-free. Cached continuations reject `targets` because cached loss computation would obscure label alignment.

## Checkpoint Contract

Training checkpoints are ordinary `torch.save` dictionaries:

- `schema_version`: checkpoint schema version.
- `objective`: training objective, currently `pretrain`, `sft`, `distill`, `dpo`, `reward_model`, `grpo`, or `ppo`.
- `model`: model state dict.
- `model_config`: model config as plain data.
- `train_config`: train config as plain data.
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

`latest.pt` is written atomically and is safe to use for resume or sampling after a completed save.

When LoRA is enabled, adapter-only checkpoints are also written under `checkpoints/adapters/`.

## Release Hygiene

Do not commit generated training artifacts, checkpoints, caches, local virtual environments, or experiment logs. They are ignored by `.gitignore` and should remain reproducible from committed configs and source.
