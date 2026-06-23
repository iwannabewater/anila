#!/usr/bin/env bash
set -euo pipefail

scratch_root="${TMPDIR:-/tmp}/anila-quickstart-smoke"
mkdir -p "$scratch_root"

uv run anila tokenizer train \
  --input examples/tiny_corpus.txt \
  --input examples/tiny_sft.jsonl \
  --out runs/tokenizer \
  --vocab-size 512 \
  --min-frequency 1

uv run anila model train --config configs/quickstart/pretrain.json
uv run anila model train --config configs/quickstart/sft.json
uv run anila model train --config configs/quickstart/lora-sft.json

uv run anila checkpoint merge-lora \
  --checkpoint runs/quickstart/lora-sft/checkpoints/latest.pt \
  --out runs/quickstart/lora-sft/checkpoints/merged.pt

uv run anila model train --config configs/quickstart/distill-hard-sft.json
uv run anila model train --config configs/quickstart/distill-soft-pretrain.json
uv run anila model train --config configs/quickstart/opd.json
uv run anila model train --config configs/quickstart/dpo.json
uv run anila model train --config configs/quickstart/reward-model.json
uv run anila model train --config configs/quickstart/grpo-rule-reward.json
uv run anila model train --config configs/quickstart/ppo-rule-reward.json
uv run anila model train --config configs/quickstart/grpo-learned-reward.json
uv run anila model train --config configs/quickstart/ppo-learned-reward.json

uv run anila model evaluate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_corpus.txt \
  --task lm \
  --objective pretrain >"$scratch_root/lm-eval.json"

uv run anila model evaluate \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_preferences.jsonl \
  --task preference >"$scratch_root/preference-eval.json"

uv run anila model evaluate \
  --checkpoint runs/quickstart/reward-model/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_preferences.jsonl \
  --task reward >"$scratch_root/reward-eval.json"

uv run anila model benchmark \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --suite configs/benchmarks/quickstart.json \
  --max-batches 1 >"$scratch_root/benchmark.json"

uv run anila checkpoint export-safetensors \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --out-dir runs/quickstart/sft/safetensors >"$scratch_root/safetensors-export.json"

uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 4 >"$scratch_root/generate.txt"

uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 4 \
  --json \
  --logprobs >"$scratch_root/generate-logprobs.json"

uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 4 \
  --num-beams 4 \
  --length-penalty 0.7 \
  --completion-only >"$scratch_root/beam.txt"

uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 4 \
  --stream >"$scratch_root/stream.txt"

uv run anila model generate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 4 \
  --ema >"$scratch_root/ema.txt"

printf 'quickstart smoke: ok (%s)\n' "$scratch_root"
