# Full-Flow Quickstart

This guide is the beginner path through Anila's local LLM workflow. It starts with a tokenizer and tiny local data, then walks through pretraining, supervised fine-tuning, adapter fine-tuning, distillation, preference optimization, reward modeling, online RL, evaluation, artifact export, and efficient native inference.

The configs are intentionally small. They are for learning the flow and checking contracts, not for producing a capable model.

Before replacing the example files with your own data, read [Data Contracts](data-contracts.md). It lists the JSONL fields, label masking rules, and failure conditions for every objective.

## Stage Map

| Stage | Command surface | Input | Output |
| --- | --- | --- | --- |
| Tokenizer | `anila tokenizer train` | `examples/tiny_corpus.txt`, `examples/tiny_sft.jsonl` | `runs/tokenizer/` |
| Pretraining | `anila model train --config configs/quickstart/pretrain.json` | plain UTF-8 text | base checkpoint |
| SFT | `configs/quickstart/sft.json` | prompt/response JSONL | instruction checkpoint |
| LoRA SFT | `configs/quickstart/lora-sft.json` | SFT JSONL plus base checkpoint | full checkpoint plus adapter checkpoint |
| Distillation | `distill-hard-sft.json`, `distill-soft-pretrain.json` | hard labels or teacher logits | student checkpoint |
| DPO | `configs/quickstart/dpo.json` | prompt/chosen/rejected JSONL | preference-tuned checkpoint |
| Reward model | `configs/quickstart/reward-model.json` | prompt/chosen/rejected JSONL | scalar reward checkpoint |
| GRPO/PPO | rule or learned reward configs | prompt JSONL plus policy checkpoint | RL-tuned checkpoints |
| Evaluation | `anila model evaluate`, `anila model benchmark` | checkpoints plus held-out local data | JSON metrics |
| Inference | `anila model generate` | native checkpoint plus tokenizer | text or JSON generation metadata |

## 1. Install And Check The CLI

```bash
uv sync --group dev
uv run anila --version
uv run anila --help
```

The quickstart uses `runs/` for generated artifacts. That directory is ignored by Git.

## 2. Train The Tokenizer

```bash
uv run anila tokenizer train \
  --input examples/tiny_corpus.txt \
  --input examples/tiny_sft.jsonl \
  --out runs/tokenizer \
  --vocab-size 512 \
  --min-frequency 1
```

The tokenizer stage is deliberately first because all later configs point at `runs/tokenizer`.

## 3. Pretrain The Base Model

```bash
uv run anila model train --config configs/quickstart/pretrain.json
```

This trains next-token prediction on plain text. The quickstart pretraining config uses packed fixed-length blocks and writes:

- `runs/quickstart/pretrain/config.json`
- `runs/quickstart/pretrain/metrics.jsonl`
- `runs/quickstart/pretrain/checkpoints/latest.pt`
- numbered step checkpoints under `runs/quickstart/pretrain/checkpoints/`

The pretraining checkpoint also includes EMA weights because the config sets `train.ema_decay`.

## 4. Run Supervised Fine-Tuning

```bash
uv run anila model train --config configs/quickstart/sft.json
```

SFT reads `examples/tiny_sft.jsonl`. Prompt, system, and user tokens are masked out of the loss; assistant response tokens train the model.

## 5. Train A LoRA Adapter

```bash
uv run anila model train --config configs/quickstart/lora-sft.json
```

This starts from the pretraining checkpoint and trains only LoRA adapter weights for the configured projection modules. It saves both a full checkpoint and adapter-only artifacts.

To fold the adapter into ordinary model weights:

```bash
uv run anila checkpoint merge-lora \
  --checkpoint runs/quickstart/lora-sft/checkpoints/latest.pt \
  --out runs/quickstart/lora-sft/checkpoints/merged.pt
```

## 6. Exercise Distillation

Hard-label distillation trains from teacher-generated records through the SFT data path:

```bash
uv run anila model train --config configs/quickstart/distill-hard-sft.json
```

Soft-logit distillation loads the native pretraining checkpoint as a teacher:

```bash
uv run anila model train --config configs/quickstart/distill-soft-pretrain.json
```

## 7. Run Preference Optimization

```bash
uv run anila model train --config configs/quickstart/dpo.json
```

DPO uses `examples/tiny_preferences.jsonl`, compares chosen and rejected responses, and regularizes against a frozen reference model. If the DPO config omits `dpo.reference_checkpoint`, Anila uses `train.init_from` as the reference checkpoint.

## 8. Train A Reward Model

```bash
uv run anila model train --config configs/quickstart/reward-model.json
```

Reward-model training uses the same chosen/rejected preference records, but trains a scalar reward head. The resulting checkpoint can score generated responses for online RL.

## 9. Run Online RL

Start with a simple rule reward:

```bash
uv run anila model train --config configs/quickstart/grpo-rule-reward.json
uv run anila model train --config configs/quickstart/ppo-rule-reward.json
```

Then run the learned reward model path:

```bash
uv run anila model train --config configs/quickstart/grpo-learned-reward.json
uv run anila model train --config configs/quickstart/ppo-learned-reward.json
```

GRPO samples groups of responses per prompt and normalizes rewards inside the group. PPO samples rollouts, applies reference KL penalties, estimates advantages with a value head, and keeps the base LM checkpoint directly sampleable.

## 10. Evaluate And Benchmark

Language-model loss and perplexity:

```bash
uv run anila model evaluate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_corpus.txt \
  --task lm \
  --objective pretrain
```

Preference accuracy:

```bash
uv run anila model evaluate \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_preferences.jsonl \
  --task preference
```

Reward-model pairwise accuracy:

```bash
uv run anila model evaluate \
  --checkpoint runs/quickstart/reward-model/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --dataset examples/tiny_preferences.jsonl \
  --task reward
```

Multi-task benchmark suite:

```bash
uv run anila model benchmark \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --suite configs/benchmarks/quickstart.json \
  --max-batches 1
```

Use `--ema` on evaluation or benchmark commands when the checkpoint contains EMA weights and you want to evaluate that snapshot.

## 11. Export Optional Tensor Artifacts

```bash
uv run anila checkpoint export-safetensors \
  --checkpoint runs/quickstart/sft/checkpoints/latest.pt \
  --out-dir runs/quickstart/sft/safetensors
```

This writes tensor-only safetensors plus an Anila manifest. It is an export adapter. Native training, resume, sampling, and evaluation continue to use restricted `.pt` checkpoint payloads.

## 12. Generate Efficiently

Default generation uses the native KV cache:

```bash
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is"
```

Structured metadata with generated-token logprobs:

```bash
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --max-new-tokens 16 \
  --json \
  --logprobs
```

Deterministic beam-search completion:

```bash
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --num-beams 4 \
  --length-penalty 0.7 \
  --completion-only
```

Streaming single-path generation:

```bash
uv run anila model generate \
  --checkpoint runs/quickstart/ppo-rule-reward/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --stream
```

EMA inference from the pretraining checkpoint:

```bash
uv run anila model generate \
  --checkpoint runs/quickstart/pretrain/checkpoints/latest.pt \
  --tokenizer runs/tokenizer \
  --prompt "Anila is" \
  --ema
```

## Where To Edit First

- Change model size in the `model` section.
- Change optimization and runtime settings in the `train` section.
- Change pretraining data behavior in the `data` section.
- Change SFT record handling in the `sft` section.
- Change preference and RL behavior in `dpo`, `reward`, `grpo`, and `ppo`.
- Add optional ecosystem formats as adapters instead of changing native checkpoint loading.

## Verification Before Sharing Changes

```bash
bash scripts/verify.sh
```

For checkpoint, data loading, generation, or training runtime changes, also run the focused tests that cover the affected contract.

To smoke-run this whole staged path from the repository root:

```bash
bash scripts/quickstart-smoke.sh
```

The smoke script writes ignored training artifacts under `runs/` and compact JSON/text outputs under the OS temp directory.
