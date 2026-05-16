# Project Status

Generated: 2026-05-16

## Current Release

`v0.1.6` establishes Anila as a compact full-flow language-model training library with stronger native generation controls, deterministic beam search, and local checkpoint lifecycle hygiene:

- Native GPT-style causal LM with RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, KV-cache generation, greedy decoding, seeded sampling, top-k/top-p/min-p filtering, repetition penalty, and native beam search.
- Training objectives: pretrain, SFT, LoRA SFT, hard/soft distillation, DPO, reward model, GRPO, and PPO.
- Reward path: rule rewards and native learned reward scorer checkpoints.
- Runtime controls: mixed precision, TF32 control, optional fused AdamW, optional activation checkpointing, gradient accumulation, gradient clipping, cosine decay, resume, run metrics, atomic checkpoints, and optional retention for numbered step checkpoints.
- Tooling: strict JSON/TOML config validation, explicit quickstart configs, checkpoint inspection/evaluation CLI, unit tests, CI.
- Data input: pretraining supports dense sliding windows, packed fixed-length blocks, and streaming local text files through `data.pretrain_mode`.
- CLI generation supports `--sample/--greedy`, `--seed`, `--top-k 0`, `--top-p`, `--min-p`, `--repetition-penalty`, `--num-beams`, `--length-penalty`, and `--full-text/--completion-only`.
- Cached prefill plus continuation logits are tested against the plain full forward path.
- Cache rebuilding keeps generation bounded by `model.context_length` without complicating the training forward path.
- LoRA checkpoints can now be exported as merged full-model checkpoints for plain native inference.
- Canonical CLI commands are grouped by resource: `tokenizer`, `model`, and `checkpoint`.
- `anila model evaluate` reports language-model loss/perplexity, policy preference accuracy, and reward-model pairwise accuracy.
- Quickstart config filenames live under `configs/quickstart/` and use explicit objective names such as `pretrain.json`, `sft.json`, and `ppo-rule-reward.json`.
- Integration tests use integration-test naming, keeping test vocabulary separate from user-facing run recipes.
- Train config validation now rejects invalid AdamW betas, worker counts, device strings, output directories, and checkpoint retention counts before runtime setup.

## Non-Goals For This Release

- Distributed or multi-node training.
- External Hugging Face model import/export.
- safetensors artifact export.
- FlashAttention-specific kernels beyond PyTorch scaled dot product attention.
- Binary token caches or distributed data loading.

## Next Iteration Candidates

1. Add optional EMA weights for evaluation-only stabilization.
2. Add lightweight benchmark/evaluation adapters without pulling in a heavy harness.
3. Add token-cache generation for larger local corpora once streaming raw text becomes a real bottleneck.
4. Consider optional RoPE scaling and sliding-window attention after the current cache and generation contracts remain stable.
5. Add batched beam search only if command-line or evaluation use cases need it; the current beam path is intentionally single-prompt and simple.
