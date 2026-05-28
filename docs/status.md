# Project Status

Generated: 2026-05-28

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

## Main Branch Since v0.1.6

- Centralized library checkpoint reads on restricted `weights_only=True` deserialization and rejected object-bearing checkpoint payloads.
- Preserved training RNG and built-in dataloader order across resume while keeping older checkpoints without data state loadable.
- Corrected batched generation so rows that emit EOS remain terminal.
- Kept strict UTF-8 input handling explicit for tokenizer and dataset reads.
- Added `anila --version` and documented the small top-level Python API for common training, tokenizer, evaluation, checkpoint, and sampling entry points.
- Added native single-path generation steps, streaming text output, text-level stop strings, structured generation metadata, and optional generated-token logprobs.

## Non-Goals For This Release

- Distributed or multi-node training.
- External Hugging Face model import/export.
- safetensors artifact export.
- FlashAttention-specific kernels beyond PyTorch scaled dot product attention.
- Binary token caches or distributed data loading.

## Next Iteration Candidates

1. Add lightweight benchmark/evaluation adapters without pulling in a heavy harness.
2. Prototype Hugging Face or safetensors import/export only as optional adapters with tests and clear unsupported paths.
3. Prototype distributed runtime support behind a separate adapter after single-process coverage remains stable.
4. Add token-cache generation for larger local corpora once streaming raw text becomes a real bottleneck.
5. Consider batched beam-search improvements only after single-prompt beam use cases need more throughput.
