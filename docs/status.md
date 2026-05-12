# Project Status

Generated: 2026-05-12

## Current Release

`v0.1.4` establishes Anila as a compact full-flow language-model training library with cleaner release, artifact, evaluation, and data-input workflows:

- Native GPT-style causal LM with RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, and top-k/top-p sampling.
- Training objectives: pretrain, SFT, LoRA SFT, hard/soft distillation, DPO, reward model, GRPO, and PPO.
- Reward path: rule rewards and native learned reward scorer checkpoints.
- Runtime controls: mixed precision, TF32 control, optional fused AdamW, optional activation checkpointing, gradient accumulation, gradient clipping, cosine decay, resume, run metrics, and atomic checkpoints.
- Tooling: strict JSON/TOML config validation, explicit quickstart configs, checkpoint inspection/evaluation CLI, unit tests, CI.
- Data input: pretraining supports dense sliding windows, packed fixed-length blocks, and streaming local text files through `data.pretrain_mode`.
- Native KV-cache generation is enabled by default in `AnilaLM.generate`.
- Cached prefill plus continuation logits are tested against the plain full forward path.
- Cache rebuilding keeps generation bounded by `model.context_length` without complicating the training forward path.
- LoRA checkpoints can now be exported as merged full-model checkpoints for plain native inference.
- Canonical CLI commands are grouped by resource: `tokenizer`, `model`, and `checkpoint`.
- `anila model evaluate` reports language-model loss/perplexity, policy preference accuracy, and reward-model pairwise accuracy.
- Quickstart config filenames live under `configs/quickstart/` and use explicit objective names such as `pretrain.json`, `sft.json`, and `ppo-rule-reward.json`.
- Integration tests use integration-test naming, keeping test vocabulary separate from user-facing run recipes.
- Train config validation now rejects invalid AdamW betas, worker counts, device strings, and output directories before runtime setup.

## Non-Goals For This Release

- Distributed or multi-node training.
- External Hugging Face model import/export.
- safetensors artifact export.
- FlashAttention-specific kernels beyond PyTorch scaled dot product attention.
- Binary token caches or distributed data loading.

## Next Iteration Candidates

1. Adapter-only load/apply commands if adapter artifacts start being distributed independently.
2. safetensors/HF interop after native checkpoint semantics stay stable.
3. Broader evaluation suites after tiny local diagnostics remain stable.
4. Binary token cache generation if streaming raw text becomes the bottleneck.
5. Distributed training after single-process objective coverage remains stable.
