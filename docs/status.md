# Project Status

Generated: 2026-05-12

## Current Release

`v0.1.1` establishes Anila as a compact full-flow language-model training library with a faster native inference path:

- Native GPT-style causal LM with RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, and top-k/top-p sampling.
- Training objectives: pretrain, SFT, LoRA SFT, hard/soft distillation, DPO, reward model, GRPO, and PPO.
- Reward path: rule rewards and native learned reward scorer checkpoints.
- Runtime controls: mixed precision, TF32 control, optional fused AdamW, optional activation checkpointing, gradient accumulation, gradient clipping, cosine decay, resume, run metrics, and atomic checkpoints.
- Tooling: strict JSON/TOML config validation, smoke configs, checkpoint inspection CLI, unit tests, CI.
- Native KV-cache generation is enabled by default in `AnilaLM.generate`.
- Cached prefill plus continuation logits are tested against the plain full forward path.
- Cache rebuilding keeps generation bounded by `model.context_length` without complicating the training forward path.

## Non-Goals For This Release

- Distributed or multi-node training.
- External Hugging Face model import/export.
- safetensors artifact export.
- FlashAttention-specific kernels beyond PyTorch scaled dot product attention.
- Dataset streaming or large-corpus preprocessing.

## Next Iteration Candidates

1. LoRA merge/export so adapter checkpoints can be folded into full native model checkpoints.
2. Evaluation harness with perplexity and lightweight preference/reward diagnostics.
3. Dataset packing and streaming for larger local corpora.
4. safetensors/HF interop after native checkpoint semantics stay stable.
5. Distributed training after single-process objective coverage remains stable.
