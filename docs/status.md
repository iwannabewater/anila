# Project Status

Generated: 2026-05-12

## Current Release Target

`v0.1.0` establishes Anila as a compact full-flow language-model training library:

- Native GPT-style causal LM with RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, and top-k/top-p sampling.
- Training objectives: pretrain, SFT, LoRA SFT, hard/soft distillation, DPO, reward model, GRPO, and PPO.
- Reward path: rule rewards and native learned reward scorer checkpoints.
- Runtime controls: mixed precision, TF32 control, optional fused AdamW, optional activation checkpointing, gradient accumulation, gradient clipping, cosine decay, resume, run metrics, and atomic checkpoints.
- Tooling: strict JSON/TOML config validation, smoke configs, checkpoint inspection CLI, unit tests, CI.

## Non-Goals For This Release

- Distributed or multi-node training.
- External Hugging Face model import/export.
- safetensors artifact export.
- FlashAttention-specific kernels beyond PyTorch scaled dot product attention.
- Dataset streaming or large-corpus preprocessing.

## Next Iteration Candidates

1. KV-cache generation for faster inference while keeping the plain forward path easy to read.
2. LoRA merge/export so adapter checkpoints can be folded into full native model checkpoints.
3. Evaluation harness with perplexity and lightweight preference/reward diagnostics.
4. Dataset packing and streaming for larger local corpora.
5. safetensors/HF interop after native checkpoint semantics stay stable.
