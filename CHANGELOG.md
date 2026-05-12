# Changelog

## v0.1.0 - 2026-05-12

Initial full-flow training release.

- Added objective-aware training for pretraining, SFT, LoRA, hard/soft distillation, DPO, learned reward models, GRPO, and PPO.
- Added native reward scorer adapters so GRPO/PPO can use either rule rewards or learned reward checkpoints.
- Added runtime controls for mixed precision, TF32, optional fused AdamW, optional activation checkpointing, gradient accumulation, clipping, cosine decay, resume, and atomic checkpointing.
- Added run artifacts: `config.json`, `metrics.jsonl`, full checkpoints, and LoRA adapter-only checkpoints.
- Added `anila inspect-checkpoint` for checkpoint summaries.
- Added runnable smoke configs, tiny examples, documentation, CI-backed unit tests, and release status notes.
