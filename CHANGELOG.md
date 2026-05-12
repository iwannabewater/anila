# Changelog

## v0.1.2 - 2026-05-12

Artifact and CLI maturity release.

- Added LoRA checkpoint merge/export so adapter weights can be folded into a plain native full-model checkpoint.
- Added canonical grouped CLI commands under `tokenizer`, `model`, and `checkpoint`, while retaining the previous flat commands as aliases.
- Renamed smoke configs to hyphen-case for consistency with run output directories.
- Tightened train config validation for AdamW betas, worker counts, device strings, and output directories.
- Reworked README quick start commands with purpose comments and updated release/status documentation.

## v0.1.1 - 2026-05-12

Incremental inference release.

- Added native KV-cache generation to `AnilaLM.generate`, enabled by default.
- Added cached continuation support to `AnilaLM.forward` through `past_key_values` and `use_cache`.
- Added tests that compare cached continuation logits against the plain full forward path.
- Documented the inference cache contract and updated project status notes.

## v0.1.0 - 2026-05-12

Initial full-flow training release.

- Added objective-aware training for pretraining, SFT, LoRA, hard/soft distillation, DPO, learned reward models, GRPO, and PPO.
- Added native reward scorer adapters so GRPO/PPO can use either rule rewards or learned reward checkpoints.
- Added runtime controls for mixed precision, TF32, optional fused AdamW, optional activation checkpointing, gradient accumulation, clipping, cosine decay, resume, and atomic checkpointing.
- Added run artifacts: `config.json`, `metrics.jsonl`, full checkpoints, and LoRA adapter-only checkpoints.
- Added `anila inspect-checkpoint` for checkpoint summaries.
- Added runnable smoke configs, tiny examples, documentation, CI-backed unit tests, and release status notes.
