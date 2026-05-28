# Changelog

## Unreleased

Reliability baseline.

- Centralized checkpoint loading on PyTorch restricted weight deserialization, rejected object-bearing payloads, and raised the minimum PyTorch version to a patched `>=2.10`.
- Saved and restored runtime RNG state, and isolated evaluation from subsequent training randomness.
- Corrected batched generation so rows remain terminal after emitting EOS.
- Rejected invalid UTF-8 training inputs and explicit zero grouped-query KV head configurations.
- Added regression coverage and a contributor contract for checkpoint, data, generation, and verification boundaries.
- Added `anila --version`, smoke-tested CLI help, and documented the package-root API for common training and sampling entry points.
- Added native generation steps, `generate_text`, `stream_text`, repeated CLI `--stop`, `--stream`, and JSON generation output with optional generated-token logprobs.
- Added lightweight benchmark suites and `anila model benchmark` for grouped LM, preference, and reward evaluation runs.
- Added optional safetensors tensor export with `anila checkpoint export-safetensors` and a native manifest.
- Added optional `train.ema_decay` with EMA validation, checkpoint resume, `--ema` sampling/evaluation/benchmark paths, LoRA merge handling, and safetensors export coverage.

## v0.1.6 - 2026-05-16

Native beam-search generation release.

- Added deterministic native beam search to `AnilaLM.generate` through `num_beams` and `length_penalty`.
- Added CLI generation flags for `--num-beams` and `--length-penalty`.
- Added sampling and model tests covering beam-search generation and validation.
- Updated README, architecture, development, status, and iteration notes for the new inference contract.

## v0.1.5 - 2026-05-15

Generation controls and checkpoint retention release.

- Added greedy decoding, seeded sampling, min-p filtering, repetition penalty, and completion-only generation output controls.
- Added CLI generation controls for `--sample/--greedy`, `--seed`, `--min-p`, `--repetition-penalty`, `--full-text/--completion-only`, and `--top-k 0` to disable top-k.
- Added `train.keep_last_checkpoints` and checkpoint manager pruning for full and adapter step checkpoints.
- Added tests for modern generation filters and checkpoint retention.

## v0.1.4 - 2026-05-12

Pretraining data pipeline release.

- Added a top-level `data` config section with `pretrain_mode` values `sliding_window`, `packed`, and `streaming`.
- Added packed fixed-length pretraining blocks and streaming local text datasets without changing the default sliding-window behavior.
- Stored `data_config` in training and adapter checkpoints and surfaced it through checkpoint inspection.
- Switched pretraining quickstart recipes to packed mode for clearer larger-corpus defaults.
- Added tests for packed blocks, sequence stride, streaming dataloaders, data config validation, and checkpoint summaries.
- Updated README, architecture, development, status, and release notes for the new data pipeline.

## v0.1.3 - 2026-05-12

Evaluation and quickstart naming release.

- Added `anila model evaluate` for language-model loss/perplexity, policy preference accuracy, and reward-model pairwise accuracy.
- Moved runnable recipes into `configs/quickstart/` with explicit objective names such as `pretrain.json`, `sft.json`, and `ppo-rule-reward.json`.
- Renamed end-to-end training tests to integration-test terminology.
- Added coverage for the evaluation API/CLI and quickstart config loading.
- Updated README, development notes, architecture docs, status, and release notes around the new naming and evaluation workflow.

## v0.1.2 - 2026-05-12

Artifact and CLI maturity release.

- Added LoRA checkpoint merge/export so adapter weights can be folded into a plain native full-model checkpoint.
- Added canonical grouped CLI commands under `tokenizer`, `model`, and `checkpoint`, while retaining the previous flat commands as aliases.
- Renamed runnable configs to hyphen-case for consistency with run output directories.
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
- Added runnable quickstart configs, tiny examples, documentation, CI-backed unit tests, and release status notes.
