# Iteration Review

This note records the review direction for the sampler and checkpoint-retention iteration.

## Repository posture

Anila already has a strong small-but-complete shape: tokenizer training, data loading, native GPT model, single-process training, checkpointing, sampling, evaluation, LoRA, distillation, DPO, reward modeling, GRPO, and PPO are all visible in a compact PyTorch codebase.

The right next step is therefore not to introduce a large orchestration framework. The better fit is to improve the native interfaces that every user touches: generation controls, reproducible sampling, checkpoint lifecycle hygiene, and documentation.

## Implemented changes

- `AnilaLM.generate` now supports greedy decoding, seeded sampling through a local generator, min-p filtering, and repetition penalty.
- `sample_text` can return either the full decoded prompt plus continuation or only the generated completion.
- The CLI exposes `--sample/--greedy`, `--seed`, `--min-p`, `--repetition-penalty`, `--full-text/--completion-only`, and `--top-k 0` to disable top-k.
- `TrainConfig.keep_last_checkpoints` lets local runs retain only the newest numbered step checkpoints while preserving `latest.pt`.
- Adapter-only LoRA step checkpoints follow the same retention policy.

## Design constraints

- Defaults preserve previous behavior: sampling remains enabled, top-k defaults to 50, top-p defaults to 1.0, min-p defaults to 0.0, repetition penalty defaults to 1.0, and checkpoint retention is disabled unless configured.
- Filtering code keeps at least one token available after min-p filtering.
- Checkpoint retention only prunes numbered `step_*.pt` files; it does not remove `latest.pt`.
- The trainer remains single-process and easy to read.

## Suggested next reviews

1. Add a tiny beam-search implementation in the same native style.
2. Add optional EMA weights for evaluation-only stabilization.
3. Add lightweight benchmark/evaluation adapters without pulling in a heavy harness.
4. Add token-cache generation for larger local corpora once streaming raw text becomes a real bottleneck.
5. Consider optional RoPE scaling and sliding-window attention only after the current cache and generation contracts remain stable.
