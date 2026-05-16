# Iteration Review

This note records the review direction for the native beam-search iteration.

## Repository posture

Anila already covers the main compact LLM training path: tokenizer training, data loading, a native GPT model, single-process training, checkpointing, sampling, evaluation, LoRA, distillation, DPO, reward modeling, GRPO, and PPO.

The right next step was not another large training framework. The useful gap was in inference maturity: keep the existing sampling and greedy paths intact, then add a small deterministic beam-search path that makes checkpoint inspection and comparison runs more practical.

## Implemented changes

- `AnilaLM.generate` now accepts `num_beams` and `length_penalty`.
- `num_beams = 1` preserves the existing cached sampling and greedy generation path.
- `num_beams > 1` uses a deterministic native beam-search path for a single prompt.
- Beam search reuses existing top-k, top-p, min-p, repetition-penalty, temperature, EOS, and context-window behavior.
- `sample_text` and `anila model generate` expose beam search through `--num-beams` and `--length-penalty`.
- README, architecture notes, development docs, project status, and changelog were updated for the new generation contract.

## Design constraints

- Existing generation defaults are unchanged.
- Beam search is intentionally single-prompt for now; batch support would add bookkeeping complexity that the CLI does not need yet.
- The cached single-path generator remains the fast default. Beam search recomputes each candidate window because a compact and obvious implementation is more valuable than premature cache sharing at this scale.
- Length penalty is non-negative, where `0.0` keeps raw accumulated log-probability ranking.

## Suggested next reviews

1. Add optional EMA weights for evaluation-only stabilization.
2. Add lightweight benchmark/evaluation adapters without pulling in a heavy harness.
3. Add token-cache generation for larger local corpora once streaming raw text becomes a real bottleneck.
4. Consider optional RoPE scaling and sliding-window attention only after the current cache and generation contracts remain stable.
5. Revisit batched beam search only when there is a concrete evaluation or serving path that needs it.
