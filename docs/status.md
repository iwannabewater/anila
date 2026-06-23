# Project Status

Generated: 2026-06-23

## Current Release

`v0.1.8` establishes Anila as a compact full-flow language-model training library with stronger native generation controls, deterministic beam search, local checkpoint lifecycle hygiene, native on-policy distillation, and a beginner-readable path through the full training and inference workflow:

- Native GPT-style causal LM with RoPE, RMSNorm, SwiGLU, GQA, tied embeddings, KV-cache generation, greedy decoding, seeded sampling, top-k/top-p/min-p filtering, repetition penalty, and native beam search.
- Training objectives: pretrain, SFT, LoRA SFT, hard/soft distillation, on-policy distillation, DPO, reward model, GRPO, and PPO.
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
- `docs/full-flow-quickstart.md` walks from tokenizer training through data, pretraining, SFT, LoRA, distillation, OPD, DPO, reward modeling, GRPO/PPO, evaluation, optional export, and efficient inference.
- `docs/data-contracts.md` documents the accepted plain-text and JSONL data shapes, label masks, configurable keys, and common failure modes for each objective.
- `scripts/verify.sh` is the shared local and CI verification gate for lockfile, lint, and test checks.
- `scripts/quickstart-smoke.sh` runs the full tiny tokenizer-to-inference workflow as a release-minded local smoke test.
- `docs/iteration-review.md` captures the stable iteration-review protocol for release-minded changes.
- Quickstart config filenames live under `configs/quickstart/` and use explicit objective names such as `pretrain.json`, `sft.json`, `opd.json`, and `ppo-rule-reward.json`.
- Integration tests use integration-test naming, keeping test vocabulary separate from user-facing run recipes.
- Train config validation now rejects invalid AdamW betas, worker counts, device strings, output directories, and checkpoint retention counts before runtime setup.

## Main Branch Since v0.1.8

No unreleased changes recorded yet.

## Non-Goals For This Release

- Distributed or multi-node training.
- External Hugging Face model import/export.
- safetensors checkpoint import or resume.
- FlashAttention-specific kernels beyond PyTorch scaled dot product attention.
- Binary token caches or distributed data loading.

## Next Iteration Candidates

1. Prototype Hugging Face import/export only as an optional adapter with tests and clear unsupported paths.
2. Prototype distributed runtime support behind a separate adapter after single-process coverage remains stable.
3. Add token-cache generation for larger local corpora once streaming raw text becomes a real bottleneck.
4. Consider batched beam-search improvements only after single-prompt beam use cases need more throughput.
5. Add richer benchmark task types only after the lightweight suite format has real project datasets.
