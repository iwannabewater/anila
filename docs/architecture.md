# Architecture

Anila is organized as a small set of explicit modules with narrow contracts. The goal is to keep model and training logic easy to inspect while still separating concerns that change at different rates.

```text
corpus files / SFT JSONL / preference JSONL / reward-prompt JSONL
    |
    v
Tokenizer -----> TextTokenDataset
    |             SupervisedFineTuneDataset
    |             PreferenceDataset
    |             PromptRewardDataset
    |                  |
    v                  v
RunConfig --------> objective-aware Trainer --------> CheckpointManager
    |                        |
    v                        v
 LoRAConfig           adapter checkpoints
 DistillConfig
 DPOConfig
 GRPOConfig
 PPOConfig
 RewardConfig
                             |
                             v
                     AnilaLM / RewardModel
                             |
                             v
                    sampler / evaluator
```

## Module Responsibilities

- `config`: parses JSON/TOML configs, applies defaults, and rejects unknown keys.
- `tokenization`: trains and loads byte-level BPE tokenizers.
- `data`: strictly decodes UTF-8 input and builds sliding-window, packed, or streaming next-token prediction examples from text corpora plus response-masked SFT examples from JSONL records.
- `distillation`: loads native teacher checkpoints and computes masked soft-logit and on-policy distillation losses.
- `dpo`: computes response sequence logprobs and Direct Preference Optimization loss.
- `grpo`: computes group-relative advantages and clipped GRPO loss with reference KL.
- `ppo`: adds a value head wrapper, token-level rollout accounting, GAE, and clipped PPO policy/value loss.
- `reward`: trains scalar reward models, loads learned reward scorer checkpoints, and keeps rule rewards behind the same scorer contract.
- `model`: implements the causal language model, optional activation checkpointing, KV-cache generation, generation-time filtering, and native beam search.
- `peft`: injects LoRA adapters into target linear modules, freezes non-adapter parameters, extracts adapter state, and merges adapters back into plain linear weights.
- `training`: owns objective selection, device/dtype selection, TF32 runtime setup, optimizer setup, learning-rate schedule, EMA tracking, evaluation, run recording, checkpointing, and resume.
- `evaluation`: restores native checkpoints and reports held-out language-model, preference, and reward-model metrics.
- `benchmark`: runs strict JSON/TOML benchmark suites over the native evaluation functions without pulling in a heavy harness.
- `checkpoint`: enforces restricted checkpoint deserialization and exposes lightweight checkpoint inspection, LoRA merge, and optional tensor artifact export for CLI and tests.
- `sampling`: restores checkpoints and exposes text generation.

## Runtime Contract

- Configs and UTF-8 input files fail before training starts when values, encodings, shapes, intervals, or unsupported dtypes are invalid.
- Checkpoints are ordinary tensor/plain-data `torch.save` dictionaries loaded through the restricted `weights_only=True` path on supported PyTorch versions, with schema version, objective, model state, model config, train config, optional objective configs, tokenizer path, optimizer state, completed step, optional EMA state, and optional RNG state.
- `latest.pt` and step checkpoints are written through a temporary file and atomically replaced; `train.keep_last_checkpoints` can prune older numbered step checkpoints while preserving `latest.pt`.
- `config.json` captures the validated run config and `metrics.jsonl` records train, eval, and checkpoint events under `train.out_dir`.
- Tokenizer vocabulary size is loaded from the tokenizer artifact and becomes the model vocabulary at runtime.
- Plain-text pretraining can use dense `sliding_window`, non-overlapping `packed`, or local-file `streaming` data modes.
- `allow_tf32`, `gradient_checkpointing`, `fused_adamw`, `keep_last_checkpoints`, and `ema_decay` are explicit train config flags so stability/performance/storage behavior is visible in saved configs.
- `AnilaLM.generate` uses a native KV cache by default for single-path generation, supports sampling, greedy decoding, and deterministic beam search, preserves terminal EOS rows in batches, and falls back to full-context recomputation whenever the context window must be rebased.
- Evaluation snapshots and restores RNG state so validation, including rollout-based objectives, does not alter subsequent training sampling. When EMA is enabled, evaluation temporarily swaps to EMA weights and restores the training weights afterward.
- `pretrain` consumes plain text and trains all next-token targets.
- `sft` consumes JSONL records and sets non-assistant labels to `-100`, so prompt/user/system tokens do not contribute to loss.
- `distill` can run hard-label distillation over pretraining or SFT data, or soft-logit distillation against a teacher checkpoint.
- `opd` consumes prompt JSONL records, samples responses from the current student policy, masks prompt tokens, and distills generated response tokens against a native teacher checkpoint.
- `dpo` consumes prompt/chosen/rejected JSONL records and compares policy log-ratios against a frozen reference model.
- `reward_model` consumes prompt/chosen/rejected JSONL records, scores chosen and rejected response spans with a scalar reward head, and trains with pairwise Bradley-Terry loss.
- `grpo` consumes prompt JSONL records, samples multiple responses per prompt, computes rule or learned-model rewards, normalizes advantages per prompt group, and regularizes against a frozen reference model.
- `ppo` consumes prompt JSONL records, samples online responses, assigns terminal scorer rewards plus per-token reference KL penalties, trains a value head with GAE returns, and keeps base LM checkpoints sampleable.
- LoRA can wrap selected projection modules before training. Full checkpoints include base and adapter weights; adapter-only checkpoints are saved separately when enabled.
- LoRA checkpoints can be exported as merged full-model checkpoints. The export keeps the native checkpoint shape, clears active LoRA metadata, records `merged_lora_targets`, and leaves the resulting `model` and optional `ema_model` state dicts loadable by plain `AnilaLM`.
- Safetensors export writes tensor-only namespaced weights and a native JSON manifest. It is an optional artifact adapter; native checkpoint loading, resume, sampling, and evaluation continue to read `.pt` payloads through `load_checkpoint_payload`.
- Evaluation restores the same native checkpoint payloads used by sampling/training, supports active LoRA checkpoints, can opt into EMA weights, and reports JSON metrics for `lm`, `preference`, and `reward` tasks.

## Extension Points

- Move dataset adapters into a `datasets/` package if more formats arrive.
- Add new model variants by keeping the `forward(input_ids, targets=None)` interface.
- Add adapter-only load/apply commands if adapter artifacts start being distributed independently from full checkpoints.
- Add richer evaluation suites once the current lightweight harness is stable on larger corpora and preference sets.
- Add sharded binary token caches if local text streaming becomes the bottleneck.
- Add DPO variants only after the base DPO path has real-data coverage.
- Add external reward backends only after native learned reward checkpoints have broader real-data coverage.
- Add hidden-state distillation only after model variant contracts are stable.
- Add distributed training behind a separate runtime module after single-process coverage is stronger.
