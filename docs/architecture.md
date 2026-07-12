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
- `tokenization`: trains and loads byte-level BPE tokenizers, with optional chat special tokens for reasoning and tool-call tags.
- `data`: strictly decodes UTF-8 input and builds sliding-window, packed, or streaming next-token prediction examples from text corpora plus response-masked SFT examples from prompt/response, chat, reasoning, and tool-call JSONL records.
- `chat`: renders native chat/tool prompts and parses reasoning/tool-call tags from generated assistant text.
- `distillation`: loads native teacher checkpoints and computes masked soft-logit and on-policy distillation losses.
- `dpo`: computes response sequence logprobs and Direct Preference Optimization loss.
- `grpo`: computes group-relative advantages and GRPO/CISPO policy losses with reference KL.
- `ppo`: adds a value head wrapper, token-level rollout accounting, GAE, and clipped PPO policy/value loss.
- `reward`: trains scalar reward models, loads learned reward scorer checkpoints, and keeps contains, exact-match, and tool-call rule rewards behind the same scorer contract.
- `model`: implements the causal language model, RoPE and optional YaRN-style RoPE scaling, optional routed MoE feed-forward layers, optional activation checkpointing, trailing-logit forward slicing, KV-cache generation, generation-time filtering, and native beam search.
- `peft`: injects LoRA adapters into target linear modules, freezes non-adapter parameters, extracts adapter state, and merges adapters back into plain linear weights.
- `training`: owns objective selection, device/dtype selection, TF32 runtime setup, optimizer setup, learning-rate schedule, EMA tracking, evaluation, run recording, checkpointing, and resume.
- `evaluation`: restores native checkpoints and reports held-out language-model, preference, and reward-model metrics.
- `benchmark`: runs strict JSON/TOML benchmark suites over native evaluation functions plus lightweight generated tool-call checks without pulling in a heavy harness.
- `checkpoint`: enforces restricted checkpoint deserialization and exposes lightweight checkpoint inspection, LoRA merge, and optional tensor artifact export for CLI and tests.
- `sampling`: restores checkpoints and exposes text generation, chat generation, and caller-supplied Python tool-call loops.

## Runtime Contract

- Configs and UTF-8 input files fail before training starts when values, encodings, shapes, intervals, or unsupported dtypes are invalid.
- Checkpoints are ordinary tensor/plain-data `torch.save` dictionaries loaded through the restricted `weights_only=True` path on supported PyTorch versions, with schema version, objective, model state, model config, train config, optional objective configs, tokenizer path, optimizer state, completed step, optional EMA state, and optional RNG state.
- `latest.pt` and step checkpoints are written through a temporary file and atomically replaced; `train.keep_last_checkpoints` can prune older numbered step checkpoints while preserving `latest.pt`.
- `config.json` captures the validated run config and `metrics.jsonl` records train, eval, and checkpoint events under `train.out_dir`.
- Tokenizer vocabulary size is loaded from the tokenizer artifact and becomes the model vocabulary at runtime.
- Plain-text pretraining can use dense `sliding_window`, non-overlapping `packed`, or local-file `streaming` data modes.
- `allow_tf32`, `gradient_checkpointing`, `fused_adamw`, `keep_last_checkpoints`, and `ema_decay` are explicit train config flags so stability/performance/storage behavior is visible in saved configs.
- Ordinary RoPE remains the default. Setting `model.rope_scaling = "yarn"` precomputes native YaRN-style scaled RoPE buffers from `rope_original_context_length`, `rope_scaling_factor`, and the optional YaRN beta/attention controls; configs that would make scaling a no-op or silently ignored fail validation.
- Dense SwiGLU blocks remain the default model path. Setting `model.moe_num_experts` to at least `2` replaces each feed-forward block with a native top-k routed SwiGLU expert layer and records `aux_loss` in `CausalLMOutput`; native training objectives add the weighted router balance loss when a MoE policy or reward backbone participates in the optimized loss.
- `AnilaLM.forward(..., logits_to_keep=N)` can materialize only trailing logits for inference-style next-token scoring, while `targets` still require the ordinary full-logit training path.
- `AnilaLM.generate` uses a native KV cache by default for single-path generation, requests only trailing next-token logits, supports sampling, greedy decoding, and deterministic beam search, preserves terminal EOS rows in batches, and falls back to full-context recomputation whenever the context window must be rebased.
- `generate_chat` and `anila model chat` render local chat prompts over the same native sampling path, use the checkpoint `sft_config` when present, and parse `<think>` plus `<tool_call>` spans without starting a server. `generate_tool_chat` can execute caller-supplied Python callbacks, append JSON tool observations as `tool` messages, and continue generation for local tool-use loops bounded by both rounds and per-turn call count; the CLI does not execute external tools.
- Evaluation snapshots and restores RNG state so validation, including rollout-based objectives, does not alter subsequent training sampling. When EMA is enabled, evaluation temporarily swaps to EMA weights and restores the training weights afterward.
- `pretrain` consumes plain text and trains all next-token targets.
- `sft` consumes JSONL records and sets non-assistant labels to `-100`, so prompt/user/system/tool-response tokens do not contribute to loss while assistant response, reasoning, and tool-call spans train the model.
- `distill` can run hard-label distillation over pretraining or SFT data, or soft-logit distillation against a teacher checkpoint.
- `opd` consumes prompt JSONL records, samples responses from the current student policy, masks prompt tokens, and distills generated response tokens against a native teacher checkpoint.
- `dpo` consumes prompt/chosen/rejected JSONL records and compares policy log-ratios against a frozen reference model.
- `reward_model` consumes prompt/chosen/rejected JSONL records, scores chosen and rejected response spans with a scalar reward head, and trains with pairwise Bradley-Terry loss.
- `grpo` consumes prompt JSONL records, samples multiple responses per prompt, computes rule or learned-model rewards, normalizes advantages per prompt group, and regularizes against a frozen reference model. `grpo.loss_type = "grpo"` uses the clipped ratio objective; `grpo.loss_type = "cispo"` keeps the same rollout path and uses a detached upper-clipped ratio as the policy weight so capped ratios still leave a log-probability gradient path. `grpo.reward_type = "tool_call"` scores native tool-call structure, expected tool names, final-answer targets, reasoning tag shape, and repetition.
- `ppo` consumes prompt JSONL records, samples online responses, assigns terminal scorer rewards plus per-token reference KL penalties, trains a value head with GAE returns, and keeps base LM checkpoints sampleable. `ppo.reward_type = "tool_call"` uses the same native tool-call rule scorer as GRPO.
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
- Add additional RoPE scaling modes only after YaRN coverage has larger-run evidence and documented compatibility boundaries.
- Add fused or distributed MoE routing only behind a separate optional runtime adapter after the native single-process path has larger-run coverage.
- Add distributed training behind a separate runtime module after single-process coverage is stronger.
