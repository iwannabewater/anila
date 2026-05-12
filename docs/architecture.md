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
                          sampler
```

## Module Responsibilities

- `config`: parses JSON/TOML configs, applies defaults, and rejects unknown keys.
- `tokenization`: trains and loads byte-level BPE tokenizers.
- `data`: builds contiguous next-token prediction examples from text corpora and response-masked SFT examples from JSONL records.
- `distillation`: loads native teacher checkpoints and computes masked soft-logit distillation loss.
- `dpo`: computes response sequence logprobs and Direct Preference Optimization loss.
- `grpo`: computes group-relative advantages and clipped GRPO loss with reference KL.
- `ppo`: adds a value head wrapper, token-level rollout accounting, GAE, and clipped PPO policy/value loss.
- `reward`: trains scalar reward models, loads learned reward scorer checkpoints, and keeps rule rewards behind the same scorer contract.
- `model`: implements the causal language model and generation-time filtering.
- `peft`: injects LoRA adapters into target linear modules, freezes non-adapter parameters, and extracts adapter state.
- `training`: owns objective selection, device/dtype selection, optimizer setup, learning-rate schedule, evaluation, run recording, checkpointing, and resume.
- `checkpoint`: exposes lightweight checkpoint inspection for CLI and tests.
- `sampling`: restores checkpoints and exposes text generation.

## Runtime Contract

- Configs fail before training starts when shapes, intervals, or unsupported dtypes are invalid.
- Checkpoints are ordinary `torch.save` dictionaries with schema version, objective, model state, model config, train config, optional objective configs, tokenizer path, optimizer state, and completed step.
- `latest.pt` and step checkpoints are written through a temporary file and atomically replaced.
- `config.json` captures the validated run config and `metrics.jsonl` records train, eval, and checkpoint events under `train.out_dir`.
- Tokenizer vocabulary size is loaded from the tokenizer artifact and becomes the model vocabulary at runtime.
- `pretrain` consumes plain text and trains all next-token targets.
- `sft` consumes JSONL records and sets non-assistant labels to `-100`, so prompt/user/system tokens do not contribute to loss.
- `distill` can run hard-label distillation over pretraining or SFT data, or soft-logit distillation against a teacher checkpoint.
- `dpo` consumes prompt/chosen/rejected JSONL records and compares policy log-ratios against a frozen reference model.
- `reward_model` consumes prompt/chosen/rejected JSONL records, scores chosen and rejected response spans with a scalar reward head, and trains with pairwise Bradley-Terry loss.
- `grpo` consumes prompt JSONL records, samples multiple responses per prompt, computes rule or learned-model rewards, normalizes advantages per prompt group, and regularizes against a frozen reference model.
- `ppo` consumes prompt JSONL records, samples online responses, assigns terminal scorer rewards plus per-token reference KL penalties, trains a value head with GAE returns, and keeps base LM checkpoints sampleable.
- LoRA can wrap selected projection modules before training. Full checkpoints include base and adapter weights; adapter-only checkpoints are saved separately when enabled.

## Extension Points

- Move dataset adapters into a `datasets/` package if more formats arrive.
- Add new model variants by keeping the `forward(input_ids, targets=None)` interface.
- Add LoRA merge/export as a small artifact-management slice.
- Add DPO variants only after the base DPO path has real-data coverage.
- Add external reward backends only after native learned reward checkpoints have broader real-data coverage.
- Add hidden-state distillation only after model variant contracts are stable.
- Add distributed training behind a separate runtime module after single-process coverage is stronger.
