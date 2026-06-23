# Data Contracts

This page documents the input files Anila accepts for each objective. Use it when you replace the tiny examples with your own data.

All training input is strict UTF-8. Invalid bytes fail during dataset or tokenizer loading instead of being skipped.

## Plain Text Pretraining

Used by:

- `train.objective = "pretrain"`
- `train.objective = "distill"` with `distill.data_objective = "pretrain"`

Config fields:

- `train.dataset_path`: one UTF-8 text file path or a list of paths.
- `data.pretrain_mode`: `sliding_window`, `packed`, or `streaming`.
- `data.sequence_stride`: optional positive integer, only valid with `sliding_window`.

Example:

```text
Anila is a small language model training project.
It learns from plain text and predicts the next token.
```

Mode behavior:

- `sliding_window` builds overlapping next-token windows. This is the default for backward-compatible configs.
- `packed` builds non-overlapping fixed-length blocks. This is usually clearer for local corpora.
- `streaming` reads local text files through an iterable dataset and emits packed blocks without materializing the whole corpus as one tensor.

Failure conditions:

- The file is not valid UTF-8.
- The corpus does not produce at least one `context_length` training example.
- `data.sequence_stride` is set outside `sliding_window`.

## Supervised Fine-Tuning

Used by:

- `train.objective = "sft"`
- `train.objective = "distill"` with `distill.mode = "hard"` and `distill.data_objective = "sft"`

Each JSONL line must be a JSON object. Blank lines are ignored.

Prompt/response format:

```json
{"system": "You are a concise assistant.", "prompt": "What does Anila train?", "response": "Small causal language models."}
```

Chat messages format:

```json
{"messages": [{"role": "system", "content": "You answer briefly."}, {"role": "user", "content": "What is SFT?"}, {"role": "assistant", "content": "Supervised fine-tuning."}]}
```

Default keys and roles:

| Config field | Default |
| --- | --- |
| `sft.format` | `auto` |
| `sft.prompt_key` | `prompt` |
| `sft.response_key` | `response` |
| `sft.system_key` | `system` |
| `sft.messages_key` | `messages` |
| `sft.role_key` | `role` |
| `sft.content_key` | `content` |
| `sft.system_role` | `system` |
| `sft.user_role` | `user` |
| `sft.assistant_role` | `assistant` |
| `sft.system_prefix` | `System:` |
| `sft.user_prefix` | `User:` |
| `sft.assistant_prefix` | `Assistant:` |

Label contract:

- System, user, and assistant-prefix tokens are masked with `IGNORE_INDEX`.
- Assistant response tokens train the model.
- The EOS token trains if the last message or response is from the assistant.

Failure conditions:

- A required string field is missing or empty.
- `messages` is empty or contains non-object items.
- A message role is not one of the configured system, user, or assistant roles.
- The record is longer than `model.context_length + 1` tokens.
- The record has no trainable assistant tokens.

## DPO Preference Data

Used by:

- `train.objective = "dpo"`

Each JSONL line must contain one prompt and two candidate responses:

```json
{"system": "You are a concise assistant.", "prompt": "What does Anila train?", "chosen": "Anila trains small causal language models.", "rejected": "Anila is an image database."}
```

Default keys:

| Config field | Default |
| --- | --- |
| `dpo.prompt_key` | `prompt` |
| `dpo.chosen_key` | `chosen` |
| `dpo.rejected_key` | `rejected` |
| `dpo.system_key` | `system` |

Label contract:

- The prompt and optional system text are masked.
- Chosen and rejected response tokens train their respective sequence log-probability comparisons.
- The DPO reference checkpoint defaults to `train.init_from` when `dpo.reference_checkpoint` is omitted.

Failure conditions:

- Any required string field is missing or empty.
- Either response path exceeds `model.context_length + 1` tokens.
- Either response path has no trainable response tokens.

## Reward-Model Preference Data

Used by:

- `train.objective = "reward_model"`

Reward-model data uses the same prompt/chosen/rejected JSONL shape as DPO:

```json
{"prompt": "How are checkpoints saved?", "chosen": "They are saved atomically.", "rejected": "They are only printed to the terminal."}
```

Default keys:

| Config field | Default |
| --- | --- |
| `reward.prompt_key` | `prompt` |
| `reward.chosen_key` | `chosen` |
| `reward.rejected_key` | `rejected` |
| `reward.system_key` | `system` |

The model scores chosen and rejected response spans, then trains a pairwise Bradley-Terry reward loss.

Failure conditions:

- Any required string field is missing or empty.
- Either response path exceeds `model.context_length + 1` tokens.
- Either response path has no trainable response tokens.
- `reward.scorer = "model"` is configured without `reward.checkpoint`.
- `reward.scale` is zero.

## GRPO And PPO Prompt Data

Used by:

- `train.objective = "grpo"`
- `train.objective = "ppo"`

Rule-reward records include a prompt and an expected string:

```json
{"system": "You are a concise assistant.", "prompt": "What does Anila train?", "expected": "language models"}
```

Learned-reward records can omit `expected` because the reward model scores generated responses:

```json
{"prompt": "How are checkpoints saved?"}
```

Default keys:

| Config field | Default |
| --- | --- |
| `grpo.prompt_key` / `ppo.prompt_key` | `prompt` |
| `grpo.expected_key` / `ppo.expected_key` | `expected` |
| `grpo.system_key` / `ppo.system_key` | `system` |

Prompt construction:

```text
<bos>System: {system}
User: {prompt}
Assistant:
```

The system line is omitted when the record has no system field.

Failure conditions:

- `prompt` is missing or empty.
- `expected` is present but empty.
- The prompt prefix exceeds `model.context_length` tokens before generation starts.

## Distillation Data

Distillation selects its data shape through `distill.data_objective`:

- `pretrain`: use plain-text pretraining data.
- `sft`: use SFT JSONL data.

Hard distillation uses the selected data path with ordinary hard-label loss. Soft distillation uses the selected data path and a native Anila teacher checkpoint from `distill.teacher_checkpoint`.

## Replacing The Tiny Examples

1. Train a tokenizer on text that resembles your target corpus.
2. Keep `model.context_length` large enough for your longest formatted record plus EOS.
3. Start with one objective and one file. Run the relevant dataloader or training test before adding more data.
4. Keep generated runs under `runs/`. Commit configs and small example files, not checkpoints.

Useful checks:

```bash
uv run pytest tests/test_data.py -q
uv run pytest tests/test_config.py -q
uv run anila model train --config configs/quickstart/pretrain.json
```
