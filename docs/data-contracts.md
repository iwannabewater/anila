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

Tool and reasoning message format:

```json
{"messages": [{"role": "user", "content": "Use the calculator."}, {"role": "assistant", "content": "", "reasoning_content": "Need a tool.", "tool_calls": [{"name": "calculate_math", "arguments": {"expression": "2+2"}}]}, {"role": "tool", "content": "{\"result\":\"4\"}"}, {"role": "assistant", "content": "The answer is 4."}]}
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
| `sft.reasoning_key` | `reasoning_content` |
| `sft.tool_calls_key` | `tool_calls` |
| `sft.system_role` | `system` |
| `sft.user_role` | `user` |
| `sft.assistant_role` | `assistant` |
| `sft.tool_role` | `tool` |
| `sft.system_prefix` | `System:` |
| `sft.user_prefix` | `User:` |
| `sft.assistant_prefix` | `Assistant:` |
| `sft.tool_prefix` | `Tool:` |
| `sft.thinking_start` | `<think>` |
| `sft.thinking_end` | `</think>` |
| `sft.tool_call_start` | `<tool_call>` |
| `sft.tool_call_end` | `</tool_call>` |
| `sft.tool_response_start` | `<tool_response>` |
| `sft.tool_response_end` | `</tool_response>` |

Label contract:

- System, user, and assistant-prefix tokens are masked with `IGNORE_INDEX`.
- Assistant response, reasoning, and tool-call tokens train the model.
- Tool role responses are wrapped in `<tool_response>` tags and masked as context.
- The EOS token trains if the last message or response is from the assistant.

Failure conditions:

- A required non-assistant string field is missing or empty.
- `messages` is empty or contains non-object items.
- A message role is not one of the configured system, user, assistant, or tool roles.
- An assistant message has no `content`, `reasoning_content`, or `tool_calls`.
- `tool_calls` is not a JSON object, a non-empty list of JSON objects, or a string containing one of those shapes.
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

## OPD, GRPO, And PPO Prompt Data

Used by:

- `train.objective = "opd"`
- `train.objective = "grpo"`
- `train.objective = "ppo"`

OPD records can be prompt-only because the teacher checkpoint supplies token-level feedback on student-generated continuations:

```json
{"system": "You are a concise assistant.", "prompt": "What does Anila train?"}
```

Rule-reward records for GRPO/PPO include a prompt and an expected string:

```json
{"system": "You are a concise assistant.", "prompt": "What does Anila train?", "expected": "language models"}
```

When `grpo.reward_type` or `ppo.reward_type` is `"tool_call"`, `expected` may also be a JSON object. The native rule scorer reads `answers` as final-answer targets and `tools` as expected tool names:

```json
{"prompt": "Use the calculator.", "expected": {"answers": ["4"], "tools": ["calculate_math"]}}
```

Tool-call prompt records can use the same chat/tool rendering path as local chat inference by providing `messages` and optional `tools`:

```json
{
  "messages": [{"role": "user", "content": "Use the calculator."}],
  "tools": [{"type": "function", "function": {"name": "calculate_math"}}],
  "expected": {"answers": ["4"], "tools": ["calculate_math"]}
}
```

Learned-reward GRPO/PPO records can omit `expected` because the reward model scores generated responses:

```json
{"prompt": "How are checkpoints saved?"}
```

Default keys:

| Config field | Default |
| --- | --- |
| `opd.prompt_key` / `grpo.prompt_key` / `ppo.prompt_key` | `prompt` |
| `opd.expected_key` / `grpo.expected_key` / `ppo.expected_key` | `expected` |
| `opd.system_key` / `grpo.system_key` / `ppo.system_key` | `system` |
| `opd.teacher_checkpoint` | required for OPD |
| `grpo.loss_type` | `grpo` |
| `grpo.cispo_ratio_cap` | `5.0` |

Prompt construction:

```text
<bos>System: {system}
User: {prompt}
Assistant:
```

The system line is omitted when the record has no system field.

When a prompt reward record includes `tools`, the prompt is rendered through the native chat renderer with a tool preamble and `Assistant:` generation prompt. Records with `messages` use that chat history directly and must not also include the plain `prompt` key.

OPD ignores `expected` when it is present. It samples `opd.num_rollouts` student responses, masks the prompt prefix, and distills only generated response tokens against `opd.teacher_checkpoint`.

GRPO prompt data is unchanged when `grpo.loss_type = "cispo"`; CISPO reuses GRPO rollouts, rewards, grouped advantages, and reference KL.

Failure conditions:

- `prompt` is missing or empty.
- `expected` is present but empty.
- `expected` is an object or list while the selected GRPO/PPO reward type is not `tool_call`.
- `messages` and `prompt` are both present in a prompt reward record.
- `tools` is present but is not a non-empty list of objects.
- The prompt prefix exceeds `model.context_length` tokens before generation starts.
- OPD is configured without `opd.teacher_checkpoint`.

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
