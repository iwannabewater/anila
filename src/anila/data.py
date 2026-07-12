from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from anila._json import dumps_strict_json, loads_strict_json
from anila.chat import render_chat_prompt
from anila.config import DataConfig, DPOConfig, GRPOConfig, OPDConfig, PPOConfig, RewardConfig, SFTConfig
from anila.tokenization import AnilaTokenizer

IGNORE_INDEX = -100
PathInput = str | Path | Sequence[str | Path]


def normalize_paths(paths: PathInput) -> list[Path]:
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(path) for path in paths]


class TextTokenDataset(Dataset):
    def __init__(
        self,
        path: PathInput,
        tokenizer: AnilaTokenizer,
        context_length: int,
        config: DataConfig | None = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.config = (config or DataConfig()).validated()
        if self.config.pretrain_mode == "streaming":
            raise ValueError("Use StreamingTextTokenDataset when data.pretrain_mode is streaming")
        text = "\n".join(input_path.read_text(encoding="utf-8") for input_path in normalize_paths(path))
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) < context_length + 1:
            raise ValueError(
                f"Dataset has {len(ids)} tokens, but at least {context_length + 1} are required. "
                "Use a smaller context_length or a larger corpus."
            )
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = _pretrain_stride(self.config, context_length)
        self.num_examples = _num_pretrain_examples(len(self.tokens), context_length, self.stride)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_start = index * self.stride
        chunk = self.tokens[chunk_start : chunk_start + self.context_length + 1]
        return chunk[:-1], chunk[1:]


class StreamingTextTokenDataset(IterableDataset):
    def __init__(
        self,
        path: PathInput,
        tokenizer: AnilaTokenizer,
        context_length: int,
        config: DataConfig | None = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.paths = normalize_paths(path)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.config = (config or DataConfig(pretrain_mode="streaming")).validated()
        if self.config.pretrain_mode != "streaming":
            raise ValueError("StreamingTextTokenDataset requires data.pretrain_mode = streaming")

    def __iter__(self):
        emitted = False
        buffer: list[int] = [self.tokenizer.bos_id]
        worker = get_worker_info()
        for input_path in _worker_paths(self.paths):
            with input_path.open("r", encoding="utf-8") as f:
                for line in f:
                    buffer.extend(self.tokenizer.encode(line))
                    while len(buffer) >= self.context_length + 1:
                        yield _tokens_to_lm_pair(buffer[: self.context_length + 1])
                        emitted = True
                        del buffer[: self.context_length]
        buffer.append(self.tokenizer.eos_id)
        while len(buffer) >= self.context_length + 1:
            yield _tokens_to_lm_pair(buffer[: self.context_length + 1])
            emitted = True
            del buffer[: self.context_length]
        if not emitted and worker is None:
            raise ValueError(
                f"Streaming dataset produced no {self.context_length}-token examples. "
                "Use a smaller context_length or a larger corpus."
            )


class SupervisedFineTuneDataset(Dataset):
    def __init__(self, path: PathInput, tokenizer: AnilaTokenizer, context_length: int, config: SFTConfig | None = None):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.config = (config or SFTConfig()).validated()
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for input_path in normalize_paths(path):
            self._load_path(input_path)
        if not self.examples:
            raise ValueError("SFT dataset is empty")

    def _load_path(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = loads_strict_json(line)
                except ValueError as exc:
                    detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {detail}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"SFT record in {path}:{line_number} must be a JSON object")
                input_ids, labels = self._record_to_tensors(record, path=path, line_number=line_number)
                self.examples.append((input_ids, labels))

    def _record_to_tensors(
        self, record: dict, *, path: Path, line_number: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, trainable = self._record_to_tokens(record, path=path, line_number=line_number)
        if len(token_ids) < 2:
            raise ValueError(f"SFT record in {path}:{line_number} produced fewer than two tokens")
        if len(token_ids) > self.context_length + 1:
            raise ValueError(
                f"SFT record in {path}:{line_number} has {len(token_ids)} tokens, "
                f"but at most {self.context_length + 1} are supported by context_length={self.context_length}"
            )
        input_ids = token_ids[:-1]
        labels = [token_ids[index + 1] if trainable[index + 1] else IGNORE_INDEX for index in range(len(token_ids) - 1)]
        if all(label == IGNORE_INDEX for label in labels):
            raise ValueError(f"SFT record in {path}:{line_number} has no trainable assistant tokens")
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _record_to_tokens(self, record: dict, *, path: Path, line_number: int) -> tuple[list[int], list[bool]]:
        fmt = self.config.format
        if fmt == "auto":
            fmt = "messages" if self.config.messages_key in record else "prompt_response"
        if fmt == "messages":
            return self._messages_to_tokens(record, path=path, line_number=line_number)
        return self._prompt_response_to_tokens(record, path=path, line_number=line_number)

    def _prompt_response_to_tokens(
        self, record: dict, *, path: Path, line_number: int
    ) -> tuple[list[int], list[bool]]:
        prompt = self._require_str(record, self.config.prompt_key, path=path, line_number=line_number)
        response = self._require_str(record, self.config.response_key, path=path, line_number=line_number)
        token_ids = [self.tokenizer.bos_id]
        trainable = [False]
        if self.config.system_key in record:
            system = self._require_str(record, self.config.system_key, path=path, line_number=line_number)
            self._append_text(token_ids, trainable, f"{self.config.system_prefix} {system}\n", train=False)
        self._append_text(token_ids, trainable, f"{self.config.user_prefix} {prompt}\n", train=False)
        self._append_text(token_ids, trainable, f"{self.config.assistant_prefix} ", train=False)
        self._append_text(token_ids, trainable, response, train=True)
        token_ids.append(self.tokenizer.eos_id)
        trainable.append(True)
        return token_ids, trainable

    def _messages_to_tokens(self, record: dict, *, path: Path, line_number: int) -> tuple[list[int], list[bool]]:
        messages = record.get(self.config.messages_key)
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"SFT record in {path}:{line_number} requires a non-empty {self.config.messages_key} list")
        token_ids = [self.tokenizer.bos_id]
        trainable = [False]
        last_role = None
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"SFT message {message_index} in {path}:{line_number} must be an object")
            role = self._require_str(message, self.config.role_key, path=path, line_number=line_number)
            self._append_message(
                token_ids,
                trainable,
                message,
                role=role,
                path=path,
                line_number=line_number,
            )
            last_role = role
        token_ids.append(self.tokenizer.eos_id)
        trainable.append(last_role == self.config.assistant_role)
        return token_ids, trainable

    def _append_message(
        self,
        token_ids: list[int],
        trainable: list[bool],
        message: dict,
        *,
        role: str,
        path: Path,
        line_number: int,
    ) -> None:
        prefix = self._prefix_for_role(role, path=path, line_number=line_number)
        self._append_text(token_ids, trainable, f"{prefix} ", train=False)
        if role == self.config.assistant_role:
            self._append_assistant_message(token_ids, trainable, message, path=path, line_number=line_number)
            return
        content = self._require_message_content(message, role=role, path=path, line_number=line_number)
        if role == self.config.tool_role:
            self._append_text(
                token_ids,
                trainable,
                f"{self.config.tool_response_start}\n{content}\n{self.config.tool_response_end}\n",
                train=False,
            )
            return
        self._append_text(token_ids, trainable, f"{content}\n", train=False)

    def _append_assistant_message(
        self,
        token_ids: list[int],
        trainable: list[bool],
        message: dict,
        *,
        path: Path,
        line_number: int,
    ) -> None:
        content = self._optional_str(message, self.config.content_key, path=path, line_number=line_number)
        reasoning_present = self.config.reasoning_key in message
        reasoning = self._optional_str(message, self.config.reasoning_key, path=path, line_number=line_number)
        tool_calls = self._tool_call_payloads(message, path=path, line_number=line_number)
        if not content and not reasoning_present and not tool_calls:
            raise ValueError(
                f"SFT assistant message in {path}:{line_number} requires content, "
                f"{self.config.reasoning_key!r}, or {self.config.tool_calls_key!r}"
            )
        if reasoning_present:
            self._append_text(
                token_ids,
                trainable,
                f"{self.config.thinking_start}\n{reasoning or ''}\n{self.config.thinking_end}\n\n",
                train=True,
            )
        if content:
            self._append_text(token_ids, trainable, content, train=True)
        if tool_calls:
            if content and not content.endswith("\n"):
                self._append_text(token_ids, trainable, "\n", train=True)
            for index, payload in enumerate(tool_calls):
                if index > 0:
                    self._append_text(token_ids, trainable, "\n", train=True)
                self._append_text(
                    token_ids,
                    trainable,
                    f"{self.config.tool_call_start}\n{payload}\n{self.config.tool_call_end}",
                    train=True,
                )
        self._append_text(token_ids, trainable, "\n", train=True)

    def _prefix_for_role(self, role: str, *, path: Path, line_number: int) -> str:
        if role == self.config.system_role:
            return self.config.system_prefix
        if role == self.config.user_role:
            return self.config.user_prefix
        if role == self.config.assistant_role:
            return self.config.assistant_prefix
        if role == self.config.tool_role:
            return self.config.tool_prefix
        raise ValueError(f"Unsupported SFT role {role!r} in {path}:{line_number}")

    @staticmethod
    def _require_str(record: dict, key: str, *, path: Path, line_number: int) -> str:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"SFT record in {path}:{line_number} requires non-empty string field {key!r}")
        return value

    def _require_message_content(self, message: dict, *, role: str, path: Path, line_number: int) -> str:
        value = self._optional_str(message, self.config.content_key, path=path, line_number=line_number)
        if not value:
            raise ValueError(
                f"SFT {role} message in {path}:{line_number} requires non-empty string field "
                f"{self.config.content_key!r}"
            )
        return value

    @staticmethod
    def _optional_str(record: dict, key: str, *, path: Path, line_number: int) -> str | None:
        if key not in record:
            return None
        value = record.get(key)
        if not isinstance(value, str):
            raise ValueError(f"SFT record in {path}:{line_number} field {key!r} must be a string")
        return value

    def _tool_call_payloads(self, message: dict, *, path: Path, line_number: int) -> list[str]:
        if self.config.tool_calls_key not in message:
            return []
        raw_value = message[self.config.tool_calls_key]
        if isinstance(raw_value, str):
            if not raw_value:
                raise ValueError(
                    f"SFT record in {path}:{line_number} field {self.config.tool_calls_key!r} cannot be empty"
            )
            try:
                value = loads_strict_json(raw_value)
            except ValueError as exc:
                detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(
                    f"Invalid JSON in SFT field {self.config.tool_calls_key!r} at {path}:{line_number}: {detail}"
                ) from exc
        else:
            value = raw_value
        calls = [value] if isinstance(value, dict) else value
        if not isinstance(calls, list) or not calls:
            raise ValueError(
                f"SFT record in {path}:{line_number} field {self.config.tool_calls_key!r} must be an object or "
                "a non-empty list of objects"
            )
        rendered: list[str] = []
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ValueError(
                    f"SFT tool call {index} in {path}:{line_number} must be an object"
                )
            try:
                rendered.append(dumps_strict_json(call, ensure_ascii=False, sort_keys=True))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"SFT tool call {index} in {path}:{line_number} must be JSON serializable with finite numbers"
                ) from exc
        return rendered

    def _append_text(self, token_ids: list[int], trainable: list[bool], text: str, *, train: bool) -> None:
        ids = self.tokenizer.encode(text)
        token_ids.extend(ids)
        trainable.extend([train] * len(ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


class SFTCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(input_ids.size(0) for input_ids, _ in batch)
        input_batch = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        label_batch = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        for index, (input_ids, labels) in enumerate(batch):
            input_batch[index, : input_ids.size(0)] = input_ids
            label_batch[index, : labels.size(0)] = labels
        return input_batch, label_batch


class PreferenceDataset(Dataset):
    def __init__(
        self,
        path: PathInput,
        tokenizer: AnilaTokenizer,
        context_length: int,
        config: DPOConfig | RewardConfig | None = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.config = (config or DPOConfig()).validated()
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for input_path in normalize_paths(path):
            self._load_path(input_path)
        if not self.examples:
            raise ValueError("preference dataset is empty")

    def _load_path(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = loads_strict_json(line)
                except ValueError as exc:
                    detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {detail}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"DPO record in {path}:{line_number} must be a JSON object")
                self.examples.append(self._record_to_tensors(record, path=path, line_number=line_number))

    def _record_to_tensors(
        self, record: dict, *, path: Path, line_number: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self._require_str(record, self.config.prompt_key, path=path, line_number=line_number)
        chosen = self._require_str(record, self.config.chosen_key, path=path, line_number=line_number)
        rejected = self._require_str(record, self.config.rejected_key, path=path, line_number=line_number)
        system = None
        if self.config.system_key in record:
            system = self._require_str(record, self.config.system_key, path=path, line_number=line_number)
        chosen_input_ids, chosen_labels = self._format_pair(
            prompt, chosen, system=system, path=path, line_number=line_number, response_kind=self.config.chosen_key
        )
        rejected_input_ids, rejected_labels = self._format_pair(
            prompt, rejected, system=system, path=path, line_number=line_number, response_kind=self.config.rejected_key
        )
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def _format_pair(
        self,
        prompt: str,
        response: str,
        *,
        system: str | None,
        path: Path,
        line_number: int,
        response_kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = [self.tokenizer.bos_id]
        trainable = [False]
        if system is not None:
            self._append_text(token_ids, trainable, f"System: {system}\n", train=False)
        self._append_text(token_ids, trainable, f"User: {prompt}\n", train=False)
        self._append_text(token_ids, trainable, "Assistant: ", train=False)
        self._append_text(token_ids, trainable, response, train=True)
        token_ids.append(self.tokenizer.eos_id)
        trainable.append(True)
        if len(token_ids) > self.context_length + 1:
            raise ValueError(
                f"DPO {response_kind} response in {path}:{line_number} has {len(token_ids)} tokens, "
                f"but at most {self.context_length + 1} are supported by context_length={self.context_length}"
            )
        input_ids = token_ids[:-1]
        labels = [token_ids[index + 1] if trainable[index + 1] else IGNORE_INDEX for index in range(len(token_ids) - 1)]
        if all(label == IGNORE_INDEX for label in labels):
            raise ValueError(f"DPO {response_kind} response in {path}:{line_number} has no trainable tokens")
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _require_str(record: dict, key: str, *, path: Path, line_number: int) -> str:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"DPO record in {path}:{line_number} requires non-empty string field {key!r}")
        return value

    def _append_text(self, token_ids: list[int], trainable: list[bool], text: str, *, train: bool) -> None:
        ids = self.tokenizer.encode(text)
        token_ids.extend(ids)
        trainable.extend([train] * len(ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[index]


class DPOCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen = [(item[0], item[1]) for item in batch]
        rejected = [(item[2], item[3]) for item in batch]
        chosen_input_ids, chosen_labels = self._pad(chosen)
        rejected_input_ids, rejected_labels = self._pad(rejected)
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def _pad(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(input_ids.size(0) for input_ids, _ in batch)
        input_batch = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        label_batch = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        for index, (input_ids, labels) in enumerate(batch):
            input_batch[index, : input_ids.size(0)] = input_ids
            label_batch[index, : labels.size(0)] = labels
        return input_batch, label_batch


class PromptRewardDataset(Dataset):
    def __init__(
        self,
        path: PathInput,
        tokenizer: AnilaTokenizer,
        context_length: int,
        config: GRPOConfig | OPDConfig | PPOConfig | None = None,
        sft_config: SFTConfig | None = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.config = (config or GRPOConfig()).validated()
        self.sft_config = (sft_config or SFTConfig()).validated()
        self.examples: list[tuple[torch.Tensor, str | None]] = []
        for input_path in normalize_paths(path):
            self._load_path(input_path)
        if not self.examples:
            raise ValueError("prompt reward dataset is empty")

    def _load_path(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = loads_strict_json(line)
                except ValueError as exc:
                    detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {detail}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Prompt reward record in {path}:{line_number} must be a JSON object")
                self.examples.append(self._record_to_example(record, path=path, line_number=line_number))

    def _record_to_example(self, record: dict, *, path: Path, line_number: int) -> tuple[torch.Tensor, str | None]:
        expected = None if isinstance(self.config, OPDConfig) else self._optional_expected(record, path=path, line_number=line_number)
        token_ids = [self.tokenizer.bos_id]
        prompt_text = self._record_to_prompt(record, path=path, line_number=line_number)
        token_ids.extend(self.tokenizer.encode(prompt_text))
        if len(token_ids) > self.context_length:
            raise ValueError(
                f"Prompt reward prompt in {path}:{line_number} has {len(token_ids)} tokens, "
                f"but at most {self.context_length} are supported by context_length={self.context_length}"
            )
        return torch.tensor(token_ids, dtype=torch.long), expected

    def _record_to_prompt(self, record: dict, *, path: Path, line_number: int) -> str:
        has_messages = self.sft_config.messages_key in record
        has_prompt = self.config.prompt_key in record
        tools = self._optional_tools(record, path=path, line_number=line_number)
        if has_messages:
            if has_prompt:
                raise ValueError(
                    f"Prompt reward record in {path}:{line_number} cannot mix "
                    f"{self.sft_config.messages_key!r} and {self.config.prompt_key!r}"
                )
            messages = record[self.sft_config.messages_key]
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Prompt reward record in {path}:{line_number} requires a non-empty "
                    f"{self.sft_config.messages_key} list"
                )
            return self._render_chat_prompt(messages, tools=tools, path=path, line_number=line_number)

        prompt = self._require_str(record, self.config.prompt_key, path=path, line_number=line_number)
        if tools is not None:
            messages: list[dict[str, str]] = []
            if self.config.system_key in record:
                system = self._require_str(record, self.config.system_key, path=path, line_number=line_number)
                messages.append({self.sft_config.role_key: self.sft_config.system_role, self.sft_config.content_key: system})
            messages.append({self.sft_config.role_key: self.sft_config.user_role, self.sft_config.content_key: prompt})
            return self._render_chat_prompt(messages, tools=tools, path=path, line_number=line_number)

        chunks: list[str] = []
        if self.config.system_key in record:
            system = self._require_str(record, self.config.system_key, path=path, line_number=line_number)
            chunks.append(f"System: {system}\n")
        chunks.append(f"User: {prompt}\nAssistant: ")
        return "".join(chunks)

    def _render_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        path: Path,
        line_number: int,
    ) -> str:
        try:
            return render_chat_prompt(messages, tools=tools, add_generation_prompt=True, config=self.sft_config)
        except ValueError as exc:
            raise ValueError(f"Invalid prompt reward chat record in {path}:{line_number}: {exc}") from exc

    def _optional_expected(self, record: dict, *, path: Path, line_number: int) -> str | None:
        if self.config.expected_key not in record:
            return None
        value = record[self.config.expected_key]
        if isinstance(value, str):
            if not value:
                raise ValueError(
                    f"Prompt reward record in {path}:{line_number} requires non-empty string field "
                    f"{self.config.expected_key!r}"
                )
            return value
        if getattr(self.config, "reward_type", None) == "tool_call" and isinstance(value, dict | list):
            if not value:
                raise ValueError(
                    f"Prompt reward record in {path}:{line_number} field {self.config.expected_key!r} cannot be empty"
                )
            try:
                return dumps_strict_json(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Prompt reward record in {path}:{line_number} field {self.config.expected_key!r} must be "
                    "JSON serializable with finite numbers"
                ) from exc
        raise ValueError(
            f"Prompt reward record in {path}:{line_number} field {self.config.expected_key!r} must be a "
            "non-empty string"
        )

    @staticmethod
    def _optional_tools(record: dict, *, path: Path, line_number: int) -> list[dict[str, Any]] | None:
        if "tools" not in record:
            return None
        tools = record["tools"]
        if not isinstance(tools, list) or not tools:
            raise ValueError(f"Prompt reward record in {path}:{line_number} field 'tools' must be a non-empty list")
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"Prompt reward tool {index} in {path}:{line_number} must be an object")
        return tools

    @staticmethod
    def _require_str(record: dict, key: str, *, path: Path, line_number: int) -> str:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Prompt reward record in {path}:{line_number} requires non-empty string field {key!r}")
        return value

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str | None]:
        return self.examples[index]


class PromptRewardCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[torch.Tensor, str | None]]) -> tuple[torch.Tensor, torch.Tensor, list[str | None]]:
        max_len = max(input_ids.size(0) for input_ids, _ in batch)
        input_batch = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        lengths = torch.empty(len(batch), dtype=torch.long)
        expected: list[str | None] = []
        for index, (input_ids, target) in enumerate(batch):
            input_batch[index, : input_ids.size(0)] = input_ids
            lengths[index] = input_ids.size(0)
            expected.append(target)
        return input_batch, lengths, expected


def create_dataloader(
    path: PathInput,
    tokenizer: AnilaTokenizer,
    *,
    context_length: int,
    batch_size: int,
    objective: str = "pretrain",
    sft_config: SFTConfig | None = None,
    dpo_config: DPOConfig | None = None,
    data_config: DataConfig | None = None,
    grpo_config: GRPOConfig | None = None,
    opd_config: OPDConfig | None = None,
    ppo_config: PPOConfig | None = None,
    reward_config: RewardConfig | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if objective == "pretrain":
        resolved_data_config = (data_config or DataConfig()).validated()
        if resolved_data_config.pretrain_mode == "streaming":
            dataset = StreamingTextTokenDataset(path, tokenizer, context_length, resolved_data_config)
        else:
            dataset = TextTokenDataset(path, tokenizer, context_length, resolved_data_config)
        collate_fn = None
    elif objective == "sft":
        dataset = SupervisedFineTuneDataset(path, tokenizer, context_length, sft_config)
        collate_fn = SFTCollator(tokenizer.pad_id)
    elif objective == "dpo":
        dataset = PreferenceDataset(path, tokenizer, context_length, dpo_config)
        collate_fn = DPOCollator(tokenizer.pad_id)
    elif objective == "reward_model":
        dataset = PreferenceDataset(path, tokenizer, context_length, reward_config)
        collate_fn = DPOCollator(tokenizer.pad_id)
    elif objective == "grpo":
        dataset = PromptRewardDataset(path, tokenizer, context_length, grpo_config, sft_config=sft_config)
        collate_fn = PromptRewardCollator(tokenizer.pad_id)
    elif objective == "opd":
        dataset = PromptRewardDataset(path, tokenizer, context_length, opd_config, sft_config=sft_config)
        collate_fn = PromptRewardCollator(tokenizer.pad_id)
    elif objective == "ppo":
        dataset = PromptRewardDataset(path, tokenizer, context_length, ppo_config, sft_config=sft_config)
        collate_fn = PromptRewardCollator(tokenizer.pad_id)
    else:
        raise ValueError("objective must be pretrain, sft, dpo, reward_model, opd, grpo, or ppo")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not isinstance(dataset, IterableDataset),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_fn,
        generator=generator,
    )


def _pretrain_stride(config: DataConfig, context_length: int) -> int:
    if config.pretrain_mode == "packed":
        return context_length
    return config.sequence_stride or 1


def _num_pretrain_examples(token_count: int, context_length: int, stride: int) -> int:
    return max(0, ((token_count - (context_length + 1)) // stride) + 1)


def _tokens_to_lm_pair(tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = torch.tensor(tokens, dtype=torch.long)
    return chunk[:-1], chunk[1:]


def _worker_paths(paths: list[Path]) -> list[Path]:
    worker = get_worker_info()
    if worker is None:
        return paths
    return paths[worker.id :: worker.num_workers]
