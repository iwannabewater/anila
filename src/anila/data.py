from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from anila.config import DPOConfig, GRPOConfig, PPOConfig, RewardConfig, SFTConfig
from anila.tokenization import AnilaTokenizer

IGNORE_INDEX = -100
PathInput = str | Path | Sequence[str | Path]


def normalize_paths(paths: PathInput) -> list[Path]:
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(path) for path in paths]


class TextTokenDataset(Dataset):
    def __init__(self, path: PathInput, tokenizer: AnilaTokenizer, context_length: int):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        text = "\n".join(
            input_path.read_text(encoding="utf-8", errors="ignore") for input_path in normalize_paths(path)
        )
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) < context_length + 2:
            raise ValueError(
                f"Dataset has {len(ids)} tokens, but at least {context_length + 2} are required. "
                "Use a smaller context_length or a larger corpus."
            )
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index : index + self.context_length + 1]
        return chunk[:-1], chunk[1:]


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
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
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
            content = self._require_str(message, self.config.content_key, path=path, line_number=line_number)
            prefix = self._prefix_for_role(role, path=path, line_number=line_number)
            is_assistant = role == self.config.assistant_role
            self._append_text(token_ids, trainable, f"{prefix} ", train=False)
            self._append_text(token_ids, trainable, f"{content}\n", train=is_assistant)
            last_role = role
        token_ids.append(self.tokenizer.eos_id)
        trainable.append(last_role == self.config.assistant_role)
        return token_ids, trainable

    def _prefix_for_role(self, role: str, *, path: Path, line_number: int) -> str:
        if role == self.config.system_role:
            return self.config.system_prefix
        if role == self.config.user_role:
            return self.config.user_prefix
        if role == self.config.assistant_role:
            return self.config.assistant_prefix
        raise ValueError(f"Unsupported SFT role {role!r} in {path}:{line_number}")

    @staticmethod
    def _require_str(record: dict, key: str, *, path: Path, line_number: int) -> str:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"SFT record in {path}:{line_number} requires non-empty string field {key!r}")
        return value

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
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
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
        config: GRPOConfig | PPOConfig | None = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.config = (config or GRPOConfig()).validated()
        self.examples: list[tuple[torch.Tensor, str | None]] = []
        for input_path in normalize_paths(path):
            self._load_path(input_path)
        if not self.examples:
            raise ValueError("prompt reward dataset is empty")

    def _load_path(self, path: Path) -> None:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Prompt reward record in {path}:{line_number} must be a JSON object")
                self.examples.append(self._record_to_example(record, path=path, line_number=line_number))

    def _record_to_example(self, record: dict, *, path: Path, line_number: int) -> tuple[torch.Tensor, str | None]:
        prompt = self._require_str(record, self.config.prompt_key, path=path, line_number=line_number)
        expected = None
        if self.config.expected_key in record:
            expected = self._require_str(record, self.config.expected_key, path=path, line_number=line_number)
        token_ids = [self.tokenizer.bos_id]
        if self.config.system_key in record:
            system = self._require_str(record, self.config.system_key, path=path, line_number=line_number)
            token_ids.extend(self.tokenizer.encode(f"System: {system}\n"))
        token_ids.extend(self.tokenizer.encode(f"User: {prompt}\nAssistant: "))
        if len(token_ids) > self.context_length:
            raise ValueError(
                f"Prompt reward prompt in {path}:{line_number} has {len(token_ids)} tokens, "
                f"but at most {self.context_length} are supported by context_length={self.context_length}"
            )
        return torch.tensor(token_ids, dtype=torch.long), expected

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
    grpo_config: GRPOConfig | None = None,
    ppo_config: PPOConfig | None = None,
    reward_config: RewardConfig | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    if objective == "pretrain":
        dataset = TextTokenDataset(path, tokenizer, context_length)
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
        dataset = PromptRewardDataset(path, tokenizer, context_length, grpo_config)
        collate_fn = PromptRewardCollator(tokenizer.pad_id)
    elif objective == "ppo":
        dataset = PromptRewardDataset(path, tokenizer, context_length, ppo_config)
        collate_fn = PromptRewardCollator(tokenizer.pad_id)
    else:
        raise ValueError("objective must be pretrain, sft, dpo, reward_model, grpo, or ppo")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
