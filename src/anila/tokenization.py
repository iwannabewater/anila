from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD = "<|pad|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
UNK = "<|unk|>"

DEFAULT_SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
DEFAULT_CHAT_SPECIAL_TOKENS = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


class AnilaTokenizer:
    def __init__(self, tokenizer: Tokenizer, *, special_tokens: Iterable[str] | None = None):
        self._tokenizer = tokenizer
        self.special_tokens = tuple(_normalize_special_tokens(special_tokens or DEFAULT_SPECIAL_TOKENS))
        self.pad_id = self.token_to_id(PAD)
        self.bos_id = self.token_to_id(BOS)
        self.eos_id = self.token_to_id(EOS)
        self.unk_id = self.token_to_id(UNK)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token: {token}")
        return int(token_id)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        preserve_added_special_tokens: bool = False,
    ) -> str:
        ids_list = list(ids)
        if skip_special_tokens and preserve_added_special_tokens:
            text = self._tokenizer.decode(ids_list, skip_special_tokens=False)
            for token in DEFAULT_SPECIAL_TOKENS:
                text = text.replace(token, "")
            return text
        return self._tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)

    def save(self, out_dir: str | Path) -> None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(output / "tokenizer.json"))
        metadata = {
            "type": "byte_bpe",
            "special_tokens": list(self.special_tokens),
            "vocab_size": self.vocab_size,
        }
        (output / "tokenizer_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AnilaTokenizer:
        root = Path(path)
        tokenizer_path = root / "tokenizer.json" if root.is_dir() else root
        special_tokens: list[str] | None = None
        config_path = root / "tokenizer_config.json" if root.is_dir() else None
        if config_path is not None and config_path.exists():
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            value = metadata.get("special_tokens")
            if isinstance(value, list) and all(isinstance(token, str) for token in value):
                special_tokens = value
        return cls(Tokenizer.from_file(str(tokenizer_path)), special_tokens=special_tokens)


def _text_iterator(paths: Iterable[str | Path]) -> Iterable[str]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def train_byte_bpe(
    input_paths: Iterable[str | Path],
    out_dir: str | Path,
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
    extra_special_tokens: list[str] | None = None,
) -> AnilaTokenizer:
    resolved_special_tokens = _resolve_special_tokens(special_tokens, extra_special_tokens)
    if vocab_size < len(resolved_special_tokens):
        raise ValueError("vocab_size is smaller than the special-token count")

    tokenizer = Tokenizer(models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=resolved_special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(_text_iterator(input_paths), trainer=trainer)
    wrapped = AnilaTokenizer(tokenizer, special_tokens=resolved_special_tokens)
    wrapped.save(out_dir)
    return wrapped


def _resolve_special_tokens(
    special_tokens: list[str] | None,
    extra_special_tokens: list[str] | None,
) -> list[str]:
    tokens = list(DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens)
    missing = [token for token in DEFAULT_SPECIAL_TOKENS if token not in tokens]
    if missing:
        raise ValueError(f"special_tokens must include required token(s): {', '.join(missing)}")
    if extra_special_tokens:
        tokens.extend(extra_special_tokens)
    return _normalize_special_tokens(tokens)


def _normalize_special_tokens(tokens: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("special tokens must be non-empty strings")
        if token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    return normalized
