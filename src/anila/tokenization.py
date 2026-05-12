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


class AnilaTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
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

    def decode(self, ids: Iterable[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def save(self, out_dir: str | Path) -> None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(output / "tokenizer.json"))
        metadata = {
            "type": "byte_bpe",
            "special_tokens": DEFAULT_SPECIAL_TOKENS,
            "vocab_size": self.vocab_size,
        }
        (output / "tokenizer_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AnilaTokenizer:
        root = Path(path)
        tokenizer_path = root / "tokenizer.json" if root.is_dir() else root
        return cls(Tokenizer.from_file(str(tokenizer_path)))


def _text_iterator(paths: Iterable[str | Path]) -> Iterable[str]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
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
) -> AnilaTokenizer:
    if vocab_size < len(DEFAULT_SPECIAL_TOKENS):
        raise ValueError("vocab_size is smaller than the required special-token count")

    special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
    tokenizer = Tokenizer(models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(_text_iterator(input_paths), trainer=trainer)
    wrapped = AnilaTokenizer(tokenizer)
    wrapped.save(out_dir)
    return wrapped
