from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from anila.tokenization import AnilaTokenizer


class TextTokenDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: AnilaTokenizer, context_length: int):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
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


def create_dataloader(
    path: str | Path,
    tokenizer: AnilaTokenizer,
    *,
    context_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TextTokenDataset(path, tokenizer, context_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
