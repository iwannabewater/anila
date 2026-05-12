from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch

from anila.config import ModelConfig
from anila.model import AnilaLM
from anila.tokenization import AnilaTokenizer
from anila.training import resolve_device


def _model_config_from_payload(payload: dict) -> ModelConfig:
    values = payload["model_config"]
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed}).validated()


def sample_text(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float = 1.0,
    device: str = "auto",
) -> str:
    runtime_device = resolve_device(device)
    payload = torch.load(checkpoint, map_location="cpu")
    tokenizer = AnilaTokenizer.load(tokenizer_path)
    model = AnilaLM(_model_config_from_payload(payload))
    model.load_state_dict(payload["model"])
    model.to(runtime_device).eval()
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=runtime_device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(out[0].tolist())
