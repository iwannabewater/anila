from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch

from anila.checkpoint import load_checkpoint_payload
from anila.config import ModelConfig
from anila.model import AnilaLM
from anila.peft import apply_lora
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
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    device: str = "auto",
    do_sample: bool = True,
    seed: int | None = None,
    return_full_text: bool = True,
) -> str:
    runtime_device = resolve_device(device)
    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config"))
    tokenizer = AnilaTokenizer.load(tokenizer_path)
    model = AnilaLM(_model_config_from_payload(payload))
    lora_config = payload.get("lora_config")
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        from anila.config import LoRAConfig

        apply_lora(model, LoRAConfig(**lora_config).validated())
    model.load_state_dict(payload["model"])
    model.to(runtime_device).eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=runtime_device)
        generator.manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=runtime_device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        length_penalty=length_penalty,
        eos_id=tokenizer.eos_id,
        do_sample=do_sample,
        generator=generator,
    )
    generated_ids = out[0].tolist()
    if return_full_text:
        return tokenizer.decode(generated_ids)
    prompt_len = ids.size(1)
    return tokenizer.decode(generated_ids[prompt_len:])
