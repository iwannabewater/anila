from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from anila.checkpoint import checkpoint_model_state, load_checkpoint_payload
from anila.config import ModelConfig
from anila.model import AnilaLM
from anila.peft import apply_lora
from anila.tokenization import AnilaTokenizer
from anila.training import resolve_device


@dataclass(frozen=True)
class GeneratedToken:
    """One generated token decoded for inspection."""

    id: int
    text: str
    logprob: float | None = None


@dataclass(frozen=True)
class GeneratedText:
    """Structured text generation result.

    `completion` excludes a matched text stop string. Token fields describe the raw generated tokens that triggered the
    result, which can include the token carrying the stop text for text-level stops.
    """

    text: str
    prompt: str
    completion: str
    finish_reason: str
    stop: str | None
    token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    tokens: tuple[GeneratedToken, ...]


def _model_config_from_payload(payload: dict) -> ModelConfig:
    values = payload["model_config"]
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed}).validated()


def _load_model_and_tokenizer(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    device: str,
    use_ema: bool = False,
) -> tuple[AnilaLM, AnilaTokenizer, torch.device]:
    runtime_device = resolve_device(device)
    payload = load_checkpoint_payload(checkpoint, required_keys=("model", "model_config"))
    tokenizer = AnilaTokenizer.load(tokenizer_path)
    model = AnilaLM(_model_config_from_payload(payload))
    lora_config = payload.get("lora_config")
    if isinstance(lora_config, dict) and lora_config.get("enabled", False):
        from anila.config import LoRAConfig

        apply_lora(model, LoRAConfig(**lora_config).validated())
    model.load_state_dict(checkpoint_model_state(payload, use_ema=use_ema))
    model.to(runtime_device).eval()
    return model, tokenizer, runtime_device


def _generator_for_seed(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _first_stop(text: str, stop_strings: Sequence[str]) -> tuple[int, str] | None:
    matches = [(idx, stop) for stop in stop_strings if stop and (idx := text.find(stop)) >= 0]
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])


def _stop_prefix_holdback(text: str, stop_strings: Sequence[str]) -> int:
    holdback = 0
    for stop in stop_strings:
        for size in range(1, min(len(stop), len(text)) + 1):
            if text.endswith(stop[:size]):
                holdback = max(holdback, size)
    return holdback


def _completion_logprobs(model: AnilaLM, sequence: torch.Tensor, *, prompt_len: int) -> list[float]:
    logprobs: list[float] = []
    for index in range(prompt_len, sequence.size(1)):
        start = max(0, index - model.config.context_length)
        context = sequence[:, start:index]
        target = sequence[:, index]
        with torch.inference_mode():
            logits = model(context).logits[:, -1, :]
            logprob = torch.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1))
        logprobs.append(float(logprob.item()))
    return logprobs


def _build_generated_text(
    *,
    model: AnilaLM,
    tokenizer: AnilaTokenizer,
    sequence: torch.Tensor,
    prompt: str,
    prompt_len: int,
    return_full_text: bool,
    finish_reason: str,
    stop: str | None,
    return_logprobs: bool,
    rendered_completion: str | None = None,
) -> GeneratedText:
    token_ids = tuple(int(token_id) for token_id in sequence[0].tolist())
    completion_token_ids = token_ids[prompt_len:]
    completion = rendered_completion if rendered_completion is not None else tokenizer.decode(completion_token_ids)
    if return_full_text and rendered_completion is None:
        text = tokenizer.decode(token_ids)
    elif return_full_text:
        text = f"{tokenizer.decode(token_ids[:prompt_len])}{completion}"
    else:
        text = completion
    logprobs = _completion_logprobs(model, sequence, prompt_len=prompt_len) if return_logprobs else [None] * len(
        completion_token_ids
    )
    tokens = tuple(
        GeneratedToken(
            id=token_id,
            text=tokenizer.decode([token_id]),
            logprob=logprobs[index],
        )
        for index, token_id in enumerate(completion_token_ids)
    )
    return GeneratedText(
        text=text,
        prompt=prompt,
        completion=completion,
        finish_reason=finish_reason,
        stop=stop,
        token_ids=token_ids,
        completion_token_ids=completion_token_ids,
        tokens=tokens,
    )


def generate_text(
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
    stop_strings: Sequence[str] | None = None,
    return_logprobs: bool = False,
    use_ema: bool = False,
) -> GeneratedText:
    model, tokenizer, runtime_device = _load_model_and_tokenizer(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
        use_ema=use_ema,
    )
    generator = _generator_for_seed(seed, runtime_device)
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=runtime_device)
    stop_strings = tuple(stop_strings or ())
    finish_reason = "length"
    matched_stop: str | None = None
    rendered_completion: str | None = None
    if num_beams == 1:
        out = ids
        for step in model.generate_steps(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            eos_id=tokenizer.eos_id,
            do_sample=do_sample,
            generator=generator,
        ):
            out = step.sequences
            completion = tokenizer.decode(out[0, ids.size(1) :].tolist())
            stop_match = _first_stop(completion, stop_strings)
            if stop_match is not None:
                stop_index, matched_stop = stop_match
                finish_reason = "stop"
                rendered_completion = completion[:stop_index]
                break
            if bool(step.finished[0].item()):
                finish_reason = "eos"
                break
    else:
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
        completion = tokenizer.decode(out[0, ids.size(1) :].tolist())
        stop_match = _first_stop(completion, stop_strings)
        if stop_match is not None:
            stop_index, matched_stop = stop_match
            finish_reason = "stop"
            rendered_completion = completion[:stop_index]
        elif out[0, ids.size(1) :].numel() > 0 and int(out[0, -1].item()) == tokenizer.eos_id:
            finish_reason = "eos"
    return _build_generated_text(
        model=model,
        tokenizer=tokenizer,
        sequence=out,
        prompt=prompt,
        prompt_len=ids.size(1),
        return_full_text=return_full_text,
        finish_reason=finish_reason,
        stop=matched_stop,
        return_logprobs=return_logprobs,
        rendered_completion=rendered_completion,
    )


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
    stop_strings: Sequence[str] | None = None,
    use_ema: bool = False,
) -> str:
    return generate_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        length_penalty=length_penalty,
        device=device,
        do_sample=do_sample,
        seed=seed,
        return_full_text=return_full_text,
        stop_strings=stop_strings,
        use_ema=use_ema,
    ).text


def stream_text(
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
    device: str = "auto",
    do_sample: bool = True,
    seed: int | None = None,
    stop_strings: Sequence[str] | None = None,
    use_ema: bool = False,
) -> Iterator[str]:
    model, tokenizer, runtime_device = _load_model_and_tokenizer(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
        use_ema=use_ema,
    )
    generator = _generator_for_seed(seed, runtime_device)
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=runtime_device)
    generated_ids: list[int] = []
    emitted = ""
    last_visible = ""
    stop_strings = tuple(stop_strings or ())
    for step in model.generate_steps(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        eos_id=tokenizer.eos_id,
        do_sample=do_sample,
        generator=generator,
    ):
        generated_ids.append(int(step.token_ids[0].item()))
        completion = tokenizer.decode(generated_ids)
        stop_match = _first_stop(completion, stop_strings)
        visible = completion
        if stop_match is not None:
            stop_index, _ = stop_match
            visible = completion[:stop_index]
        elif not bool(step.finished[0].item()):
            holdback = _stop_prefix_holdback(visible, stop_strings)
            if holdback:
                visible = visible[:-holdback]
        last_visible = visible if stop_match is not None else completion
        delta = visible[len(emitted) :]
        if delta:
            yield delta
            emitted = visible
        if stop_match is not None or bool(step.finished[0].item()):
            break
    delta = last_visible[len(emitted) :]
    if delta:
        yield delta
