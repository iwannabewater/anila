from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from anila._json import dumps_strict_json, loads_strict_json
from anila.chat import ChatMessage, ParsedAssistantMessage, parse_assistant_message, render_chat_prompt
from anila.checkpoint import checkpoint_model_state, load_checkpoint_payload
from anila.config import ModelConfig, SFTConfig
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


@dataclass(frozen=True)
class GeneratedChat:
    prompt: str
    text: str
    assistant: ParsedAssistantMessage
    generation: GeneratedText


@dataclass(frozen=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None
    call: dict[str, Any]
    message: ChatMessage


@dataclass(frozen=True)
class ToolChatStep:
    chat: GeneratedChat
    tool_results: tuple[ToolExecution, ...]


@dataclass(frozen=True)
class GeneratedToolChat:
    messages: tuple[ChatMessage, ...]
    steps: tuple[ToolChatStep, ...]
    assistant: ParsedAssistantMessage
    text: str
    finish_reason: str


def _model_config_from_payload(payload: dict) -> ModelConfig:
    values = payload["model_config"]
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in values.items() if key in allowed}).validated()


def _sft_config_from_payload(payload: dict) -> SFTConfig:
    values = payload.get("sft_config")
    if values is None:
        return SFTConfig().validated()
    if not isinstance(values, dict):
        raise ValueError("checkpoint sft_config must be an object")
    allowed = {field.name for field in fields(SFTConfig)}
    return SFTConfig(**{key: value for key, value in values.items() if key in allowed}).validated()


def _load_model_and_tokenizer(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    device: str,
    use_ema: bool = False,
) -> tuple[AnilaLM, AnilaTokenizer, torch.device]:
    model, tokenizer, runtime_device, _ = _load_model_tokenizer_payload(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
        use_ema=use_ema,
    )
    return model, tokenizer, runtime_device


def _load_model_tokenizer_payload(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    device: str,
    use_ema: bool = False,
) -> tuple[AnilaLM, AnilaTokenizer, torch.device, dict[str, Any]]:
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
    return model, tokenizer, runtime_device, payload


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


def _decode_generated(tokenizer: AnilaTokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(ids, preserve_added_special_tokens=True)


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
    completion = rendered_completion if rendered_completion is not None else _decode_generated(tokenizer, completion_token_ids)
    if return_full_text and rendered_completion is None:
        text = _decode_generated(tokenizer, token_ids)
    elif return_full_text:
        text = f"{_decode_generated(tokenizer, token_ids[:prompt_len])}{completion}"
    else:
        text = completion
    logprobs = _completion_logprobs(model, sequence, prompt_len=prompt_len) if return_logprobs else [None] * len(
        completion_token_ids
    )
    tokens = tuple(
        GeneratedToken(
            id=token_id,
            text=_decode_generated(tokenizer, [token_id]),
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


def _generate_text_with_model(
    *,
    model: AnilaLM,
    tokenizer: AnilaTokenizer,
    runtime_device: torch.device,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    do_sample: bool = True,
    seed: int | None = None,
    return_full_text: bool = True,
    stop_strings: Sequence[str] | None = None,
    return_logprobs: bool = False,
) -> GeneratedText:
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
            completion = _decode_generated(tokenizer, out[0, ids.size(1) :].tolist())
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
        completion = _decode_generated(tokenizer, out[0, ids.size(1) :].tolist())
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
    return _generate_text_with_model(
        model=model,
        tokenizer=tokenizer,
        runtime_device=runtime_device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_sample=do_sample,
        seed=seed,
        return_full_text=return_full_text,
        stop_strings=stop_strings,
        return_logprobs=return_logprobs,
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


def _generate_chat_with_model(
    *,
    model: AnilaLM,
    tokenizer: AnilaTokenizer,
    runtime_device: torch.device,
    messages: list[ChatMessage | dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cfg: SFTConfig,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    do_sample: bool = True,
    seed: int | None = None,
    stop_strings: Sequence[str] | None = None,
    return_logprobs: bool = False,
    open_thinking: bool = False,
) -> GeneratedChat:
    prompt = render_chat_prompt(messages, tools=tools, open_thinking=open_thinking, config=cfg)
    result = _generate_text_with_model(
        model=model,
        tokenizer=tokenizer,
        runtime_device=runtime_device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_sample=do_sample,
        seed=seed,
        return_full_text=False,
        stop_strings=stop_strings,
        return_logprobs=return_logprobs,
    )
    parsed = parse_assistant_message(result.completion, started_thinking=open_thinking, config=cfg)
    return GeneratedChat(prompt=prompt, text=result.completion, assistant=parsed, generation=result)


def generate_chat(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    messages: list[ChatMessage | dict],
    tools: list[dict] | None = None,
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
    stop_strings: Sequence[str] | None = None,
    return_logprobs: bool = False,
    use_ema: bool = False,
    open_thinking: bool = False,
    sft_config: SFTConfig | None = None,
) -> GeneratedChat:
    model, tokenizer, runtime_device, payload = _load_model_tokenizer_payload(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
        use_ema=use_ema,
    )
    cfg = sft_config or _sft_config_from_payload(payload)
    return _generate_chat_with_model(
        model=model,
        tokenizer=tokenizer,
        runtime_device=runtime_device,
        messages=messages,
        tools=tools,
        cfg=cfg,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_sample=do_sample,
        seed=seed,
        stop_strings=stop_strings,
        return_logprobs=return_logprobs,
        open_thinking=open_thinking,
    )


def generate_tool_chat(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    messages: list[ChatMessage | dict],
    tool_handlers: Mapping[str, Callable[[dict[str, Any]], Any]],
    tools: list[dict] | None = None,
    max_tool_rounds: int = 3,
    max_tool_calls_per_round: int = 8,
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
    stop_strings: Sequence[str] | None = None,
    return_logprobs: bool = False,
    use_ema: bool = False,
    open_thinking: bool = False,
    sft_config: SFTConfig | None = None,
) -> GeneratedToolChat:
    if isinstance(max_tool_rounds, bool) or not isinstance(max_tool_rounds, int) or max_tool_rounds < 0:
        raise ValueError("max_tool_rounds must be a non-negative integer")
    if (
        isinstance(max_tool_calls_per_round, bool)
        or not isinstance(max_tool_calls_per_round, int)
        or max_tool_calls_per_round <= 0
    ):
        raise ValueError("max_tool_calls_per_round must be a positive integer")
    model, tokenizer, runtime_device, payload = _load_model_tokenizer_payload(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_path,
        device=device,
        use_ema=use_ema,
    )
    cfg = sft_config or _sft_config_from_payload(payload)
    conversation = [_chat_message_from_value(message, config=cfg) for message in messages]
    steps: list[ToolChatStep] = []
    tool_rounds = 0
    while True:
        chat = _generate_chat_with_model(
            model=model,
            tokenizer=tokenizer,
            runtime_device=runtime_device,
            messages=conversation,
            tools=tools,
            cfg=cfg,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample,
            seed=seed,
            stop_strings=stop_strings,
            return_logprobs=return_logprobs,
            open_thinking=open_thinking,
        )
        conversation.append(
            ChatMessage(
                role=cfg.assistant_role,
                content=chat.assistant.content,
                reasoning_content=chat.assistant.reasoning_content,
                tool_calls=chat.assistant.tool_calls,
            )
        )
        if not chat.assistant.tool_calls:
            steps.append(ToolChatStep(chat=chat, tool_results=()))
            return GeneratedToolChat(
                messages=tuple(conversation),
                steps=tuple(steps),
                assistant=chat.assistant,
                text=chat.assistant.content,
                finish_reason=chat.generation.finish_reason,
            )
        if tool_rounds >= max_tool_rounds:
            steps.append(ToolChatStep(chat=chat, tool_results=()))
            return GeneratedToolChat(
                messages=tuple(conversation),
                steps=tuple(steps),
                assistant=chat.assistant,
                text=chat.assistant.content,
                finish_reason="tool_round_limit",
            )
        if len(chat.assistant.tool_calls) > max_tool_calls_per_round:
            steps.append(ToolChatStep(chat=chat, tool_results=()))
            return GeneratedToolChat(
                messages=tuple(conversation),
                steps=tuple(steps),
                assistant=chat.assistant,
                text=chat.assistant.content,
                finish_reason="tool_call_limit",
            )
        executions = tuple(_execute_tool_call(call, tool_handlers, config=cfg) for call in chat.assistant.tool_calls)
        steps.append(ToolChatStep(chat=chat, tool_results=executions))
        conversation.extend(execution.message for execution in executions)
        tool_rounds += 1


def _chat_message_from_value(value: ChatMessage | dict[str, Any], *, config: SFTConfig | None = None) -> ChatMessage:
    cfg = (config or SFTConfig()).validated()
    if isinstance(value, ChatMessage):
        return value
    if not isinstance(value, dict):
        raise ValueError("chat messages must be ChatMessage instances or dictionaries")
    role = value.get(cfg.role_key)
    if not isinstance(role, str) or not role:
        raise ValueError(f"chat message requires non-empty string field {cfg.role_key!r}")
    content = value.get(cfg.content_key, "")
    if not isinstance(content, str):
        raise ValueError(f"chat message field {cfg.content_key!r} must be a string")
    reasoning_content = value.get(cfg.reasoning_key)
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ValueError(f"chat message field {cfg.reasoning_key!r} must be a string")
    return ChatMessage(
        role=role,
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=_tool_call_tuple(value.get(cfg.tool_calls_key), config=cfg),
    )


def _tool_call_tuple(value: Any, *, config: SFTConfig) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value:
            raise ValueError(f"chat message field {config.tool_calls_key!r} cannot be empty")
        try:
            value = loads_strict_json(value)
        except ValueError as exc:
            raise ValueError(f"chat message field {config.tool_calls_key!r} must contain JSON") from exc
    calls = [value] if isinstance(value, dict) else value
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"chat message field {config.tool_calls_key!r} must be an object or non-empty list")
    if not all(isinstance(call, dict) for call in calls):
        raise ValueError(f"chat message field {config.tool_calls_key!r} must contain only objects")
    return tuple(calls)


def _execute_tool_call(
    call: dict[str, Any],
    handlers: Mapping[str, Callable[[dict[str, Any]], Any]],
    *,
    config: SFTConfig | None = None,
) -> ToolExecution:
    cfg = (config or SFTConfig()).validated()
    name, arguments, argument_error = _tool_call_name_and_arguments(call)
    error = None
    if not name:
        result: Any = {"error": "tool call is missing a name"}
        error = result["error"]
    elif argument_error is not None:
        result = {"error": argument_error}
        error = result["error"]
    elif name not in handlers:
        result = {"error": f"unknown tool: {name}"}
        error = result["error"]
    else:
        try:
            result = handlers[name](arguments)
        except Exception:  # pragma: no cover - exact tool exceptions are caller-defined.
            result = {"error": "tool execution failed"}
            error = result["error"]
    content, response_error = _json_response(result)
    if response_error is not None:
        result = {"error": response_error}
        error = response_error
    return ToolExecution(
        name=name,
        arguments=arguments,
        result=result,
        error=error,
        call=call,
        message=ChatMessage(role=cfg.tool_role, content=content),
    )


def _tool_call_name_and_arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    if not isinstance(call, dict):
        return "", {}, "tool call must be an object"
    if isinstance(call.get("function"), dict):
        function = call["function"]
        raw_name = function.get("name", "")
        raw_arguments = function.get("arguments", {})
    else:
        raw_name = call.get("name", "")
        raw_arguments = call.get("arguments", {})
    name = raw_name if isinstance(raw_name, str) else ""
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = loads_strict_json(raw_arguments) if raw_arguments else {}
        except ValueError:
            return name, {}, "tool call arguments must be valid JSON"
    if not isinstance(raw_arguments, dict):
        return name, {}, "tool call arguments must be a JSON object"
    return name, raw_arguments, None


def _json_response(value: Any) -> tuple[str, str | None]:
    try:
        return dumps_strict_json(value, ensure_ascii=False, sort_keys=True), None
    except (TypeError, ValueError, RecursionError):
        error = "tool result is not JSON serializable"
        return dumps_strict_json({"error": error}, ensure_ascii=False, sort_keys=True), error


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
        completion = _decode_generated(tokenizer, generated_ids)
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
