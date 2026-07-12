from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from anila._json import dumps_strict_json, loads_strict_json
from anila.config import SFTConfig

TOOL_BLOCK_START = "<tools>"
TOOL_BLOCK_END = "</tools>"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ParsedAssistantMessage:
    content: str
    reasoning_content: str | None
    tool_calls: tuple[dict[str, Any], ...]
    invalid_tool_calls: tuple[str, ...]
    raw: str


def render_chat_prompt(
    messages: list[ChatMessage | dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    open_thinking: bool = False,
    config: SFTConfig | None = None,
) -> str:
    """Render chat messages using Anila's native SFT tags."""

    cfg = (config or SFTConfig()).validated()
    normalized = [_message_from_value(message, config=cfg) for message in messages]
    if not normalized:
        raise ValueError("messages must contain at least one chat message")

    chunks: list[str] = []
    tool_preamble = _render_tool_preamble(tools, cfg)
    if tool_preamble:
        chunks.append(f"{cfg.system_prefix} {tool_preamble}\n")
    for message in normalized:
        chunks.append(_render_message(message, cfg))
    if add_generation_prompt:
        chunks.append(f"{cfg.assistant_prefix} ")
        if open_thinking:
            chunks.append(f"{cfg.thinking_start}\n")
    return "".join(chunks)


def parse_assistant_message(
    text: str,
    *,
    config: SFTConfig | None = None,
    started_thinking: bool = False,
) -> ParsedAssistantMessage:
    """Parse Anila reasoning and tool-call tags from generated assistant text."""

    cfg = (config or SFTConfig()).validated()
    reasoning_content = None
    cleaned = text
    if started_thinking and cfg.thinking_start not in cleaned and cfg.thinking_end in cleaned:
        cleaned = f"{cfg.thinking_start}\n{cleaned}"
    reasoning_match = re.search(
        rf"{re.escape(cfg.thinking_start)}\s*(.*?)\s*{re.escape(cfg.thinking_end)}",
        cleaned,
        flags=re.DOTALL,
    )
    if reasoning_match is not None:
        reasoning_content = reasoning_match.group(1)
        cleaned = cleaned[: reasoning_match.start()] + cleaned[reasoning_match.end() :]

    tool_calls: list[dict[str, Any]] = []
    invalid_tool_calls: list[str] = []
    for match in re.finditer(
        rf"{re.escape(cfg.tool_call_start)}\s*(.*?)\s*{re.escape(cfg.tool_call_end)}",
        cleaned,
        flags=re.DOTALL,
    ):
        raw_payload = match.group(1).strip()
        try:
            parsed = loads_strict_json(raw_payload)
        except ValueError:
            invalid_tool_calls.append(raw_payload)
            continue
        calls = [parsed] if isinstance(parsed, dict) else parsed
        if isinstance(calls, list) and all(isinstance(item, dict) for item in calls):
            tool_calls.extend(calls)
        else:
            invalid_tool_calls.append(raw_payload)
    cleaned = re.sub(
        rf"{re.escape(cfg.tool_call_start)}.*?{re.escape(cfg.tool_call_end)}",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    return ParsedAssistantMessage(
        content=cleaned.strip(),
        reasoning_content=reasoning_content,
        tool_calls=tuple(tool_calls),
        invalid_tool_calls=tuple(invalid_tool_calls),
        raw=text,
    )


def _message_from_value(value: ChatMessage | dict[str, Any], *, config: SFTConfig) -> ChatMessage:
    if isinstance(value, ChatMessage):
        return value
    if not isinstance(value, dict):
        raise ValueError("chat messages must be ChatMessage instances or dictionaries")
    role = value.get(config.role_key)
    if not isinstance(role, str) or not role:
        raise ValueError(f"chat message requires non-empty string field {config.role_key!r}")
    content = value.get(config.content_key, "")
    if not isinstance(content, str):
        raise ValueError(f"chat message field {config.content_key!r} must be a string")
    reasoning = value.get(config.reasoning_key)
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError(f"chat message field {config.reasoning_key!r} must be a string")
    return ChatMessage(
        role=role,
        content=content,
        reasoning_content=reasoning,
        tool_calls=tuple(_tool_call_dicts(value.get(config.tool_calls_key), config=config)),
    )


def _render_tool_preamble(tools: list[dict[str, Any]] | None, cfg: SFTConfig) -> str | None:
    if not tools:
        return None
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tool spec {index} must be an object")
    rendered_tool_chunks = []
    for index, tool in enumerate(tools):
        try:
            rendered_tool_chunks.append(dumps_strict_json(tool, ensure_ascii=False, sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tool spec {index} must be JSON serializable with finite numbers") from exc
    rendered_tools = "\n".join(rendered_tool_chunks)
    return (
        "You may call tools by returning JSON inside "
        f"{cfg.tool_call_start} and {cfg.tool_call_end} tags.\n"
        f"{TOOL_BLOCK_START}\n{rendered_tools}\n{TOOL_BLOCK_END}"
    )


def _render_message(message: ChatMessage, cfg: SFTConfig) -> str:
    if message.role == cfg.system_role:
        _require_content(message)
        return f"{cfg.system_prefix} {message.content}\n"
    if message.role == cfg.user_role:
        _require_content(message)
        return f"{cfg.user_prefix} {message.content}\n"
    if message.role == cfg.tool_role:
        _require_content(message)
        return f"{cfg.tool_prefix} {cfg.tool_response_start}\n{message.content}\n{cfg.tool_response_end}\n"
    if message.role == cfg.assistant_role:
        return _render_assistant_message(message, cfg)
    raise ValueError(f"unsupported chat role {message.role!r}")


def _render_assistant_message(message: ChatMessage, cfg: SFTConfig) -> str:
    if not message.content and message.reasoning_content is None and not message.tool_calls:
        raise ValueError("assistant messages require content, reasoning_content, or tool_calls")
    chunks = [f"{cfg.assistant_prefix} "]
    if message.reasoning_content is not None:
        chunks.append(f"{cfg.thinking_start}\n{message.reasoning_content}\n{cfg.thinking_end}\n\n")
    if message.content:
        chunks.append(message.content)
    if message.tool_calls:
        if message.content and not message.content.endswith("\n"):
            chunks.append("\n")
        for index, call in enumerate(message.tool_calls):
            if index > 0:
                chunks.append("\n")
            try:
                payload = dumps_strict_json(call, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"assistant tool call {index} must be JSON serializable with finite numbers") from exc
            chunks.append(f"{cfg.tool_call_start}\n{payload}\n{cfg.tool_call_end}")
    chunks.append("\n")
    return "".join(chunks)


def _require_content(message: ChatMessage) -> None:
    if not message.content:
        raise ValueError(f"{message.role} messages require non-empty content")


def _tool_call_dicts(value: Any, *, config: SFTConfig) -> list[dict[str, Any]]:
    if value is None:
        return []
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
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"chat tool call {index} must be an object")
    return calls
