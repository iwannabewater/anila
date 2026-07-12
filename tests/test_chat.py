import math

import pytest

from anila.chat import ChatMessage, parse_assistant_message, render_chat_prompt


def test_render_chat_prompt_includes_tools_and_open_thinking() -> None:
    prompt = render_chat_prompt(
        [
            {"role": "system", "content": "Answer tersely."},
            {"role": "user", "content": "Use math."},
        ],
        tools=[{"type": "function", "function": {"name": "calculate_math"}}],
        open_thinking=True,
    )

    assert "System: You may call tools" in prompt
    assert "<tools>" in prompt
    assert '"name": "calculate_math"' in prompt
    assert "System: Answer tersely." in prompt
    assert "User: Use math." in prompt
    assert prompt.endswith("Assistant: <think>\n")


def test_render_chat_prompt_supports_assistant_tool_calls_and_tool_responses() -> None:
    prompt = render_chat_prompt(
        [
            ChatMessage(role="user", content="Use math."),
            ChatMessage(
                role="assistant",
                reasoning_content="Need a calculator.",
                tool_calls=({"name": "calculate_math", "arguments": {"expression": "2+2"}},),
            ),
            ChatMessage(role="tool", content='{"result":"4"}'),
            ChatMessage(role="assistant", content="The answer is 4."),
        ],
        add_generation_prompt=False,
    )

    assert "<think>\nNeed a calculator.\n</think>" in prompt
    assert "<tool_call>" in prompt
    assert "<tool_response>\n{\"result\":\"4\"}\n</tool_response>" in prompt
    assert prompt.endswith("Assistant: The answer is 4.\n")


def test_parse_assistant_message_extracts_reasoning_and_tool_calls() -> None:
    parsed = parse_assistant_message(
        "<think>\nNeed a calculator.\n</think>\n\n"
        "<tool_call>\n{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"2+2\"}}\n</tool_call>"
    )

    assert parsed.content == ""
    assert parsed.reasoning_content == "Need a calculator."
    assert parsed.tool_calls == ({"name": "calculate_math", "arguments": {"expression": "2+2"}},)
    assert parsed.invalid_tool_calls == ()


def test_parse_assistant_message_handles_prompt_started_thinking() -> None:
    parsed = parse_assistant_message("Need a calculator.\n</think>\n\nThe answer is 4.", started_thinking=True)

    assert parsed.reasoning_content == "Need a calculator."
    assert parsed.content == "The answer is 4."
    assert parsed.raw == "Need a calculator.\n</think>\n\nThe answer is 4."


def test_parse_assistant_message_keeps_invalid_tool_calls_separate() -> None:
    parsed = parse_assistant_message("Answer.\n<tool_call>\nnot json\n</tool_call>")

    assert parsed.content == "Answer."
    assert parsed.tool_calls == ()
    assert parsed.invalid_tool_calls == ("not json",)


def test_parse_assistant_message_rejects_non_finite_tool_call_json() -> None:
    parsed = parse_assistant_message('Answer.\n<tool_call>\n{"name":"calculate","arguments":{"value":NaN}}\n</tool_call>')

    assert parsed.content == "Answer."
    assert parsed.tool_calls == ()
    assert parsed.invalid_tool_calls == ('{"name":"calculate","arguments":{"value":NaN}}',)


def test_render_chat_prompt_rejects_non_finite_tool_specs() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        render_chat_prompt(
            [{"role": "user", "content": "Use a tool."}],
            tools=[{"type": "function", "function": {"name": "nan", "threshold": math.nan}}],
        )


def test_render_chat_prompt_rejects_non_finite_assistant_tool_calls() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        render_chat_prompt(
            [
                ChatMessage(role="user", content="Use a tool."),
                ChatMessage(role="assistant", tool_calls=({"name": "nan", "arguments": {"value": math.nan}},)),
            ],
            add_generation_prompt=False,
        )


def test_render_chat_prompt_rejects_empty_user_message() -> None:
    with pytest.raises(ValueError, match="non-empty content"):
        render_chat_prompt([{"role": "user", "content": ""}])
