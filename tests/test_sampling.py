import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

import anila.sampling as sampling_module
from anila.cli import app
from anila.config import LoRAConfig, ModelConfig, SFTConfig
from anila.model import AnilaLM, GenerationStep
from anila.sampling import generate_chat, generate_text, generate_tool_chat, sample_text, stream_text
from anila.tokenization import DEFAULT_CHAT_SPECIAL_TOKENS, AnilaTokenizer, train_byte_bpe


def _tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Anila trains small language models.\n"
        "Beam search keeps multiple deterministic continuations alive.\n"
        "Generation should stay simple and inspectable.\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe(
        [corpus],
        tokenizer_dir,
        vocab_size=300,
        min_frequency=1,
        extra_special_tokens=DEFAULT_CHAT_SPECIAL_TOKENS,
    )
    return tokenizer_dir


def _checkpoint(
    tmp_path: Path,
    cfg: ModelConfig,
    *,
    include_ema: bool = False,
    sft_config: SFTConfig | None = None,
    model_config_payload: dict[str, object] | None = None,
) -> Path:
    checkpoint = tmp_path / "checkpoint.pt"
    model = AnilaLM(cfg)
    payload = {
        "schema_version": 1,
        "objective": "pretrain",
        "model": model.state_dict(),
        "model_config": asdict(cfg) if model_config_payload is None else model_config_payload,
        "lora_config": asdict(LoRAConfig()),
        "tokenizer_path": "tokenizer",
        "step": 0,
    }
    if sft_config is not None:
        payload["sft_config"] = asdict(sft_config)
    if include_ema:
        payload["ema_model"] = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        payload["ema_decay"] = 0.99
    torch.save(payload, checkpoint)
    return checkpoint


def test_sample_text_supports_beam_search(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)

    text = sample_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        prompt="Anila",
        max_new_tokens=2,
        num_beams=2,
        length_penalty=0.7,
        device="cpu",
    )

    assert isinstance(text, str)


def test_sample_text_can_use_ema_weights(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg, include_ema=True)

    text = sample_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        prompt="Anila",
        max_new_tokens=1,
        device="cpu",
        use_ema=True,
    )

    assert isinstance(text, str)


def test_sample_text_requires_ema_weights_when_requested(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)

    with pytest.raises(ValueError, match="EMA"):
        sample_text(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_dir,
            prompt="Anila",
            max_new_tokens=1,
            device="cpu",
            use_ema=True,
        )


def test_generate_text_returns_metadata_and_logprobs(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)

    result = generate_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        prompt="Anila",
        max_new_tokens=2,
        top_k=None,
        device="cpu",
        do_sample=False,
        return_full_text=False,
        return_logprobs=True,
    )

    assert result.text == result.completion
    assert result.finish_reason in {"length", "eos"}
    assert len(result.completion_token_ids) == len(result.tokens)
    assert all(token.logprob is not None for token in result.tokens)


def test_generate_text_loads_checkpoint_without_new_model_config_fields(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    legacy_model_config = {
        key: getattr(cfg, key)
        for key in (
            "vocab_size",
            "context_length",
            "n_layer",
            "n_head",
            "n_kv_head",
            "n_embd",
            "dropout",
            "rope_base",
            "bias",
            "tie_embeddings",
        )
    }
    checkpoint = _checkpoint(tmp_path, cfg, model_config_payload=legacy_model_config)

    result = generate_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        prompt="Anila",
        max_new_tokens=1,
        top_k=None,
        device="cpu",
        do_sample=False,
    )

    assert isinstance(result.text, str)


def test_stop_strings_trim_generation_and_stream_chunks(tmp_path: Path, monkeypatch) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    tokenizer = AnilaTokenizer.load(tokenizer_dir)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)
    planned_ids = tokenizer.encode(" stop after", add_bos=False)

    def generate_steps(self, input_ids: torch.Tensor, **_kwargs):
        sequences = input_ids
        for token_id in planned_ids:
            next_id = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
            sequences = torch.cat([sequences, next_id], dim=1)
            yield GenerationStep(
                token_ids=next_id,
                token_logprobs=torch.zeros((1, 1), device=input_ids.device),
                sequences=sequences,
                finished=torch.zeros((1,), dtype=torch.bool, device=input_ids.device),
            )

    monkeypatch.setattr(AnilaLM, "generate_steps", generate_steps)

    result = generate_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        prompt="Anila",
        max_new_tokens=10,
        stop_strings=["after"],
        device="cpu",
        do_sample=False,
        return_full_text=False,
    )
    chunks = list(
        stream_text(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_dir,
            prompt="Anila",
            max_new_tokens=10,
            stop_strings=["after"],
            device="cpu",
            do_sample=False,
        )
    )

    assert result.finish_reason == "stop"
    assert result.stop == "after"
    assert result.completion == " stop "
    assert "".join(chunks) == " stop "

    planned_ids = tokenizer.encode(" aft", add_bos=False)
    chunks = list(
        stream_text(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer_dir,
            prompt="Anila",
            max_new_tokens=10,
            stop_strings=["after"],
            device="cpu",
            do_sample=False,
        )
    )

    assert "".join(chunks) == " aft"


def test_generate_cli_accepts_beam_search_flags(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg, include_ema=True)

    result = CliRunner().invoke(
        app,
        [
            "model",
            "generate",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--prompt",
            "Anila",
            "--max-new-tokens",
            "2",
            "--num-beams",
            "2",
            "--length-penalty",
            "0.7",
            "--ema",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0


def test_generate_cli_can_print_json_with_logprobs(tmp_path: Path) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    cfg = ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)

    result = CliRunner().invoke(
        app,
        [
            "model",
            "generate",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--prompt",
            "Anila",
            "--max-new-tokens",
            "1",
            "--top-k",
            "0",
            "--greedy",
            "--device",
            "cpu",
            "--json",
            "--logprobs",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["finish_reason"] in {"length", "eos"}
    assert len(payload["tokens"]) == 1
    assert payload["tokens"][0]["logprob"] is not None


def test_generate_chat_parses_reasoning_and_tool_calls(tmp_path: Path, monkeypatch) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    tokenizer = AnilaTokenizer.load(tokenizer_dir)
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)
    planned_ids = tokenizer.encode(
        "<think>\nNeed math.\n</think>\n\n"
        "<tool_call>\n{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"2+2\"}}\n</tool_call>",
        add_bos=False,
    )

    def generate_steps(self, input_ids: torch.Tensor, **_kwargs):
        sequences = input_ids
        for token_id in planned_ids:
            next_id = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
            sequences = torch.cat([sequences, next_id], dim=1)
            yield GenerationStep(
                token_ids=next_id,
                token_logprobs=torch.zeros((1, 1), device=input_ids.device),
                sequences=sequences,
                finished=torch.zeros((1,), dtype=torch.bool, device=input_ids.device),
            )

    monkeypatch.setattr(AnilaLM, "generate_steps", generate_steps)

    result = generate_chat(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[{"type": "function", "function": {"name": "calculate_math"}}],
        max_new_tokens=len(planned_ids),
        device="cpu",
        do_sample=False,
    )

    assert "<tools>" in result.prompt
    assert result.assistant.reasoning_content == "Need math."
    assert result.assistant.tool_calls == ({"name": "calculate_math", "arguments": {"expression": "2+2"}},)
    assert result.assistant.content == ""


def test_generate_chat_parses_open_thinking_completion(tmp_path: Path, monkeypatch) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    tokenizer = AnilaTokenizer.load(tokenizer_dir)
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)
    planned_ids = tokenizer.encode("Need math.\n</think>\n\nThe answer is 4.", add_bos=False)

    def generate_steps(self, input_ids: torch.Tensor, **_kwargs):
        sequences = input_ids
        for token_id in planned_ids:
            next_id = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
            sequences = torch.cat([sequences, next_id], dim=1)
            yield GenerationStep(
                token_ids=next_id,
                token_logprobs=torch.zeros((1, 1), device=input_ids.device),
                sequences=sequences,
                finished=torch.zeros((1,), dtype=torch.bool, device=input_ids.device),
            )

    monkeypatch.setattr(AnilaLM, "generate_steps", generate_steps)

    result = generate_chat(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_new_tokens=len(planned_ids),
        device="cpu",
        do_sample=False,
        open_thinking=True,
    )

    assert result.prompt.endswith("Assistant: <think>\n")
    assert result.assistant.reasoning_content == "Need math."
    assert result.assistant.content == "The answer is 4."


def test_generate_chat_uses_checkpoint_sft_config(tmp_path: Path, monkeypatch) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    tokenizer = AnilaTokenizer.load(tokenizer_dir)
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    sft_cfg = SFTConfig(
        user_prefix="Human:",
        assistant_prefix="Bot:",
        thinking_start="<reason>",
        thinking_end="</reason>",
        tool_call_start="<call>",
        tool_call_end="</call>",
    ).validated()
    checkpoint = _checkpoint(tmp_path, cfg, sft_config=sft_cfg)
    planned_ids = tokenizer.encode("<reason>\nCustom path.\n</reason>\n\nDone.", add_bos=False)

    def generate_steps(self, input_ids: torch.Tensor, **_kwargs):
        sequences = input_ids
        for token_id in planned_ids:
            next_id = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
            sequences = torch.cat([sequences, next_id], dim=1)
            yield GenerationStep(
                token_ids=next_id,
                token_logprobs=torch.zeros((1, 1), device=input_ids.device),
                sequences=sequences,
                finished=torch.zeros((1,), dtype=torch.bool, device=input_ids.device),
            )

    monkeypatch.setattr(AnilaLM, "generate_steps", generate_steps)

    result = generate_chat(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        messages=[{"role": "user", "content": "Use custom chat tags."}],
        max_new_tokens=len(planned_ids),
        device="cpu",
        do_sample=False,
    )

    assert "Human: Use custom chat tags." in result.prompt
    assert result.prompt.endswith("Bot: ")
    assert result.assistant.reasoning_content == "Custom path."
    assert result.assistant.content == "Done."


def _fake_generated_chat(completion: str, *, finish_reason: str = "length") -> sampling_module.GeneratedChat:
    assistant = sampling_module.parse_assistant_message(completion)
    generation = sampling_module.GeneratedText(
        text=completion,
        prompt="",
        completion=completion,
        finish_reason=finish_reason,
        stop=None,
        token_ids=(),
        completion_token_ids=(),
        tokens=(),
    )
    return sampling_module.GeneratedChat(prompt="", text=completion, assistant=assistant, generation=generation)


def test_generate_tool_chat_executes_tools_and_continues(monkeypatch) -> None:
    load_calls = 0
    prompts: list[tuple[object, ...]] = []

    def load_once(**_kwargs):
        nonlocal load_calls
        load_calls += 1
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    completions = [
        '<tool_call>{"function":{"name":"calculate_math","arguments":"{\\"expression\\":\\"2+2\\"}"}}</tool_call>',
        "The answer is 4.",
    ]

    def fake_generate(**kwargs):
        prompts.append(tuple(kwargs["messages"]))
        return _fake_generated_chat(completions[len(prompts) - 1], finish_reason="stop")

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[{"type": "function", "function": {"name": "calculate_math"}}],
        tool_handlers={"calculate_math": lambda args: {"result": args["expression"].replace("2+2", "4")}},
        max_tool_rounds=1,
        device="cpu",
    )

    assert load_calls == 1
    assert len(prompts) == 2
    assert [message.role for message in result.messages] == ["user", "assistant", "tool", "assistant"]
    assert result.steps[0].tool_results[0].name == "calculate_math"
    assert result.steps[0].tool_results[0].arguments == {"expression": "2+2"}
    assert result.steps[0].tool_results[0].message.content == '{"result": "4"}'
    assert result.assistant.content == "The answer is 4."
    assert result.finish_reason == "stop"


def test_generate_tool_chat_rejects_malformed_tool_arguments(monkeypatch) -> None:
    load_calls = 0
    handler_calls = 0

    def load_once(**_kwargs):
        nonlocal load_calls
        load_calls += 1
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    completions = [
        '<tool_call>{"name":"calculate_math","arguments":"{not json"}</tool_call>',
        "I could not call the tool.",
    ]

    def fake_generate(**kwargs):
        return _fake_generated_chat(completions[len(kwargs["messages"]) // 2])

    def handler(_args):
        nonlocal handler_calls
        handler_calls += 1
        return {"result": "should not run"}

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "Use a tool."}],
        tool_handlers={"calculate_math": handler},
        max_tool_rounds=1,
        device="cpu",
    )

    tool_result = result.steps[0].tool_results[0]
    assert load_calls == 1
    assert handler_calls == 0
    assert tool_result.error == "tool call arguments must be valid JSON"
    assert tool_result.arguments == {}
    assert tool_result.message.content == '{"error": "tool call arguments must be valid JSON"}'
    assert result.assistant.content == "I could not call the tool."


def test_generate_tool_chat_rejects_non_finite_tool_argument_json(monkeypatch) -> None:
    handler_calls = 0

    def load_once(**_kwargs):
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    completions = [
        '<tool_call>{"name":"calculate_math","arguments":"{\\"value\\":NaN}"}</tool_call>',
        "I could not call the tool.",
    ]

    def fake_generate(**kwargs):
        return _fake_generated_chat(completions[len(kwargs["messages"]) // 2])

    def handler(_args):
        nonlocal handler_calls
        handler_calls += 1
        return {"result": "should not run"}

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "Use a tool."}],
        tool_handlers={"calculate_math": handler},
        max_tool_rounds=1,
        device="cpu",
    )

    tool_result = result.steps[0].tool_results[0]
    assert handler_calls == 0
    assert tool_result.error == "tool call arguments must be valid JSON"
    assert tool_result.message.content == '{"error": "tool call arguments must be valid JSON"}'


def test_generate_tool_chat_bounds_tool_result_serialization_failures(monkeypatch) -> None:
    def load_once(**_kwargs):
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    completions = [
        '<tool_call>{"name":"cyclic","arguments":{}}</tool_call>',
        "Handled.",
    ]

    def fake_generate(**kwargs):
        return _fake_generated_chat(completions[len(kwargs["messages"]) // 2])

    def handler(_args):
        value: dict[str, object] = {}
        value["self"] = value
        return value

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "Use a cyclic tool."}],
        tool_handlers={"cyclic": handler},
        max_tool_rounds=1,
        device="cpu",
    )

    tool_result = result.steps[0].tool_results[0]
    assert tool_result.error == "tool result is not JSON serializable"
    assert tool_result.result == {"error": "tool result is not JSON serializable"}
    assert tool_result.message.content == '{"error": "tool result is not JSON serializable"}'
    assert result.assistant.content == "Handled."


def test_generate_tool_chat_rejects_non_finite_tool_results(monkeypatch) -> None:
    def load_once(**_kwargs):
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    completions = [
        '<tool_call>{"name":"nan","arguments":{}}</tool_call>',
        "Handled.",
    ]

    def fake_generate(**kwargs):
        return _fake_generated_chat(completions[len(kwargs["messages"]) // 2])

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "Use a non-finite tool."}],
        tool_handlers={"nan": lambda _args: {"value": float("nan")}},
        max_tool_rounds=1,
        device="cpu",
    )

    tool_result = result.steps[0].tool_results[0]
    assert tool_result.error == "tool result is not JSON serializable"
    assert tool_result.message.content == '{"error": "tool result is not JSON serializable"}'


def test_generate_tool_chat_redacts_tool_exception_messages(monkeypatch) -> None:
    def load_once(**_kwargs):
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    completions = [
        '<tool_call>{"name":"fail","arguments":{}}</tool_call>',
        "Handled.",
    ]

    def fake_generate(**kwargs):
        return _fake_generated_chat(completions[len(kwargs["messages"]) // 2])

    def handler(_args):
        raise RuntimeError("token sk-live-secret leaked")

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "Use a failing tool."}],
        tool_handlers={"fail": handler},
        max_tool_rounds=1,
        device="cpu",
    )

    tool_result = result.steps[0].tool_results[0]
    assert tool_result.error == "tool execution failed"
    assert tool_result.result == {"error": "tool execution failed"}
    assert tool_result.message.content == '{"error": "tool execution failed"}'
    assert "sk-live-secret" not in tool_result.message.content


def test_generate_tool_chat_stops_at_tool_round_limit(monkeypatch) -> None:
    def load_once(**_kwargs):
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    def fake_generate(**_kwargs):
        return _fake_generated_chat('<tool_call>{"name":"unknown","arguments":{}}</tool_call>')

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "Use a tool."}],
        tool_handlers={},
        max_tool_rounds=0,
        device="cpu",
    )

    assert result.finish_reason == "tool_round_limit"
    assert result.steps[0].tool_results == ()
    assert [message.role for message in result.messages] == ["user", "assistant"]


@pytest.mark.parametrize("max_tool_rounds", [True, 1.5, -1])
def test_generate_tool_chat_rejects_invalid_tool_round_limits(max_tool_rounds) -> None:
    with pytest.raises(ValueError, match="max_tool_rounds"):
        generate_tool_chat(
            checkpoint="checkpoint.pt",
            tokenizer_path="tokenizer",
            messages=[{"role": "user", "content": "Use a tool."}],
            tool_handlers={},
            max_tool_rounds=max_tool_rounds,
            device="cpu",
        )


def test_generate_tool_chat_caps_tool_calls_before_execution(monkeypatch) -> None:
    handler_calls = 0

    def load_once(**_kwargs):
        return object(), object(), torch.device("cpu"), {"model_config": {}}

    def fake_generate(**_kwargs):
        return _fake_generated_chat(
            '<tool_call>{"name":"one","arguments":{}}</tool_call>\n'
            '<tool_call>{"name":"one","arguments":{}}</tool_call>'
        )

    def handler(_args):
        nonlocal handler_calls
        handler_calls += 1
        return {"ok": True}

    monkeypatch.setattr(sampling_module, "_load_model_tokenizer_payload", load_once)
    monkeypatch.setattr(sampling_module, "_generate_chat_with_model", fake_generate)

    result = generate_tool_chat(
        checkpoint="checkpoint.pt",
        tokenizer_path="tokenizer",
        messages=[{"role": "user", "content": "Use tools."}],
        tool_handlers={"one": handler},
        max_tool_rounds=1,
        max_tool_calls_per_round=1,
        device="cpu",
    )

    assert result.finish_reason == "tool_call_limit"
    assert handler_calls == 0
    assert result.steps[0].tool_results == ()
    assert [message.role for message in result.messages] == ["user", "assistant"]


def test_chat_cli_can_print_parsed_json(tmp_path: Path, monkeypatch) -> None:
    tokenizer_dir = _tokenizer(tmp_path)
    tokenizer = AnilaTokenizer.load(tokenizer_dir)
    cfg = ModelConfig(vocab_size=300, context_length=64, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    checkpoint = _checkpoint(tmp_path, cfg)
    tools = tmp_path / "tools.json"
    tools.write_text('{"type": "function", "function": {"name": "calculate_math"}}', encoding="utf-8")
    planned_ids = tokenizer.encode(
        "<tool_call>\n{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"2+2\"}}\n</tool_call>",
        add_bos=False,
    )

    def generate_steps(self, input_ids: torch.Tensor, **_kwargs):
        sequences = input_ids
        for token_id in planned_ids:
            next_id = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
            sequences = torch.cat([sequences, next_id], dim=1)
            yield GenerationStep(
                token_ids=next_id,
                token_logprobs=torch.zeros((1, 1), device=input_ids.device),
                sequences=sequences,
                finished=torch.zeros((1,), dtype=torch.bool, device=input_ids.device),
            )

    monkeypatch.setattr(AnilaLM, "generate_steps", generate_steps)

    result = CliRunner().invoke(
        app,
        [
            "model",
            "chat",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--prompt",
            "What is 2+2?",
            "--tools",
            str(tools),
            "--max-new-tokens",
            str(len(planned_ids)),
            "--greedy",
            "--device",
            "cpu",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["assistant"]["tool_calls"][0]["name"] == "calculate_math"
