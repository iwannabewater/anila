import json
from dataclasses import asdict
from pathlib import Path

import torch
from typer.testing import CliRunner

from anila.cli import app
from anila.config import LoRAConfig, ModelConfig
from anila.model import AnilaLM, GenerationStep
from anila.sampling import generate_text, sample_text, stream_text
from anila.tokenization import AnilaTokenizer, train_byte_bpe


def _tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Anila trains small language models.\n"
        "Beam search keeps multiple deterministic continuations alive.\n"
        "Generation should stay simple and inspectable.\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    return tokenizer_dir


def _checkpoint(tmp_path: Path, cfg: ModelConfig) -> Path:
    checkpoint = tmp_path / "checkpoint.pt"
    model = AnilaLM(cfg)
    torch.save(
        {
            "schema_version": 1,
            "objective": "pretrain",
            "model": model.state_dict(),
            "model_config": asdict(cfg),
            "lora_config": asdict(LoRAConfig()),
            "tokenizer_path": "tokenizer",
            "step": 0,
        },
        checkpoint,
    )
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
            "2",
            "--num-beams",
            "2",
            "--length-penalty",
            "0.7",
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
