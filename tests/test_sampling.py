from dataclasses import asdict
from pathlib import Path

import torch
from typer.testing import CliRunner

from anila.cli import app
from anila.config import LoRAConfig, ModelConfig
from anila.model import AnilaLM
from anila.sampling import sample_text
from anila.tokenization import train_byte_bpe


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
