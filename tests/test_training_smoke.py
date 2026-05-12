from pathlib import Path

from anila.config import ModelConfig, RunConfig, TrainConfig
from anila.tokenization import train_byte_bpe
from anila.training import Trainer


def test_tiny_training_smoke(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("anila learns from text\n" * 80), encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)

    run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=16, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            dataset_path=str(corpus),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "run"),
            batch_size=4,
            max_steps=2,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
    )
    Trainer(run).train()
    assert (tmp_path / "run" / "checkpoints" / "latest.pt").exists()
