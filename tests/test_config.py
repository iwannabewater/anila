from pathlib import Path

import pytest

from anila.config import ModelConfig, load_run_config


def test_model_config_fills_kv_heads() -> None:
    cfg = ModelConfig(n_embd=64, n_head=4, n_kv_head=None).validated()
    assert cfg.n_kv_head == 4


def test_invalid_attention_shape_fails() -> None:
    with pytest.raises(ValueError, match="n_embd"):
        ModelConfig(n_embd=65, n_head=4).validated()


def test_load_run_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """
        {
          "model": {"context_length": 8, "unknown": 1},
          "train": {"dataset_path": "x", "tokenizer_path": "y"}
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown config key"):
        load_run_config(path)
