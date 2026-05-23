from pathlib import Path

import pytest

from anila.tokenization import AnilaTokenizer, train_byte_bpe


def test_train_save_load_tokenizer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("anila trains tiny language models\nanila samples text\n", encoding="utf-8")
    out_dir = tmp_path / "tokenizer"

    tokenizer = train_byte_bpe([corpus], out_dir, vocab_size=300, min_frequency=1)
    ids = tokenizer.encode("anila", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id

    loaded = AnilaTokenizer.load(out_dir)
    assert loaded.decode(loaded.encode("anila")).strip() == "anila"


def test_train_tokenizer_rejects_invalid_utf8(tmp_path: Path) -> None:
    corpus = tmp_path / "invalid.txt"
    corpus.write_bytes(b"valid text\n\xff\n")

    with pytest.raises(UnicodeDecodeError):
        train_byte_bpe([corpus], tmp_path / "tokenizer", vocab_size=300, min_frequency=1)
