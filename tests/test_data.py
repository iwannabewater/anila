from pathlib import Path

import pytest

from anila.config import DataConfig, DPOConfig, GRPOConfig, PPOConfig, RewardConfig, SFTConfig, load_run_config
from anila.data import (
    IGNORE_INDEX,
    PreferenceDataset,
    PromptRewardDataset,
    StreamingTextTokenDataset,
    SupervisedFineTuneDataset,
    TextTokenDataset,
    create_dataloader,
)
from anila.tokenization import train_byte_bpe


def _tokenizer(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "System: You are concise.\n"
        "User: What does Anila train?\n"
        "Assistant: Anila trains small models.\n"
        "More text for byte pair encoding.\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    return train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)


def test_pretrain_dataset_accepts_multiple_files(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Anila trains from plain text.\n", encoding="utf-8")
    second.write_text("Multiple files can form one corpus.\n", encoding="utf-8")

    dataset = TextTokenDataset([first, second], tokenizer, context_length=8)

    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == y.shape == (8,)


def test_pretrain_dataset_rejects_invalid_utf8(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "invalid.txt"
    data.write_bytes(b"enough valid bytes for decoding first\n\xff\n")

    with pytest.raises(UnicodeDecodeError):
        TextTokenDataset(data, tokenizer, context_length=8)


def test_pretrain_dataset_supports_packed_blocks(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "corpus.txt"
    data.write_text("Anila packs local text into fixed training blocks.\n" * 20, encoding="utf-8")

    sliding = TextTokenDataset(data, tokenizer, context_length=8)
    packed = TextTokenDataset(data, tokenizer, context_length=8, config=DataConfig(pretrain_mode="packed"))

    assert 0 < len(packed) < len(sliding)
    first_x, first_y = packed[0]
    second_x, _ = packed[1]
    assert first_x.shape == first_y.shape == (8,)
    assert second_x[0].item() == first_y[-1].item()


def test_pretrain_dataset_supports_sliding_stride(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "corpus.txt"
    data.write_text("Sliding strides reduce overlap while staying map-style.\n" * 20, encoding="utf-8")

    dense = TextTokenDataset(data, tokenizer, context_length=8)
    strided = TextTokenDataset(data, tokenizer, context_length=8, config=DataConfig(sequence_stride=4))

    assert 0 < len(strided) < len(dense)
    second_x, _ = strided[1]
    assert second_x[0].item() == dense.tokens[4].item()


def test_streaming_pretrain_dataset_yields_fixed_length_blocks(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "corpus.txt"
    data.write_text("Streaming avoids materializing the whole corpus tensor.\n" * 20, encoding="utf-8")

    dataset = StreamingTextTokenDataset(data, tokenizer, context_length=8)
    x, y = next(iter(dataset))

    assert x.shape == y.shape == (8,)


def test_streaming_pretrain_dataset_rejects_invalid_utf8(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "invalid.txt"
    data.write_bytes(b"\xff\n")

    dataset = StreamingTextTokenDataset(data, tokenizer, context_length=8)
    with pytest.raises(UnicodeDecodeError):
        next(iter(dataset))


def test_streaming_pretrain_dataloader_batches_examples(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "corpus.txt"
    data.write_text("Streaming dataloaders emit regular language-model batches.\n" * 20, encoding="utf-8")

    loader = create_dataloader(
        data,
        tokenizer,
        context_length=8,
        batch_size=2,
        objective="pretrain",
        data_config=DataConfig(pretrain_mode="streaming"),
        shuffle=True,
        drop_last=False,
    )
    input_ids, labels = next(iter(loader))

    assert input_ids.shape == labels.shape == (2, 8)


def test_streaming_pretrain_dataset_rejects_too_small_corpus(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "tiny.txt"
    data.write_text("short\n", encoding="utf-8")

    dataset = StreamingTextTokenDataset(data, tokenizer, context_length=128)
    with pytest.raises(ValueError, match="produced no"):
        next(iter(dataset))


@pytest.mark.parametrize("config_name", ["pretrain.json", "distill-soft-pretrain.json"])
def test_packed_pretrain_quickstart_configs_have_training_batches(tmp_path: Path, config_name: str) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = train_byte_bpe(
        [Path("examples/tiny_corpus.txt"), Path("examples/tiny_sft.jsonl")],
        tokenizer_dir,
        vocab_size=512,
        min_frequency=1,
    )
    cfg = load_run_config(Path("configs/quickstart") / config_name)
    data_objective = cfg.distill.data_objective if cfg.train.objective == "distill" else cfg.train.objective

    loader = create_dataloader(
        cfg.train.dataset_path,
        tokenizer,
        context_length=cfg.model.context_length,
        batch_size=cfg.train.batch_size,
        objective=data_objective,
        data_config=cfg.data,
        drop_last=True,
    )
    input_ids, labels = next(iter(loader))

    assert input_ids.shape == labels.shape
    assert input_ids.size(0) == cfg.train.batch_size


def test_sft_dataset_masks_prompt_tokens(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "sft.jsonl"
    data.write_text('{"prompt": "What does Anila train?", "response": "Small language models."}\n', encoding="utf-8")

    dataset = SupervisedFineTuneDataset(data, tokenizer, context_length=64, config=SFTConfig())
    input_ids, labels = dataset[0]

    assert input_ids.shape == labels.shape
    assert labels[0].item() == IGNORE_INDEX
    assert any(label.item() != IGNORE_INDEX for label in labels)
    assert labels[-1].item() == tokenizer.eos_id


def test_sft_dataset_rejects_invalid_utf8(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "invalid.jsonl"
    data.write_bytes(b'\xff{"prompt": "Question", "response": "Answer"}\n')

    with pytest.raises(UnicodeDecodeError):
        SupervisedFineTuneDataset(data, tokenizer, context_length=64)


def test_sft_dataloader_pads_inputs_and_masks_labels(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "sft.jsonl"
    data.write_text(
        '{"prompt": "Short?", "response": "Yes."}\n'
        '{"messages": [{"role": "user", "content": "Longer question?"}, '
        '{"role": "assistant", "content": "A slightly longer answer."}]}\n',
        encoding="utf-8",
    )

    loader = create_dataloader(
        data,
        tokenizer,
        context_length=64,
        batch_size=2,
        objective="sft",
        sft_config=SFTConfig(format="auto"),
        shuffle=False,
        drop_last=False,
    )
    input_ids, labels = next(iter(loader))

    assert input_ids.shape == labels.shape
    assert input_ids.size(0) == 2
    assert (labels == IGNORE_INDEX).any()


def test_preference_dataset_builds_chosen_and_rejected_pairs(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "prefs.jsonl"
    data.write_text(
        '{"prompt": "What does Anila train?", "chosen": "Small models.", "rejected": "Nothing."}\n',
        encoding="utf-8",
    )

    dataset = PreferenceDataset(data, tokenizer, context_length=64, config=DPOConfig())
    chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = dataset[0]

    assert chosen_input_ids.shape == chosen_labels.shape
    assert rejected_input_ids.shape == rejected_labels.shape
    assert chosen_labels[0].item() == IGNORE_INDEX
    assert rejected_labels[0].item() == IGNORE_INDEX
    assert any(label.item() != IGNORE_INDEX for label in chosen_labels)
    assert any(label.item() != IGNORE_INDEX for label in rejected_labels)


def test_dpo_dataloader_pads_chosen_and_rejected_pairs(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "prefs.jsonl"
    data.write_text(
        '{"prompt": "Short?", "chosen": "Yes.", "rejected": "No."}\n'
        '{"prompt": "Longer question?", "chosen": "A slightly longer chosen answer.", '
        '"rejected": "Bad."}\n',
        encoding="utf-8",
    )

    loader = create_dataloader(
        data,
        tokenizer,
        context_length=64,
        batch_size=2,
        objective="dpo",
        dpo_config=DPOConfig(),
        shuffle=False,
        drop_last=False,
    )
    chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = next(iter(loader))

    assert chosen_input_ids.shape == chosen_labels.shape
    assert rejected_input_ids.shape == rejected_labels.shape
    assert chosen_input_ids.size(0) == 2
    assert rejected_input_ids.size(0) == 2
    assert (chosen_labels == IGNORE_INDEX).any()
    assert (rejected_labels == IGNORE_INDEX).any()


def test_prompt_reward_dataset_builds_grpo_prompts(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "prompts.jsonl"
    data.write_text(
        '{"prompt": "What does Anila train?", "expected": "models", "system": "Answer tersely."}\n',
        encoding="utf-8",
    )

    dataset = PromptRewardDataset(data, tokenizer, context_length=64, config=GRPOConfig())
    input_ids, expected = dataset[0]
    decoded = tokenizer.decode(input_ids.tolist())

    assert input_ids[0].item() == tokenizer.bos_id
    assert "User:" in decoded
    assert "Assistant:" in decoded
    assert expected == "models"


def test_grpo_dataloader_pads_prompts_and_keeps_expected_strings(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "prompts.jsonl"
    data.write_text(
        '{"prompt": "Short?", "expected": "yes"}\n'
        '{"prompt": "What does Anila train?", "expected": "models"}\n',
        encoding="utf-8",
    )

    loader = create_dataloader(
        data,
        tokenizer,
        context_length=64,
        batch_size=2,
        objective="grpo",
        grpo_config=GRPOConfig(num_generations=2),
        shuffle=False,
        drop_last=False,
    )
    input_ids, lengths, expected = next(iter(loader))

    assert input_ids.size(0) == 2
    assert lengths.shape == (2,)
    assert expected == ["yes", "models"]


def test_prompt_reward_dataloader_allows_prompt_only_records(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "prompts.jsonl"
    data.write_text('{"prompt": "What does Anila train?"}\n', encoding="utf-8")

    loader = create_dataloader(
        data,
        tokenizer,
        context_length=64,
        batch_size=1,
        objective="grpo",
        grpo_config=GRPOConfig(num_generations=2),
        shuffle=False,
        drop_last=False,
    )
    input_ids, lengths, expected = next(iter(loader))

    assert input_ids.size(0) == 1
    assert lengths.shape == (1,)
    assert expected == [None]


def test_ppo_dataloader_uses_prompt_reward_records(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "prompts.jsonl"
    data.write_text(
        '{"prompt": "Short?", "expected": "yes"}\n'
        '{"prompt": "What does Anila train?", "expected": "models"}\n',
        encoding="utf-8",
    )

    loader = create_dataloader(
        data,
        tokenizer,
        context_length=64,
        batch_size=2,
        objective="ppo",
        ppo_config=PPOConfig(num_rollouts=1),
        shuffle=False,
        drop_last=False,
    )
    input_ids, lengths, expected = next(iter(loader))

    assert input_ids.size(0) == 2
    assert lengths.shape == (2,)
    assert expected == ["yes", "models"]


def test_reward_model_dataloader_uses_preference_records(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    data = tmp_path / "prefs.jsonl"
    data.write_text(
        '{"prompt": "What does Anila train?", "chosen": "Small models.", "rejected": "Nothing."}\n',
        encoding="utf-8",
    )

    loader = create_dataloader(
        data,
        tokenizer,
        context_length=64,
        batch_size=1,
        objective="reward_model",
        reward_config=RewardConfig(),
        shuffle=False,
        drop_last=False,
    )
    chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = next(iter(loader))

    assert chosen_input_ids.shape == chosen_labels.shape
    assert rejected_input_ids.shape == rejected_labels.shape
    assert any(label.item() != IGNORE_INDEX for label in chosen_labels[0])
    assert any(label.item() != IGNORE_INDEX for label in rejected_labels[0])
