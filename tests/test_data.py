from pathlib import Path

from anila.config import DPOConfig, GRPOConfig, PPOConfig, RewardConfig, SFTConfig
from anila.data import (
    IGNORE_INDEX,
    PreferenceDataset,
    PromptRewardDataset,
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
