import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from anila.checkpoint import load_checkpoint_payload
from anila.config import (
    DataConfig,
    DistillConfig,
    GRPOConfig,
    LoRAConfig,
    ModelConfig,
    PPOConfig,
    RewardConfig,
    RunConfig,
    SFTConfig,
    TrainConfig,
)
from anila.sampling import sample_text
from anila.tokenization import train_byte_bpe
from anila.training import Trainer, configure_optimizer


def test_tiny_training_integration(tmp_path: Path) -> None:
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
            ema_decay=0.9,
        ),
        data=DataConfig(pretrain_mode="packed"),
    )
    trainer = Trainer(run)
    trainer.train()
    model_before_eval = {name: tensor.detach().clone() for name, tensor in trainer._raw_model().state_dict().items()}
    trainer.evaluate()
    for name, expected_tensor in model_before_eval.items():
        torch.testing.assert_close(trainer._raw_model().state_dict()[name], expected_tensor)

    checkpoint = tmp_path / "run" / "checkpoints" / "latest.pt"
    config_snapshot = tmp_path / "run" / "config.json"
    metrics = tmp_path / "run" / "metrics.jsonl"
    assert checkpoint.exists()
    assert config_snapshot.exists()
    assert metrics.exists()
    payload = load_checkpoint_payload(checkpoint)
    metric_events = [json.loads(line)["event"] for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert payload["schema_version"] == 1
    assert payload["objective"] == "pretrain"
    assert payload["data_config"]["pretrain_mode"] == "packed"
    assert payload["train_config"]["ema_decay"] == 0.9
    assert "ema_model" in payload
    assert set(payload["ema_model"]) == set(payload["model"])
    assert "rng_state" in payload
    assert "data_state" in payload
    snapshot = json.loads(config_snapshot.read_text(encoding="utf-8"))
    assert snapshot["train"]["objective"] == "pretrain"
    assert snapshot["data"]["pretrain_mode"] == "packed"
    assert {"train", "eval", "checkpoint"}.issubset(metric_events)

    resumed = Trainer(replace(run, train=replace(run.train, resume=str(checkpoint), max_steps=3)))
    assert resumed.start_step == 2
    assert resumed.ema_state is not None
    assert torch.equal(torch.get_rng_state(), payload["rng_state"]["torch"])

    legacy_checkpoint = tmp_path / "legacy-without-data-state.pt"
    legacy_payload = dict(payload)
    legacy_payload.pop("data_state")
    legacy_payload.pop("ema_model")
    legacy_payload.pop("ema_decay")
    torch.save(legacy_payload, legacy_checkpoint)
    legacy_resumed = Trainer(replace(run, train=replace(run.train, resume=str(legacy_checkpoint), max_steps=3)))
    assert legacy_resumed.start_step == 2
    assert legacy_resumed.ema_state is not None
    next(legacy_resumed.train_iterator)


def test_resume_replays_next_shuffled_batch_mid_epoch(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(f"record {index}: distinct training tokens {chr(65 + index)}" for index in range(24)) + "\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=8, n_layer=1, n_head=2, n_kv_head=1, n_embd=32, dropout=0.1),
        train=TrainConfig(
            dataset_path=str(corpus),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "run"),
            batch_size=2,
            grad_accum_steps=2,
            max_steps=2,
            eval_interval=2,
            save_interval=2,
            log_interval=2,
            device="cpu",
            dtype="float32",
        ),
        data=DataConfig(pretrain_mode="packed"),
    )
    original = Trainer(run)
    next(original.train_iterator)
    next(original.train_iterator)
    checkpoint = original.save(1)
    expected = next(original.train_iterator)

    resumed = Trainer(replace(run, train=replace(run.train, resume=str(checkpoint))))
    actual = next(resumed.train_iterator)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)

    with pytest.raises(ValueError, match="data contract"):
        Trainer(replace(run, train=replace(run.train, resume=str(checkpoint), batch_size=1)))

    boundary_run = replace(run, train=replace(run.train, out_dir=str(tmp_path / "boundary")))
    boundary = Trainer(boundary_run)
    for _ in range(len(boundary.train_loader)):
        next(boundary.train_iterator)
    boundary_checkpoint = boundary.save(1)
    expected_after_epoch = next(boundary.train_iterator)
    resumed_after_epoch = Trainer(
        replace(boundary_run, train=replace(boundary_run.train, resume=str(boundary_checkpoint)))
    )
    actual_after_epoch = next(resumed_after_epoch.train_iterator)
    for actual_tensor, expected_tensor in zip(actual_after_epoch, expected_after_epoch, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)

    continuous_run = replace(
        run,
        train=replace(run.train, out_dir=str(tmp_path / "continuous"), save_interval=1),
    )
    Trainer(continuous_run).train()
    step_one = tmp_path / "continuous" / "checkpoints" / "step_00000001.pt"
    expected_payload = load_checkpoint_payload(tmp_path / "continuous" / "checkpoints" / "latest.pt")

    resumed_run = replace(
        continuous_run,
        train=replace(
            continuous_run.train,
            out_dir=str(tmp_path / "resumed"),
            resume=str(step_one),
        ),
    )
    Trainer(resumed_run).train()
    actual_payload = load_checkpoint_payload(tmp_path / "resumed" / "checkpoints" / "latest.pt")

    for key, expected_tensor in expected_payload["model"].items():
        torch.testing.assert_close(actual_payload["model"][key], expected_tensor)


def test_configure_optimizer_accepts_fused_flag_on_cpu() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=8, n_layer=1, n_head=2, n_kv_head=1, n_embd=32).validated()
    model = torch.nn.Linear(cfg.n_embd, cfg.vocab_size)
    optimizer = configure_optimizer(
        model,
        TrainConfig(dataset_path="data.txt", tokenizer_path="tokenizer", fused_adamw=True),
        torch.device("cpu"),
    )

    assert isinstance(optimizer, torch.optim.AdamW)


def test_tiny_sft_training_integration(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\nAssistant: Anila trains small causal language models.\n" * 20,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "What does Anila train?", "response": "Small causal language models."}\n'
        '{"messages": [{"role": "user", "content": "How are checkpoints saved?"}, '
        '{"role": "assistant", "content": "They are written atomically."}]}\n',
        encoding="utf-8",
    )

    run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="sft",
            dataset_path=str(sft_data),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "sft-run"),
            batch_size=2,
            max_steps=2,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        sft=SFTConfig(format="auto"),
    )
    Trainer(run).train()
    checkpoint = tmp_path / "sft-run" / "checkpoints" / "latest.pt"
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["schema_version"] == 1
    assert payload["objective"] == "sft"
    assert payload["sft_config"]["format"] == "auto"


def test_tiny_lora_sft_training_integration(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\nAssistant: Anila trains small causal language models.\n" * 20,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)

    base_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="pretrain",
            dataset_path=str(corpus),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "base-run"),
            batch_size=2,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
    )
    Trainer(base_run).train()

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "What does Anila train?", "response": "Small causal language models."}\n'
        '{"prompt": "How are adapters saved?", "response": "As adapter checkpoints."}\n',
        encoding="utf-8",
    )
    lora_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="sft",
            init_from=str(tmp_path / "base-run" / "checkpoints" / "latest.pt"),
            dataset_path=str(sft_data),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "lora-run"),
            batch_size=2,
            max_steps=2,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        lora=LoRAConfig(enabled=True, rank=2, alpha=4.0, target_modules=["q_proj", "v_proj"]),
        sft=SFTConfig(format="auto"),
    )
    Trainer(lora_run).train()

    checkpoint = tmp_path / "lora-run" / "checkpoints" / "latest.pt"
    adapter = tmp_path / "lora-run" / "checkpoints" / "adapters" / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu")
    adapter_payload = torch.load(adapter, map_location="cpu")

    assert payload["lora_config"]["enabled"] is True
    assert payload["adapter_checkpoint"] is not None
    assert adapter.exists()
    assert adapter_payload["artifact"] == "lora_adapter"
    assert sorted(adapter_payload["adapter"]) == [
        "blocks.0.attn.q_proj.lora_a.weight",
        "blocks.0.attn.q_proj.lora_b.weight",
        "blocks.0.attn.v_proj.lora_a.weight",
        "blocks.0.attn.v_proj.lora_b.weight",
    ]
    text = sample_text(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer_dir,
        prompt="User: What does Anila train? Assistant:",
        max_new_tokens=1,
        device="cpu",
    )
    assert isinstance(text, str)


def test_tiny_soft_distillation_training_integration(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("Anila distills teacher logits into a smaller student.\n" * 80), encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)

    teacher_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=32, n_layer=2, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="pretrain",
            dataset_path=str(corpus),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "teacher-run"),
            batch_size=4,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
    )
    Trainer(teacher_run).train()

    student_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="distill",
            dataset_path=str(corpus),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "student-run"),
            batch_size=4,
            max_steps=2,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        distill=DistillConfig(
            mode="soft",
            data_objective="pretrain",
            teacher_checkpoint=str(tmp_path / "teacher-run" / "checkpoints" / "latest.pt"),
            temperature=2.0,
            kl_weight=1.0,
            ce_weight=0.5,
        ),
    )
    Trainer(student_run).train()
    payload = torch.load(tmp_path / "student-run" / "checkpoints" / "latest.pt", map_location="cpu")

    assert payload["objective"] == "distill"
    assert payload["distill_config"]["mode"] == "soft"
    assert payload["distill_config"]["data_objective"] == "pretrain"


def test_tiny_dpo_training_integration(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\nAssistant: Anila trains small causal language models.\n" * 80,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "What does Anila train?", "response": "Small causal language models."}\n'
        '{"prompt": "How are checkpoints saved?", "response": "Atomically."}\n',
        encoding="utf-8",
    )
    sft_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="sft",
            dataset_path=str(sft_data),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "sft-run"),
            batch_size=2,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        sft=SFTConfig(format="auto"),
    )
    Trainer(sft_run).train()

    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text(
        '{"prompt": "What does Anila train?", "chosen": "Small causal language models.", '
        '"rejected": "Image databases."}\n'
        '{"prompt": "How are checkpoints saved?", "chosen": "Atomically.", "rejected": "Never."}\n',
        encoding="utf-8",
    )
    dpo_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="dpo",
            init_from=str(tmp_path / "sft-run" / "checkpoints" / "latest.pt"),
            dataset_path=str(prefs),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "dpo-run"),
            batch_size=2,
            max_steps=2,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
    )
    Trainer(dpo_run).train()
    payload = torch.load(tmp_path / "dpo-run" / "checkpoints" / "latest.pt", map_location="cpu")

    assert payload["objective"] == "dpo"
    assert payload["dpo_config"]["beta"] == 0.1


def test_dpo_requires_policy_initial_checkpoint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("User: x\nAssistant: y\n" * 20, encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text('{"prompt": "x", "chosen": "y", "rejected": "z"}\n', encoding="utf-8")
    run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="dpo",
            dataset_path=str(prefs),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "dpo-run"),
            batch_size=1,
            max_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
    )

    with pytest.raises(ValueError, match="train.init_from"):
        Trainer(run)


def test_tiny_reward_model_training_integration(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\nAssistant: Anila trains small causal language models.\n" * 40,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text(
        '{"prompt": "What does Anila train?", "chosen": "Small causal language models.", '
        '"rejected": "Image databases."}\n'
        '{"prompt": "How are checkpoints saved?", "chosen": "Atomically.", "rejected": "Never."}\n',
        encoding="utf-8",
    )

    run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="reward_model",
            dataset_path=str(prefs),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "reward-run"),
            batch_size=2,
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
    payload = torch.load(tmp_path / "reward-run" / "checkpoints" / "latest.pt", map_location="cpu")

    assert payload["objective"] == "reward_model"
    assert payload["reward_head"] is not None
    assert payload["reward_config"]["scorer"] == "rule"


def test_grpo_and_ppo_can_use_learned_reward_scorer_with_prompt_only_data(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\nAssistant: Anila trains small causal language models.\n" * 60,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "What does Anila train?", "response": "Small causal language models."}\n'
        '{"prompt": "How are checkpoints saved?", "response": "Atomically."}\n',
        encoding="utf-8",
    )
    sft_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="sft",
            dataset_path=str(sft_data),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "sft-run"),
            batch_size=2,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        sft=SFTConfig(format="auto"),
    )
    Trainer(sft_run).train()

    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text(
        '{"prompt": "What does Anila train?", "chosen": "Small causal language models.", '
        '"rejected": "Image databases."}\n'
        '{"prompt": "How are checkpoints saved?", "chosen": "Atomically.", "rejected": "Never."}\n',
        encoding="utf-8",
    )
    reward_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="reward_model",
            init_from=str(tmp_path / "sft-run" / "checkpoints" / "latest.pt"),
            dataset_path=str(prefs),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "reward-run"),
            batch_size=2,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
    )
    Trainer(reward_run).train()

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt": "What does Anila train?"}\n'
        '{"prompt": "How are checkpoints saved?"}\n',
        encoding="utf-8",
    )
    reward_checkpoint = str(tmp_path / "reward-run" / "checkpoints" / "latest.pt")
    learned_reward = RewardConfig(scorer="model", checkpoint=reward_checkpoint)

    grpo_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="grpo",
            init_from=str(tmp_path / "sft-run" / "checkpoints" / "latest.pt"),
            dataset_path=str(prompts),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "grpo-run"),
            batch_size=1,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        grpo=GRPOConfig(num_generations=2, max_new_tokens=2, temperature=1.0, top_k=20),
        reward=learned_reward,
    )
    trainer = Trainer(grpo_run)
    rng_state = torch.get_rng_state().clone()
    trainer.evaluate()
    assert torch.equal(torch.get_rng_state(), rng_state)
    trainer.train()

    ppo_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="ppo",
            init_from=str(tmp_path / "sft-run" / "checkpoints" / "latest.pt"),
            dataset_path=str(prompts),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "ppo-run"),
            batch_size=1,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        ppo=PPOConfig(num_rollouts=1, max_new_tokens=2, temperature=1.0, top_k=20),
        reward=learned_reward,
    )
    Trainer(ppo_run).train()

    grpo_payload = torch.load(tmp_path / "grpo-run" / "checkpoints" / "latest.pt", map_location="cpu")
    ppo_payload = torch.load(tmp_path / "ppo-run" / "checkpoints" / "latest.pt", map_location="cpu")
    assert grpo_payload["reward_config"]["scorer"] == "model"
    assert ppo_payload["reward_config"]["scorer"] == "model"


def test_tiny_grpo_training_integration(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\nAssistant: Anila trains small causal language models.\n" * 80,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "What does Anila train?", "response": "Small causal language models."}\n'
        '{"prompt": "How are checkpoints saved?", "response": "Atomically."}\n',
        encoding="utf-8",
    )
    sft_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="sft",
            dataset_path=str(sft_data),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "sft-run"),
            batch_size=2,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        sft=SFTConfig(format="auto"),
    )
    Trainer(sft_run).train()

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt": "What does Anila train?", "expected": "models"}\n'
        '{"prompt": "How are checkpoints saved?", "expected": "atomically"}\n',
        encoding="utf-8",
    )
    grpo_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="grpo",
            init_from=str(tmp_path / "sft-run" / "checkpoints" / "latest.pt"),
            dataset_path=str(prompts),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "grpo-run"),
            batch_size=1,
            max_steps=2,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        grpo=GRPOConfig(num_generations=2, max_new_tokens=2, temperature=1.0, top_k=20),
    )
    Trainer(grpo_run).train()
    payload = torch.load(tmp_path / "grpo-run" / "checkpoints" / "latest.pt", map_location="cpu")

    assert payload["objective"] == "grpo"
    assert payload["grpo_config"]["num_generations"] == 2


def test_grpo_requires_policy_initial_checkpoint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("User: x\nAssistant: y\n" * 20, encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "x", "expected": "y"}\n', encoding="utf-8")
    run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="grpo",
            dataset_path=str(prompts),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "grpo-run"),
            batch_size=1,
            max_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        grpo=GRPOConfig(num_generations=2, max_new_tokens=2),
    )

    with pytest.raises(ValueError, match="train.init_from"):
        Trainer(run)


def test_tiny_ppo_training_integration(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: What does Anila train?\nAssistant: Anila trains small causal language models.\n" * 80,
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        '{"prompt": "What does Anila train?", "response": "Small causal language models."}\n'
        '{"prompt": "How are checkpoints saved?", "response": "Atomically."}\n',
        encoding="utf-8",
    )
    sft_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="sft",
            dataset_path=str(sft_data),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "sft-run"),
            batch_size=2,
            max_steps=1,
            warmup_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        sft=SFTConfig(format="auto"),
    )
    Trainer(sft_run).train()

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt": "What does Anila train?", "expected": "models"}\n'
        '{"prompt": "How are checkpoints saved?", "expected": "atomically"}\n',
        encoding="utf-8",
    )
    ppo_run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=96, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="ppo",
            init_from=str(tmp_path / "sft-run" / "checkpoints" / "latest.pt"),
            dataset_path=str(prompts),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "ppo-run"),
            batch_size=1,
            max_steps=2,
            warmup_steps=1,
            eval_interval=2,
            save_interval=2,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        ppo=PPOConfig(num_rollouts=1, max_new_tokens=2, temperature=1.0, top_k=20),
    )
    Trainer(ppo_run).train()
    checkpoint = tmp_path / "ppo-run" / "checkpoints" / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu")

    assert payload["objective"] == "ppo"
    assert payload["ppo_config"]["num_rollouts"] == 1
    assert payload["value_head"] is not None
    assert isinstance(sample_text(checkpoint=checkpoint, tokenizer_path=tokenizer_dir, prompt="User:", max_new_tokens=1, device="cpu"), str)


def test_ppo_requires_policy_initial_checkpoint(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("User: x\nAssistant: y\n" * 20, encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe([corpus], tokenizer_dir, vocab_size=300, min_frequency=1)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "x", "expected": "y"}\n', encoding="utf-8")
    run = RunConfig(
        model=ModelConfig(vocab_size=300, context_length=32, n_layer=1, n_head=2, n_kv_head=1, n_embd=32),
        train=TrainConfig(
            objective="ppo",
            dataset_path=str(prompts),
            tokenizer_path=str(tokenizer_dir),
            out_dir=str(tmp_path / "ppo-run"),
            batch_size=1,
            max_steps=1,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            device="cpu",
            dtype="float32",
        ),
        ppo=PPOConfig(num_rollouts=1, max_new_tokens=2),
    )

    with pytest.raises(ValueError, match="train.init_from"):
        Trainer(run)
