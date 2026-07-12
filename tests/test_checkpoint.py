from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from typer.testing import CliRunner

from anila.checkpoint import (
    export_safetensors_checkpoint,
    inspect_checkpoint,
    load_checkpoint_payload,
    merge_lora_checkpoint,
)
from anila.cli import app
from anila.config import DataConfig, LoRAConfig, ModelConfig, TrainConfig
from anila.model import AnilaLM
from anila.peft import LoRALinear, apply_lora


class _UnsupportedCheckpointObject:
    pass


def test_load_checkpoint_payload_rejects_unsafe_objects(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"unsupported": _UnsupportedCheckpointObject()}, checkpoint)

    with pytest.raises(ValueError, match="loaded safely"):
        load_checkpoint_payload(checkpoint)


def test_inspect_checkpoint_summarizes_native_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "reward_model",
            "step": 3,
            "model": {},
            "ema_model": {"weight": torch.zeros(1, 32)},
            "ema_decay": 0.99,
            "optimizer": {},
            "model_config": asdict(
                ModelConfig(
                    vocab_size=128,
                    context_length=64,
                    n_layer=1,
                    n_head=2,
                    n_embd=32,
                    rope_scaling="yarn",
                    rope_scaling_factor=4.0,
                    rope_original_context_length=16,
                    rope_yarn_attention_factor=1.2,
                    moe_num_experts=4,
                    moe_top_k=2,
                    moe_intermediate_size=64,
                    moe_aux_loss_coef=0.01,
                )
            ),
            "train_config": asdict(
                TrainConfig(dataset_path="data.jsonl", tokenizer_path="tokenizer", objective="reward_model")
            ),
            "data_config": asdict(DataConfig(pretrain_mode="packed")),
            "tokenizer_path": "tokenizer",
            "lora_config": {"enabled": False},
            "value_head": None,
            "reward_head": {"weight": torch.zeros(1, 32)},
            "adapter_checkpoint": None,
        },
        checkpoint,
    )

    summary = inspect_checkpoint(checkpoint)

    assert summary["objective"] == "reward_model"
    assert summary["step"] == 3
    assert summary["has_model"] is True
    assert summary["has_ema"] is True
    assert summary["has_reward_head"] is True
    assert summary["model"]["context_length"] == 64
    assert summary["model"]["rope_scaling"] == "yarn"
    assert summary["model"]["rope_scaling_factor"] == 4.0
    assert summary["model"]["rope_original_context_length"] == 16
    assert summary["model"]["rope_yarn_attention_factor"] == 1.2
    assert summary["model"]["moe_num_experts"] == 4
    assert summary["model"]["moe_top_k"] == 2
    assert summary["model"]["moe_intermediate_size"] == 64
    assert summary["model"]["moe_normalize_top_k"] is True
    assert summary["model"]["moe_aux_loss_coef"] == 0.01
    assert summary["train"]["objective"] == "reward_model"
    assert summary["data"]["pretrain_mode"] == "packed"


def test_inspect_checkpoint_cli_prints_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "pretrain",
            "step": 1,
            "model": {},
            "model_config": asdict(ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=2, n_embd=32)),
            "train_config": asdict(TrainConfig(dataset_path="data.txt", tokenizer_path="tokenizer")),
            "tokenizer_path": "tokenizer",
        },
        checkpoint,
    )

    result = CliRunner().invoke(app, ["inspect-checkpoint", "--checkpoint", str(checkpoint)])

    assert result.exit_code == 0
    assert '"objective": "pretrain"' in result.output

    grouped = CliRunner().invoke(app, ["checkpoint", "inspect", "--checkpoint", str(checkpoint)])
    assert grouped.exit_code == 0
    assert '"objective": "pretrain"' in grouped.output


def test_merge_lora_checkpoint_exports_plain_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    lora_cfg = LoRAConfig(enabled=True, rank=2, alpha=4.0, target_modules=["q_proj", "v_proj"])
    model = AnilaLM(cfg)
    targets = apply_lora(model, lora_cfg)
    model.eval()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.lora_b.weight.fill_(0.01)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    expected = model(input_ids).logits.detach()

    checkpoint = tmp_path / "lora.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "sft",
            "model": model.state_dict(),
            "ema_model": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
            "ema_decay": 0.99,
            "model_config": asdict(cfg),
            "train_config": asdict(TrainConfig(dataset_path="data.jsonl", tokenizer_path="tokenizer", objective="sft")),
            "tokenizer_path": "tokenizer",
            "lora_config": asdict(lora_cfg),
            "lora_targets": targets,
            "adapter_checkpoint": None,
            "step": 1,
        },
        checkpoint,
    )

    out = merge_lora_checkpoint(checkpoint, tmp_path / "merged.pt")
    payload = load_checkpoint_payload(out)
    merged = AnilaLM(cfg)
    merged.load_state_dict(payload["model"])
    merged.eval()

    assert payload["lora_config"]["enabled"] is False
    assert payload["merged_lora_targets"] == targets
    assert not any(".lora_" in key or ".base." in key for key in payload["model"])
    assert not any(".lora_" in key or ".base." in key for key in payload["ema_model"])
    assert inspect_checkpoint(out)["is_merged_lora"] is True
    torch.testing.assert_close(merged(input_ids).logits, expected, atol=1e-5, rtol=1e-5)


def test_merge_lora_checkpoint_cli(tmp_path: Path) -> None:
    cfg = ModelConfig(vocab_size=32, context_length=8, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    lora_cfg = LoRAConfig(enabled=True, rank=2, target_modules=["q_proj"])
    model = AnilaLM(cfg)
    targets = apply_lora(model, lora_cfg)
    checkpoint = tmp_path / "lora.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "sft",
            "model": model.state_dict(),
            "model_config": asdict(cfg),
            "train_config": asdict(TrainConfig(dataset_path="data.jsonl", tokenizer_path="tokenizer", objective="sft")),
            "tokenizer_path": "tokenizer",
            "lora_config": asdict(lora_cfg),
            "lora_targets": targets,
            "adapter_checkpoint": None,
            "step": 1,
        },
        checkpoint,
    )

    out = tmp_path / "merged.pt"
    result = CliRunner().invoke(app, ["checkpoint", "merge-lora", "--checkpoint", str(checkpoint), "--out", str(out)])

    assert result.exit_code == 0
    assert out.exists()
    assert "saved merged checkpoint" in result.output


def test_export_safetensors_checkpoint_writes_weights_and_manifest(tmp_path: Path) -> None:
    cfg = ModelConfig(vocab_size=32, context_length=8, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    checkpoint = tmp_path / "policy.pt"
    model = AnilaLM(cfg)
    torch.save(
        {
            "schema_version": 1,
            "objective": "pretrain",
            "model": model.state_dict(),
            "ema_model": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
            "ema_decay": 0.99,
            "model_config": asdict(cfg),
            "train_config": asdict(TrainConfig(dataset_path="data.txt", tokenizer_path="tokenizer")),
            "tokenizer_path": "tokenizer",
            "step": 2,
        },
        checkpoint,
    )

    summary = export_safetensors_checkpoint(checkpoint, tmp_path / "export")

    weights = Path(summary["weights_path"])
    manifest = Path(summary["manifest_path"])
    tensors = load_file(weights)

    assert weights.exists()
    assert manifest.exists()
    assert summary["artifact"] == "anila_safetensors"
    assert summary["weights"] == "model.safetensors"
    assert summary["tensor_groups"]["model"] == len(model.state_dict())
    assert summary["tensor_groups"]["ema_model"] == len(model.state_dict())
    assert "model.embed.weight" in tensors
    assert "ema_model.embed.weight" in tensors
    assert '"objective": "pretrain"' in manifest.read_text(encoding="utf-8")


def test_export_safetensors_checkpoint_cli(tmp_path: Path) -> None:
    cfg = ModelConfig(vocab_size=32, context_length=8, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "pretrain",
            "model": AnilaLM(cfg).state_dict(),
            "model_config": asdict(cfg),
            "tokenizer_path": "tokenizer",
        },
        checkpoint,
    )

    result = CliRunner().invoke(
        app,
        ["checkpoint", "export-safetensors", "--checkpoint", str(checkpoint), "--out-dir", str(tmp_path / "export")],
    )

    assert result.exit_code == 0
    assert '"artifact": "anila_safetensors"' in result.output
    assert (tmp_path / "export" / "model.safetensors").exists()


def test_export_safetensors_checkpoint_rejects_nested_weight_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model": {},
            "model_config": asdict(ModelConfig(vocab_size=32, context_length=8, n_layer=1, n_head=4, n_embd=32)),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="weights_name"):
        export_safetensors_checkpoint(checkpoint, tmp_path / "export", weights_name="nested/model.safetensors")
