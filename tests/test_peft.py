import torch

from anila.config import LoRAConfig, ModelConfig
from anila.model import AnilaLM
from anila.peft import apply_lora, lora_state_dict, mark_lora_trainable, trainable_parameter_names


def test_lora_injection_preserves_initial_output_and_freezes_base() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    before = model(x).logits.detach()

    replaced = apply_lora(model, LoRAConfig(enabled=True, rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"]))
    mark_lora_trainable(model, train_base=False, train_bias=False)
    after = model(x).logits.detach()

    assert replaced == ["blocks.0.attn.q_proj", "blocks.0.attn.v_proj"]
    assert torch.allclose(before, after)
    trainable = trainable_parameter_names(model)
    assert trainable
    assert all(".lora_" in name for name in trainable)


def test_lora_state_dict_contains_only_adapter_weights() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    apply_lora(model, LoRAConfig(enabled=True, rank=2, target_modules=["q_proj"]))

    adapter = lora_state_dict(model)

    assert sorted(adapter) == [
        "blocks.0.attn.q_proj.lora_a.weight",
        "blocks.0.attn.q_proj.lora_b.weight",
    ]
