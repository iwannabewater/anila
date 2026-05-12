import pytest
import torch

from anila.config import ModelConfig
from anila.model import AnilaLM


def test_model_forward_and_generate() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x, targets=x)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss is not None
    hidden_out = model(x, return_hidden_states=True)
    assert hidden_out.hidden_states is not None
    assert hidden_out.hidden_states.shape == (2, 8, cfg.n_embd)

    generated = model.generate(x[:, :4], max_new_tokens=3, top_k=10)
    assert generated.shape == (2, 7)


def test_model_supports_gradient_checkpointing_backward() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    model.set_gradient_checkpointing(True)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(x, targets=x)
    assert out.loss is not None
    out.loss.backward()

    assert model.gradient_checkpointing is True
    assert model.embed.weight.grad is not None


def test_generate_with_cache_works_when_gradient_checkpointing_is_enabled() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    model.set_gradient_checkpointing(True)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 4))

    generated = model.generate(x, max_new_tokens=3, top_k=10)

    assert generated.shape == (2, 7)


def test_model_kv_cache_matches_full_forward() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=2, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 7))

    full = model(x).logits
    prefill = model(x[:, :3], use_cache=True)
    assert prefill.past_key_values is not None
    cached = model(x[:, 3:], past_key_values=prefill.past_key_values, use_cache=True)

    torch.testing.assert_close(cached.logits, full[:, 3:, :], atol=1e-5, rtol=1e-5)
    assert cached.past_key_values is not None
    key, value = cached.past_key_values[0]
    assert key.shape == (2, cfg.n_kv_head, x.size(1), cfg.n_embd // cfg.n_head)
    assert value.shape == key.shape


def test_model_rejects_cached_targets() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 7))
    prefill = model(x[:, :3], use_cache=True)

    with pytest.raises(ValueError, match="targets cannot be used with past_key_values"):
        model(x[:, 3:], targets=x[:, 3:], past_key_values=prefill.past_key_values)
