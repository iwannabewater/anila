from types import MethodType

import pytest
import torch

from anila.config import ModelConfig
from anila.model import AnilaLM, CausalLMOutput, GenerationStep, apply_repetition_penalty, filter_logits


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


def test_generate_can_run_greedily_with_modern_filters() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 4))

    generated = model.generate(
        x,
        max_new_tokens=3,
        top_k=10,
        top_p=0.9,
        min_p=0.01,
        repetition_penalty=1.1,
        do_sample=False,
    )

    assert generated.shape == (2, 7)


def test_generate_steps_matches_greedy_generate() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 4))

    steps = list(model.generate_steps(x, max_new_tokens=3, top_k=10, do_sample=False))
    generated = model.generate(x, max_new_tokens=3, top_k=10, do_sample=False)

    assert len(steps) == 3
    assert all(isinstance(step, GenerationStep) for step in steps)
    assert steps[-1].sequences.tolist() == generated.tolist()
    assert steps[-1].token_ids.shape == (2, 1)
    assert steps[-1].token_logprobs.shape == (2, 1)


def test_batched_generation_fills_finished_sequences_with_eos() -> None:
    cfg = ModelConfig(vocab_size=16, context_length=8, n_layer=1, n_head=2, n_kv_head=1, n_embd=16).validated()
    model = AnilaLM(cfg)
    steps = iter((torch.tensor([1, 2]), torch.tensor([3, 1])))

    def forward(_self, input_ids: torch.Tensor, **_kwargs) -> CausalLMOutput:
        next_ids = next(steps)
        logits = torch.full((2, input_ids.size(1), cfg.vocab_size), -1000.0)
        logits[torch.arange(2), -1, next_ids] = 0.0
        return CausalLMOutput(logits=logits)

    model.forward = MethodType(forward, model)
    generated = model.generate(
        torch.tensor([[4], [5]]),
        max_new_tokens=2,
        top_k=None,
        eos_id=1,
        use_cache=False,
        do_sample=False,
    )

    assert generated.tolist() == [[4, 1, 1], [5, 2, 1]]


def test_generate_supports_beam_search_for_single_prompt() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))

    generated = model.generate(
        x,
        max_new_tokens=3,
        num_beams=3,
        length_penalty=0.7,
        top_k=10,
        top_p=0.9,
        min_p=0.01,
        repetition_penalty=1.1,
    )

    assert generated.shape == (1, 7)


def test_beam_search_rejects_batched_prompts() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 4))

    with pytest.raises(ValueError, match="batch_size=1"):
        model.generate(x, max_new_tokens=1, num_beams=2)


def test_filter_logits_supports_min_p_and_keeps_at_least_one_token() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0]])

    filtered = filter_logits(logits, top_k=None, top_p=1.0, min_p=0.5)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isinf(filtered[0, 1:]).all()


def test_repetition_penalty_penalizes_seen_tokens_by_score_sign() -> None:
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    input_ids = torch.tensor([[0, 1, 1]])

    adjusted = apply_repetition_penalty(logits, input_ids, penalty=2.0)

    torch.testing.assert_close(adjusted, torch.tensor([[1.0, -4.0, 0.5]]))


def test_generation_filter_validation() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = AnilaLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))

    with pytest.raises(ValueError, match="top_k"):
        model.generate(x, max_new_tokens=1, top_k=0)
    with pytest.raises(ValueError, match="min_p"):
        model.generate(x, max_new_tokens=1, min_p=-0.1)
    with pytest.raises(ValueError, match="repetition_penalty"):
        model.generate(x, max_new_tokens=1, repetition_penalty=0.0)
    with pytest.raises(ValueError, match="num_beams"):
        model.generate(x, max_new_tokens=1, num_beams=0)
    with pytest.raises(ValueError, match="length_penalty"):
        model.generate(x, max_new_tokens=1, length_penalty=-0.1)
