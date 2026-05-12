import pytest
import torch

from anila.config import ModelConfig, PPOConfig
from anila.data import IGNORE_INDEX
from anila.model import AnilaLM
from anila.ppo import PolicyValueModel, compute_gae, ppo_loss, token_logprobs


def test_policy_value_model_returns_logits_and_values() -> None:
    cfg = ModelConfig(vocab_size=64, context_length=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    model = PolicyValueModel(AnilaLM(cfg))
    x = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(x)

    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.values.shape == (2, 8)


def test_token_logprobs_masks_ignored_labels() -> None:
    logits = torch.full((1, 3, 4), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    labels = torch.tensor([[1, IGNORE_INDEX, 3]])

    logps = token_logprobs(logits, labels)

    assert logps[0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert logps[0, 1].item() == 0.0
    assert logps[0, 2].item() == pytest.approx(0.0, abs=1e-5)


def test_compute_gae_places_terminal_reward_on_previous_tokens() -> None:
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[True, True, False]])

    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, gae_lambda=1.0)

    assert advantages[0, 0].item() == pytest.approx(1.0)
    assert advantages[0, 1].item() == pytest.approx(1.0)
    assert returns[0, 2].item() == 0.0


def test_ppo_loss_decreases_for_positive_advantage_higher_policy_logprob() -> None:
    mask = torch.tensor([[True]])
    low = ppo_loss(
        policy_logprobs=torch.tensor([[0.0]]),
        old_logprobs=torch.tensor([[0.0]]),
        values=torch.tensor([[0.0]]),
        old_values=torch.tensor([[0.0]]),
        returns=torch.tensor([[0.0]]),
        advantages=torch.tensor([[1.0]]),
        entropy=torch.tensor([[0.0]]),
        mask=mask,
        config=PPOConfig(value_loss_coef=0.0),
    )
    high = ppo_loss(
        policy_logprobs=torch.tensor([[0.1]]),
        old_logprobs=torch.tensor([[0.0]]),
        values=torch.tensor([[0.0]]),
        old_values=torch.tensor([[0.0]]),
        returns=torch.tensor([[0.0]]),
        advantages=torch.tensor([[1.0]]),
        entropy=torch.tensor([[0.0]]),
        mask=mask,
        config=PPOConfig(value_loss_coef=0.0),
    )

    assert high.loss.item() < low.loss.item()
