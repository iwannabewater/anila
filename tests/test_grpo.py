import pytest
import torch

from anila.config import GRPOConfig
from anila.grpo import group_advantages, grpo_loss, score_response


def test_score_response_supports_contains_and_exact_match() -> None:
    assert score_response("Small causal language models.", "causal language", "contains") == 1.0
    assert score_response("Small causal language models.", "image database", "contains") == 0.0
    assert score_response("  Atomically.  ", "atomically.", "exact_match") == 1.0


def test_group_advantages_centers_each_prompt_group() -> None:
    rewards = torch.tensor([[0.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    advantages = group_advantages(rewards)

    assert advantages[0].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert torch.equal(advantages[1], torch.zeros(3))


def test_grpo_loss_decreases_for_positive_advantage_higher_policy_logprob() -> None:
    low = grpo_loss(
        policy_logps=torch.tensor([0.0]),
        old_logps=torch.tensor([0.0]),
        reference_logps=torch.tensor([0.0]),
        advantages=torch.tensor([1.0]),
        rewards=torch.tensor([1.0]),
        config=GRPOConfig(beta=0.0),
    )
    high = grpo_loss(
        policy_logps=torch.tensor([0.1]),
        old_logps=torch.tensor([0.0]),
        reference_logps=torch.tensor([0.0]),
        advantages=torch.tensor([1.0]),
        rewards=torch.tensor([1.0]),
        config=GRPOConfig(beta=0.0),
    )

    assert high.loss.item() < low.loss.item()
