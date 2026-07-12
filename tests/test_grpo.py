import json

import pytest
import torch

from anila.config import GRPOConfig
from anila.grpo import group_advantages, grpo_loss, score_response


def test_score_response_supports_contains_and_exact_match() -> None:
    assert score_response("Small causal language models.", "causal language", "contains") == 1.0
    assert score_response("Small causal language models.", "image database", "contains") == 0.0
    assert score_response("  Atomically.  ", "atomically.", "exact_match") == 1.0


def test_score_response_supports_tool_call_rewards() -> None:
    expected = json.dumps({"answers": ["4"], "tools": ["calculate_math"]})
    response = (
        "<think>\nThis needs one calculator call before a concise final answer.\n</think>\n"
        '<tool_call>\n{"name":"calculate_math","arguments":{"expression":"2+2"}}\n</tool_call>\n'
        "The answer is 4."
    )
    bad_response = '<tool_call>{"name":"calculate_math","arguments":"{not json"}</tool_call>\nNo answer.'

    assert score_response(response, expected, "tool_call") == pytest.approx(3.0)
    assert score_response(bad_response, expected, "tool_call") < 0.0


def test_tool_call_reward_counts_repeated_expected_tools() -> None:
    expected = json.dumps({"answers": ["4"], "tools": ["calculate_math", "calculate_math"]})
    one_call = '<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\n4'
    two_calls = (
        '<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\n'
        '<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\n4'
    )

    assert score_response(two_calls, expected, "tool_call") > score_response(one_call, expected, "tool_call")


def test_tool_call_reward_rejects_missing_required_tool() -> None:
    expected = json.dumps({"answers": ["4"], "tools": ["calculate_math"]})

    assert score_response("The answer is 4.", expected, "tool_call") <= 0.0


def test_tool_call_reward_rejects_duplicate_wrong_allowed_tool() -> None:
    expected = json.dumps({"answers": ["ok"], "tools": ["search", "calculate_math"]})
    response = (
        '<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\n'
        '<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\nok'
    )

    assert score_response(response, expected, "tool_call") <= 0.0


def test_tool_call_reward_scores_open_thinking_completions() -> None:
    expected = json.dumps({"answers": ["4"], "tools": ["calculate_math"]})
    response = (
        "This is a sufficiently detailed arithmetic plan before using the tool.\n</think>\n"
        '<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call>\n4'
    )

    assert score_response(response, expected, "tool_call") == pytest.approx(3.0)


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


def test_cispo_loss_keeps_gradient_when_ratio_is_capped() -> None:
    policy_logps = torch.tensor([2.0], requires_grad=True)
    out = grpo_loss(
        policy_logps=policy_logps,
        old_logps=torch.tensor([0.0]),
        reference_logps=torch.tensor([0.0]),
        advantages=torch.tensor([1.0]),
        rewards=torch.tensor([1.0]),
        config=GRPOConfig(beta=0.0, loss_type="cispo", cispo_ratio_cap=1.1),
    )

    out.loss.backward()

    assert policy_logps.grad is not None
    torch.testing.assert_close(policy_logps.grad, torch.tensor([-1.1]))


def test_cispo_loss_uses_reference_kl_penalty() -> None:
    without_kl = grpo_loss(
        policy_logps=torch.tensor([0.0]),
        old_logps=torch.tensor([0.0]),
        reference_logps=torch.tensor([-0.2]),
        advantages=torch.tensor([0.0]),
        rewards=torch.tensor([1.0]),
        config=GRPOConfig(beta=0.0, loss_type="cispo"),
    )
    with_kl = grpo_loss(
        policy_logps=torch.tensor([0.0]),
        old_logps=torch.tensor([0.0]),
        reference_logps=torch.tensor([-0.2]),
        advantages=torch.tensor([0.0]),
        rewards=torch.tensor([1.0]),
        config=GRPOConfig(beta=0.5, loss_type="cispo"),
    )

    assert with_kl.loss.item() > without_kl.loss.item()
