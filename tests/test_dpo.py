import pytest
import torch

from anila.config import DPOConfig
from anila.data import IGNORE_INDEX
from anila.dpo import dpo_loss, sequence_logprobs


def test_sequence_logprobs_sums_only_trainable_labels() -> None:
    logits = torch.full((1, 3, 4), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    labels = torch.tensor([[1, IGNORE_INDEX, 3]])

    logps = sequence_logprobs(logits, labels)

    assert logps.item() == pytest.approx(0.0, abs=1e-5)


def test_dpo_loss_is_log_two_when_policy_matches_reference() -> None:
    values = torch.tensor([1.0, 2.0])
    out = dpo_loss(values, values - 0.5, values, values - 0.5, DPOConfig(beta=0.1))

    assert out.loss.item() == pytest.approx(0.693147, abs=1e-5)
    assert out.preference_margin.item() == pytest.approx(0.0, abs=1e-6)


def test_dpo_loss_decreases_when_policy_prefers_chosen_more_than_reference() -> None:
    out = dpo_loss(
        policy_chosen_logps=torch.tensor([3.0]),
        policy_rejected_logps=torch.tensor([0.0]),
        reference_chosen_logps=torch.tensor([1.0]),
        reference_rejected_logps=torch.tensor([0.0]),
        config=DPOConfig(beta=1.0),
    )

    assert out.loss.item() < 0.2
    assert out.preference_margin.item() > 0
