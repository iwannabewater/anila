from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from anila.config import LoRAConfig, ModelConfig, RewardConfig
from anila.data import IGNORE_INDEX
from anila.model import AnilaLM
from anila.reward import (
    RewardModel,
    RuleRewardScorer,
    build_reward_scorer,
    last_response_hidden_state,
    reward_model_loss,
)


def test_last_response_hidden_state_uses_last_trainable_label() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 1, 2, IGNORE_INDEX],
            [IGNORE_INDEX, IGNORE_INDEX, 3, 4],
        ]
    )

    pooled = last_response_hidden_state(hidden, labels)

    assert torch.equal(pooled[0], hidden[0, 2])
    assert torch.equal(pooled[1], hidden[1, 3])


def test_reward_model_loss_prefers_chosen_over_rejected() -> None:
    low = reward_model_loss(torch.tensor([0.0]), torch.tensor([0.0]))
    high = reward_model_loss(torch.tensor([2.0]), torch.tensor([0.0]))

    assert high.loss.item() < low.loss.item()
    assert high.margin.item() == pytest.approx(2.0)
    assert high.accuracy.item() == pytest.approx(1.0)


def test_rule_reward_scorer_requires_expected_targets() -> None:
    scorer = RuleRewardScorer("contains")

    rewards = scorer.score(
        torch.empty(0),
        torch.empty(0),
        responses=["Small causal language models."],
        targets=["causal language"],
    )

    assert rewards.tolist() == [1.0]
    with pytest.raises(ValueError, match="expected target"):
        scorer.score(torch.empty(0), torch.empty(0), responses=["anything"], targets=[None])


def test_learned_reward_scorer_loads_native_reward_checkpoint(tmp_path: Path) -> None:
    cfg = ModelConfig(vocab_size=32, context_length=8, n_layer=1, n_head=4, n_kv_head=2, n_embd=32).validated()
    reward_model = RewardModel(AnilaLM(cfg))
    checkpoint = tmp_path / "reward.pt"
    torch.save(
        {
            "schema_version": 1,
            "objective": "reward_model",
            "model": reward_model.backbone.state_dict(),
            "model_config": asdict(cfg),
            "lora_config": asdict(LoRAConfig()),
            "reward_config": asdict(RewardConfig()),
            "reward_head": reward_model.reward_head.state_dict(),
            "step": 1,
        },
        checkpoint,
    )
    scorer = build_reward_scorer(
        RewardConfig(scorer="model", checkpoint=str(checkpoint), scale=2.0, bias=1.0),
        reward_type="contains",
        device=torch.device("cpu"),
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    labels = torch.full((2, 5), IGNORE_INDEX)
    labels[:, 2:] = input_ids[:, 2:]

    rewards = scorer.score(input_ids, labels, responses=["", ""], targets=[None, None])

    assert rewards.shape == (2,)
    assert rewards.isfinite().all()
