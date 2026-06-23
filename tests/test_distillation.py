import pytest
import torch

from anila.config import DistillConfig, ModelConfig, OPDConfig
from anila.data import IGNORE_INDEX
from anila.distillation import on_policy_distillation_loss, soft_distillation_loss
from anila.model import AnilaLM


def test_soft_distillation_loss_is_zero_for_identical_logits_without_ce() -> None:
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, IGNORE_INDEX], [3, 4, 5]])
    cfg = DistillConfig(mode="soft", teacher_checkpoint="teacher.pt", ce_weight=0.0, kl_weight=1.0)

    loss = soft_distillation_loss(logits, logits, labels, cfg)

    assert loss.kl_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert loss.loss.item() == pytest.approx(0.0, abs=1e-6)


def test_on_policy_distillation_loss_matches_teacher_on_rollout_tokens() -> None:
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[IGNORE_INDEX, 2, 3], [IGNORE_INDEX, IGNORE_INDEX, 5]])
    cfg = OPDConfig(teacher_checkpoint="teacher.pt", ce_weight=0.0, kl_weight=1.0)

    loss = on_policy_distillation_loss(logits, logits, labels, cfg)

    assert loss.kl_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert loss.loss.item() == pytest.approx(0.0, abs=1e-6)


def test_soft_distillation_rejects_mismatched_logits() -> None:
    cfg = DistillConfig(mode="soft", teacher_checkpoint="teacher.pt")
    labels = torch.ones((2, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="identical shape"):
        soft_distillation_loss(torch.randn(2, 3, 7), torch.randn(2, 3, 8), labels, cfg)


def test_teacher_and_student_can_have_different_depth_with_same_vocab() -> None:
    teacher = AnilaLM(ModelConfig(vocab_size=32, context_length=8, n_layer=2, n_head=4, n_kv_head=2, n_embd=32))
    student = AnilaLM(ModelConfig(vocab_size=32, context_length=8, n_layer=1, n_head=4, n_kv_head=2, n_embd=32))
    input_ids = torch.randint(0, 32, (2, 5))
    labels = torch.randint(0, 32, (2, 5))
    cfg = DistillConfig(mode="soft", teacher_checkpoint="teacher.pt")

    loss = soft_distillation_loss(student(input_ids).logits, teacher(input_ids).logits, labels, cfg)

    assert loss.loss.isfinite()
