from types import SimpleNamespace

import pytest
import torch

from anila.config import DPOConfig, GRPOConfig, PPOConfig
from anila.model import CausalLMOutput
from anila.tokenization import DEFAULT_CHAT_SPECIAL_TOKENS, AnilaTokenizer, train_byte_bpe
from anila.training import Trainer


class _FixedModel:
    def __init__(self, outputs: list[CausalLMOutput]):
        self._outputs = iter(outputs)

    def __call__(self, _input_ids: torch.Tensor) -> CausalLMOutput:
        return next(self._outputs)


class _FixedGenerateModel:
    def __init__(self, response_ids: list[int]):
        self.response_ids = torch.tensor(response_ids, dtype=torch.long)
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, prompt: torch.Tensor, **_kwargs) -> torch.Tensor:
        response = self.response_ids.to(prompt.device).unsqueeze(0)
        return torch.cat((prompt, response), dim=1)


def test_dpo_training_loss_includes_policy_aux_loss_average() -> None:
    logits = torch.zeros((1, 1, 4))
    labels = torch.tensor([[1]])
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(dpo=DPOConfig(beta=0.1))
    trainer.model = _FixedModel(
        [
            CausalLMOutput(logits=logits, aux_loss=torch.tensor(1.0)),
            CausalLMOutput(logits=logits, aux_loss=torch.tensor(3.0)),
        ]
    )
    trainer.reference_model = _FixedModel([CausalLMOutput(logits=logits), CausalLMOutput(logits=logits)])

    loss = Trainer._compute_dpo_loss(trainer, labels, labels, labels, labels)

    assert loss.item() == pytest.approx(0.693147 + 2.0, abs=1e-5)


@pytest.mark.parametrize("objective", ["grpo", "ppo"])
def test_tool_call_rollout_responses_preserve_chat_special_tokens(tmp_path, objective: str) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "User: Use a tool.\n"
        "Assistant: <tool_call>{\"name\":\"calculate_math\",\"arguments\":{}}</tool_call>\n4\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    train_byte_bpe(
        [corpus],
        tokenizer_dir,
        vocab_size=300,
        min_frequency=1,
        extra_special_tokens=DEFAULT_CHAT_SPECIAL_TOKENS,
    )
    tokenizer = AnilaTokenizer.load(tokenizer_dir)
    response_ids = tokenizer.encode('<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\n4')
    trainer = Trainer.__new__(Trainer)
    trainer.tokenizer = tokenizer
    trainer.model = _FixedGenerateModel(response_ids)
    trainer.model_cfg = SimpleNamespace(context_length=64)
    trainer.device = torch.device("cpu")
    trainer.train_cfg = SimpleNamespace(objective=objective)
    trainer.config = SimpleNamespace(
        grpo=GRPOConfig(num_generations=2, reward_type="tool_call"),
        ppo=PPOConfig(num_rollouts=1, reward_type="tool_call"),
    )

    _, _, _, _, responses, _ = Trainer._collect_rollout_batches(
        trainer,
        torch.tensor([[tokenizer.bos_id]], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        [None],
        rollouts_per_prompt=1,
        max_new_tokens=8,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
    )

    assert "<tool_call>" in responses[0]
    assert "</tool_call>" in responses[0]
