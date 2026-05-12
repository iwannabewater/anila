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
