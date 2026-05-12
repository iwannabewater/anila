from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anila.config import ModelConfig


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight).type_as(x)


def precompute_rope(head_dim: int, seq_len: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[: x.size(-2)].to(device=x.device, dtype=x.dtype)
    sin = sin[: x.size(-2)].to(device=x.device, dtype=x.dtype)
    return (x * cos[None, None, :, :]) + (rotate_half(x) * sin[None, None, :, :])


def repeat_kv(x: torch.Tensor, repeat: int) -> torch.Tensor:
    if repeat == 1:
        return x
    batch, n_kv_head, seq_len, head_dim = x.shape
    return x[:, :, None, :, :].expand(batch, n_kv_head, repeat, seq_len, head_dim).reshape(
        batch, n_kv_head * repeat, seq_len, head_dim
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config = config.validated()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.kv_repeat = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embd = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = repeat_kv(k, self.kv_repeat), repeat_kv(v, self.kv_repeat)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, embd)
        return self.resid_dropout(self.o_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 64 * math.ceil(hidden / 64)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class AnilaLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.validated()
        self.embed = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.norm = RMSNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = precompute_rope(self.config.n_embd // self.config.n_head, self.config.context_length, self.config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.size(1) > self.config.context_length:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds context_length={self.config.context_length}"
            )
        x = self.drop(self.embed(input_ids))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return CausalLMOutput(logits=logits, loss=loss, hidden_states=hidden_states if return_hidden_states else None)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float = 1.0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.context_length :]
            logits = self(idx_cond).logits[:, -1, :] / temperature
            logits = filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and torch.all(next_id.eq(eos_id)):
                break
        return input_ids

    def num_parameters(self, *, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_embeddings:
            total -= self.embed.weight.numel()
        return total


def filter_logits(logits: torch.Tensor, *, top_k: int | None, top_p: float) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1]
        logits = logits.masked_fill(logits < kth[:, None], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(remove.scatter(1, sorted_indices, remove), float("-inf"))
    return logits
