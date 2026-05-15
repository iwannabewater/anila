from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from anila.config import ModelConfig

KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    past_key_values: KVCache | None = None


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


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
    cos = cos[offset : offset + x.size(-2)].to(device=x.device, dtype=x.dtype)
    sin = sin[offset : offset + x.size(-2)].to(device=x.device, dtype=x.dtype)
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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, embd = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        past_len = 0 if past_key_value is None else past_key_value[0].size(-2)
        q, k = apply_rope(q, cos, sin, offset=past_len), apply_rope(k, cos, sin, offset=past_len)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present = (k, v) if use_cache else None
        k, v = repeat_kv(k, self.kv_repeat), repeat_kv(v, self.kv_repeat)
        attn_mask = None
        is_causal = past_key_value is None
        if past_key_value is not None and seq_len > 1:
            total_len = k.size(-2)
            query_positions = torch.arange(past_len, past_len + seq_len, device=x.device)
            key_positions = torch.arange(total_len, device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            attn_mask = torch.zeros((seq_len, total_len), dtype=q.dtype, device=x.device).masked_fill(
                ~allowed,
                float("-inf"),
            )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, embd)
        return self.resid_dropout(self.o_proj(y)), present


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

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out
        return x + self.mlp(self.mlp_norm(x)), present


class AnilaLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.validated()
        self.embed = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.gradient_checkpointing = False
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
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per transformer block")
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if input_ids.size(1) + past_len > self.config.context_length:
            raise ValueError(
                f"sequence length {input_ids.size(1) + past_len} exceeds context_length={self.config.context_length}"
            )
        if targets is not None and past_key_values is not None:
            raise ValueError("targets cannot be used with past_key_values")
        x = self.drop(self.embed(input_ids))
        next_past_key_values = [] if use_cache else None
        for block_index, block in enumerate(self.blocks):
            past_key_value = None if past_key_values is None else past_key_values[block_index]
            if self.gradient_checkpointing and self.training and not use_cache and past_key_values is None:
                x = checkpoint(
                    lambda hidden, cos, sin, block=block: block(hidden, cos, sin)[0],
                    x,
                    self.rope_cos,
                    self.rope_sin,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                present = None
            else:
                x, present = block(
                    x,
                    self.rope_cos,
                    self.rope_sin,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            if next_past_key_values is not None:
                if present is None:
                    raise RuntimeError("block did not return cache")
                next_past_key_values.append(present)
        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states if return_hidden_states else None,
            past_key_values=tuple(next_past_key_values) if next_past_key_values is not None else None,
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_id: int | None = None,
        use_cache: bool = True,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        past_key_values = None
        for _ in range(max_new_tokens):
            if use_cache and past_key_values is not None and past_key_values[0][0].size(-2) < self.config.context_length:
                idx_cond = input_ids[:, -1:]
            else:
                past_key_values = None
                idx_cond = input_ids[:, -self.config.context_length :]
            out = self(idx_cond, past_key_values=past_key_values, use_cache=use_cache)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, input_ids, penalty=repetition_penalty)
            logits = filter_logits(logits, top_k=top_k, top_p=top_p, min_p=min_p)
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and torch.all(next_id.eq(eos_id)):
                break
        return input_ids

    def num_parameters(self, *, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.tie_embeddings:
            total -= self.embed.weight.numel()
        return total


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, *, penalty: float) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, vocab], got {tuple(logits.shape)}")
    if input_ids.ndim != 2 or input_ids.size(0) != logits.size(0):
        raise ValueError("input_ids must have shape [batch, seq] and share the logits batch size")
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    adjusted = logits.clone()
    for batch_index in range(input_ids.size(0)):
        seen = torch.unique(input_ids[batch_index])
        seen_scores = adjusted[batch_index, seen]
        adjusted[batch_index, seen] = torch.where(seen_scores < 0, seen_scores * penalty, seen_scores / penalty)
    return adjusted


def filter_logits(
    logits: torch.Tensor, *, top_k: int | None, top_p: float, min_p: float = 0.0
) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1]
        logits = logits.masked_fill(logits < kth[:, None], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        threshold = probs.max(dim=-1, keepdim=True).values * min_p
        remove = probs < threshold
        max_indices = probs.argmax(dim=-1, keepdim=True)
        remove.scatter_(1, max_indices, False)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits
