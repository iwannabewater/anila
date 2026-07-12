from __future__ import annotations

import math
from collections.abc import Iterator
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
    ce_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    past_key_values: KVCache | None = None


@dataclass
class GenerationStep:
    token_ids: torch.Tensor
    token_logprobs: torch.Tensor
    sequences: torch.Tensor
    finished: torch.Tensor


@dataclass
class _Beam:
    input_ids: torch.Tensor
    score: float
    ended: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight).type_as(x)


def precompute_rope(
    head_dim: int,
    seq_len: int,
    base: float,
    *,
    rope_scaling: str | None = None,
    rope_scaling_factor: float = 1.0,
    original_context_length: int | None = None,
    yarn_beta_fast: float = 32.0,
    yarn_beta_slow: float = 1.0,
    yarn_attention_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if isinstance(base, bool) or not isinstance(base, int | float) or base <= 0 or not math.isfinite(base):
        raise ValueError("RoPE base must be a positive number")
    if rope_scaling not in (None, "yarn"):
        raise ValueError("RoPE scaling must be null or 'yarn'")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    attention_factor = 1.0
    if rope_scaling == "yarn":
        if original_context_length is None:
            raise ValueError("YaRN scaling requires original_context_length")
        if (
            isinstance(original_context_length, bool)
            or not isinstance(original_context_length, int)
            or original_context_length <= 0
        ):
            raise ValueError("YaRN original_context_length must be a positive integer")
        for name, value in (
            ("rope_scaling_factor", rope_scaling_factor),
            ("yarn_beta_fast", yarn_beta_fast),
            ("yarn_beta_slow", yarn_beta_slow),
            ("yarn_attention_factor", yarn_attention_factor),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or value <= 0
                or not math.isfinite(value)
            ):
                raise ValueError(f"YaRN {name} must be a positive number")
        if rope_scaling_factor <= 1.0:
            raise ValueError("YaRN rope_scaling_factor must be greater than 1")
        if yarn_beta_fast < yarn_beta_slow:
            raise ValueError("YaRN yarn_beta_fast must be greater than or equal to yarn_beta_slow")
        attention_factor = yarn_attention_factor
        if seq_len > original_context_length:
            low, high = _yarn_correction_range(
                head_dim=head_dim,
                original_context_length=original_context_length,
                base=base,
                beta_fast=yarn_beta_fast,
                beta_slow=yarn_beta_slow,
            )
            freq_index = torch.arange(head_dim // 2).float()
            ramp = torch.clamp((freq_index - low) / max(high - low, 0.001), min=0.0, max=1.0)
            inv_freq = inv_freq * (1.0 - ramp + ramp / rope_scaling_factor)
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * attention_factor
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1) * attention_factor
    return cos, sin


def _yarn_correction_range(
    *,
    head_dim: int,
    original_context_length: int,
    base: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[int, int]:
    def correction_dim(beta: float) -> float:
        return head_dim * math.log(original_context_length / (beta * 2.0 * math.pi)) / (2.0 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), head_dim // 2 - 1)
    return low, high


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
        if config.n_kv_head is None:
            raise RuntimeError("validated model config must define n_kv_head")
        self.n_kv_head = config.n_kv_head
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


def swiglu_hidden_size(config: ModelConfig, override: int | None = None) -> int:
    if override is not None:
        return override
    hidden = int(8 * config.n_embd / 3)
    return 64 * math.ceil(hidden / 64)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig, *, hidden: int | None = None):
        super().__init__()
        hidden = swiglu_hidden_size(config, override=hidden)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RoutedSwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.moe_num_experts <= 1:
            raise ValueError("RoutedSwiGLU requires at least two experts")
        hidden = swiglu_hidden_size(config, override=config.moe_intermediate_size)
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.normalize_top_k = config.moe_normalize_top_k
        self.aux_loss_coef = float(config.moe_aux_loss_coef)
        self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config, hidden=hidden) for _ in range(self.num_experts)])
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embd = x.shape
        x_flat = x.reshape(batch * seq_len, embd)
        router_probs = F.softmax(self.router(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1, sorted=False)
        if self.normalize_top_k:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_weight.dtype).eps)

        y = torch.zeros_like(x_flat)
        for expert_index, expert in enumerate(self.experts):
            selected = topk_idx.eq(expert_index)
            if not torch.any(selected):
                continue
            token_idx = torch.nonzero(selected.any(dim=-1), as_tuple=False).flatten()
            weight = topk_weight[selected].unsqueeze(-1)
            y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight.to(dtype=x_flat.dtype))

        if self.training and self.aux_loss_coef > 0:
            load = F.one_hot(topk_idx, num_classes=self.num_experts).float().mean(dim=(0, 1))
            importance = router_probs.float().mean(dim=0)
            self.aux_loss = (load * importance).sum() * self.num_experts * self.aux_loss_coef
        else:
            self.aux_loss = x_flat.new_zeros(())
        return y.view(batch, seq_len, embd)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = RoutedSwiGLU(config) if config.moe_num_experts > 0 else SwiGLU(config)

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
        cos, sin = precompute_rope(
            self.config.n_embd // self.config.n_head,
            self.config.context_length,
            self.config.rope_base,
            rope_scaling=self.config.rope_scaling,
            rope_scaling_factor=self.config.rope_scaling_factor,
            original_context_length=self.config.rope_original_context_length,
            yarn_beta_fast=self.config.rope_yarn_beta_fast,
            yarn_beta_slow=self.config.rope_yarn_beta_slow,
            yarn_attention_factor=self.config.rope_yarn_attention_factor,
        )
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
        logits_to_keep: int = 0,
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
        if isinstance(logits_to_keep, bool) or not isinstance(logits_to_keep, int):
            raise ValueError("logits_to_keep must be a non-negative integer")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be non-negative")
        if logits_to_keep > input_ids.size(1):
            raise ValueError("logits_to_keep cannot exceed the current input sequence length")
        if targets is not None and logits_to_keep:
            raise ValueError("logits_to_keep cannot be used with targets")
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
        logit_hidden_states = hidden_states[:, -logits_to_keep:] if logits_to_keep else hidden_states
        logits = self.lm_head(logit_hidden_states)
        aux_loss = self._router_aux_loss(hidden_states)
        loss = None
        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            loss = ce_loss if aux_loss is None else ce_loss + aux_loss
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            aux_loss=aux_loss,
            hidden_states=hidden_states if return_hidden_states else None,
            past_key_values=tuple(next_past_key_values) if next_past_key_values is not None else None,
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def _router_aux_loss(self, reference: torch.Tensor) -> torch.Tensor | None:
        aux_losses = [
            block.mlp.aux_loss
            for block in self.blocks
            if isinstance(block.mlp, RoutedSwiGLU) and block.mlp.aux_loss is not None
        ]
        if not aux_losses:
            return None
        return sum(
            (loss.to(device=reference.device, dtype=torch.float32) for loss in aux_losses),
            reference.new_zeros((), dtype=torch.float32),
        )

    @staticmethod
    def _validate_generation_args(
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        num_beams: int,
        length_penalty: float,
    ) -> None:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise ValueError("temperature must be a finite positive number")
        if temperature <= 0 or not math.isfinite(float(temperature)):
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int)):
            raise ValueError("top_k must be a positive integer when provided")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if isinstance(top_p, bool) or not isinstance(top_p, int | float):
            raise ValueError("top_p must be a finite number in (0, 1]")
        if not 0.0 < top_p <= 1.0 or not math.isfinite(float(top_p)):
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(min_p, bool) or not isinstance(min_p, int | float):
            raise ValueError("min_p must be a finite number in [0, 1]")
        if not 0.0 <= min_p <= 1.0 or not math.isfinite(float(min_p)):
            raise ValueError("min_p must be in [0, 1]")
        if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, int | float):
            raise ValueError("repetition_penalty must be a finite positive number")
        if repetition_penalty <= 0 or not math.isfinite(float(repetition_penalty)):
            raise ValueError("repetition_penalty must be finite and positive")
        if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams <= 0:
            raise ValueError("num_beams must be a positive integer")
        if isinstance(length_penalty, bool) or not isinstance(length_penalty, int | float):
            raise ValueError("length_penalty must be a finite non-negative number")
        if length_penalty < 0 or not math.isfinite(float(length_penalty)):
            raise ValueError("length_penalty must be finite and non-negative")

    def generate_steps(
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
    ) -> Iterator[GenerationStep]:
        self._validate_generation_args(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            num_beams=1,
            length_penalty=1.0,
        )
        past_key_values = None
        finished = torch.zeros((input_ids.size(0), 1), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            with torch.inference_mode():
                cache_has_room = (
                    use_cache
                    and past_key_values is not None
                    and past_key_values[0][0].size(-2) < self.config.context_length
                )
                if cache_has_room:
                    idx_cond = input_ids[:, -1:]
                else:
                    past_key_values = None
                    idx_cond = input_ids[:, -self.config.context_length :]
                out = self(idx_cond, past_key_values=past_key_values, use_cache=use_cache, logits_to_keep=1)
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
                token_logprobs = F.log_softmax(logits, dim=-1).gather(1, next_id)
                already_finished = finished
                if eos_id is not None:
                    next_id = torch.where(already_finished, torch.full_like(next_id, eos_id), next_id)
                    token_logprobs = torch.where(already_finished, torch.zeros_like(token_logprobs), token_logprobs)
                    finished = finished | next_id.eq(eos_id)
                input_ids = torch.cat([input_ids, next_id], dim=1)
                step = GenerationStep(
                    token_ids=next_id,
                    token_logprobs=token_logprobs,
                    sequences=input_ids,
                    finished=finished.squeeze(1),
                )
            yield step
            if eos_id is not None and torch.all(finished):
                break

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
        num_beams: int = 1,
        length_penalty: float = 1.0,
        eos_id: int | None = None,
        use_cache: bool = True,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self._validate_generation_args(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        if num_beams > 1:
            return self._generate_beam_search(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                length_penalty=length_penalty,
                eos_id=eos_id,
            )
        for step in self.generate_steps(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            eos_id=eos_id,
            use_cache=use_cache,
            do_sample=do_sample,
            generator=generator,
        ):
            input_ids = step.sequences
        return input_ids

    def _generate_beam_search(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        num_beams: int,
        length_penalty: float,
        eos_id: int | None,
    ) -> torch.Tensor:
        if input_ids.size(0) != 1:
            raise ValueError("beam search currently supports batch_size=1")
        if max_new_tokens == 0:
            return input_ids
        prompt_len = input_ids.size(1)
        beams = [_Beam(input_ids=input_ids[0], score=0.0)]
        for _ in range(max_new_tokens):
            candidates: list[_Beam] = []
            for beam in beams:
                if beam.ended:
                    candidates.append(beam)
                    continue
                idx_cond = beam.input_ids[-self.config.context_length :].unsqueeze(0)
                logits = self(idx_cond, logits_to_keep=1).logits[:, -1, :] / temperature
                if repetition_penalty != 1.0:
                    logits = apply_repetition_penalty(logits, beam.input_ids.unsqueeze(0), penalty=repetition_penalty)
                logits = filter_logits(logits, top_k=top_k, top_p=top_p, min_p=min_p)
                log_probs = F.log_softmax(logits, dim=-1)
                k = min(num_beams, log_probs.size(-1))
                next_scores, next_ids = torch.topk(log_probs[0], k=k)
                for next_score, next_id in zip(next_scores, next_ids, strict=True):
                    token = next_id.view(1)
                    candidates.append(
                        _Beam(
                            input_ids=torch.cat((beam.input_ids, token)),
                            score=beam.score + float(next_score.item()),
                            ended=eos_id is not None and int(next_id.item()) == eos_id,
                        )
                    )
            beams = sorted(
                candidates,
                key=lambda beam: _normalized_beam_score(beam, prompt_len, length_penalty),
                reverse=True,
            )[:num_beams]
            if all(beam.ended for beam in beams):
                break
        best = max(beams, key=lambda beam: _normalized_beam_score(beam, prompt_len, length_penalty))
        return best.input_ids.unsqueeze(0)

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


def _normalized_beam_score(beam: _Beam, prompt_len: int, length_penalty: float) -> float:
    generated_len = max(int(beam.input_ids.numel()) - prompt_len, 1)
    if length_penalty == 0:
        return beam.score
    return beam.score / (generated_len**length_penalty)


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
