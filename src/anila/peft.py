from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from anila.config import LoRAConfig


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def apply_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    cfg = config.validated()
    if not cfg.enabled:
        return []
    target_names = set(cfg.target_modules)
    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout),
                )
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                replaced.append(full_name)
    if not replaced:
        raise ValueError(f"LoRA target modules were not found: {', '.join(sorted(target_names))}")
    return replaced


def mark_lora_trainable(model: nn.Module, *, train_base: bool, train_bias: bool) -> None:
    for name, param in model.named_parameters():
        is_lora = ".lora_a." in name or ".lora_b." in name
        is_bias = name.endswith(".bias")
        param.requires_grad = train_base or is_lora or (train_bias and is_bias)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def merge_lora(model: nn.Module) -> list[str]:
    merged: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            setattr(module, child_name, _merged_linear(child))
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            merged.append(full_name)
    if not merged:
        raise ValueError("No LoRA modules were found to merge")
    return merged


def _merged_linear(layer: LoRALinear) -> nn.Linear:
    base = layer.base
    merged = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    delta = layer.lora_b.weight @ layer.lora_a.weight
    with torch.no_grad():
        merged.weight.copy_(base.weight + delta.mul(layer.scaling).to(dtype=base.weight.dtype))
        if base.bias is not None and merged.bias is not None:
            merged.bias.copy_(base.bias)
    return merged


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def adapt_state_dict_for_lora(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_keys = set(model.state_dict())
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        mapped_key = _map_key_for_lora(key, target_keys)
        if mapped_key is not None:
            adapted[mapped_key] = value
    return adapted


def _map_key_for_lora(key: str, target_keys: Iterable[str]) -> str | None:
    if key in target_keys:
        return key
    target_key_set = target_keys if isinstance(target_keys, set) else set(target_keys)
    for suffix in (".weight", ".bias"):
        if key.endswith(suffix):
            mapped = f"{key[: -len(suffix)]}.base{suffix}"
            if mapped in target_key_set:
                return mapped
    return None
