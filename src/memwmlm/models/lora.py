"""Minimal runtime-switchable LoRA for the memory-interface experiment."""

from __future__ import annotations

from torch import Tensor, nn


class ToggleableLoRALinear(nn.Module):
    """Frozen linear projection plus a LoRA delta enabled only by its owner.

    The base projection remains frozen. ``enabled`` is a runtime switch rather
    than a learned gate: source encoding keeps it false, while the application
    pass enables it. This preserves the application-only LoRA ablation.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.lora_down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.enabled = False

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self.enabled:
            output = output + self.lora_up(self.lora_down(self.dropout(inputs))) * self.scale
        return output


def replace_qwen_projection(
    attention: nn.Module,
    name: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> ToggleableLoRALinear:
    """Replace one dense Qwen attention projection with a toggleable LoRA path."""

    projection = getattr(attention, name, None)
    if not isinstance(projection, nn.Linear):
        raise TypeError(f"Expected attention.{name} to be nn.Linear, got {type(projection)!r}")
    wrapped = ToggleableLoRALinear(projection, rank=rank, alpha=alpha, dropout=dropout)
    setattr(attention, name, wrapped)
    return wrapped
