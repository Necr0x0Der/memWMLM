"""Minimal runtime-switchable LoRA for the memory-interface experiment."""

from __future__ import annotations

from torch import Tensor, nn


class ToggleableLoRALinear(nn.Module):
    """Frozen projection with independently trainable source/application LoRA.

    The two adapters let source encoding become write-friendly without forcing
    it to share the application-side interface adaptation. Existing application
    checkpoints retain the `lora_down` / `lora_up` parameter names.
    """

    def __init__(
        self,
        base: nn.Linear,
        application_rank: int,
        application_alpha: float,
        application_dropout: float,
        source_rank: int = 0,
        source_alpha: float = 16.0,
        source_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if application_rank < 0 or source_rank < 0:
            raise ValueError("LoRA ranks must be non-negative")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_down, self.lora_up, self.scale, self.dropout = self._make_adapter(
            application_rank, application_alpha, application_dropout, "application"
        )
        self.source_lora_down, self.source_lora_up, self.source_scale, self.source_dropout = self._make_adapter(
            source_rank, source_alpha, source_dropout, "source"
        )
        self.mode = "none"

    def _make_adapter(
        self, rank: int, alpha: float, dropout: float, name: str
    ) -> tuple[nn.Linear | None, nn.Linear | None, float, nn.Dropout]:
        if rank == 0:
            return None, None, 1.0, nn.Dropout(0.0)
        if alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError(f"{name} LoRA requires alpha > 0 and dropout in [0, 1)")
        down = nn.Linear(self.base.in_features, rank, bias=False)
        up = nn.Linear(rank, self.base.out_features, bias=False)
        down.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        up.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        nn.init.kaiming_uniform_(down.weight, a=5**0.5)
        nn.init.zeros_(up.weight)
        return down, up, alpha / rank, nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self.mode == "application" and self.lora_down is not None and self.lora_up is not None:
            return output + self.lora_up(self.lora_down(self.dropout(inputs))) * self.scale
        if self.mode == "source" and self.source_lora_down is not None and self.source_lora_up is not None:
            return output + self.source_lora_up(self.source_lora_down(self.source_dropout(inputs))) * self.source_scale
        return output


def replace_qwen_projection(
    attention: nn.Module,
    name: str,
    application_rank: int,
    application_alpha: float,
    application_dropout: float,
    source_rank: int = 0,
    source_alpha: float = 16.0,
    source_dropout: float = 0.0,
) -> ToggleableLoRALinear:
    """Replace one dense Qwen attention projection with a toggleable LoRA path."""

    projection = getattr(attention, name, None)
    if not isinstance(projection, nn.Linear):
        raise TypeError(f"Expected attention.{name} to be nn.Linear, got {type(projection)!r}")
    wrapped = ToggleableLoRALinear(
        projection,
        application_rank=application_rank,
        application_alpha=application_alpha,
        application_dropout=application_dropout,
        source_rank=source_rank,
        source_alpha=source_alpha,
        source_dropout=source_dropout,
    )
    setattr(attention, name, wrapped)
    return wrapped
