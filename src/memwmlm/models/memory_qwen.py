"""Frozen Qwen3 wrapper with resettable memory and optional interface LoRA."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

from memwmlm.memory.embedding_memory import MemoryReader, MemoryState, SourceWriter
from memwmlm.models.lora import ToggleableLoRALinear, replace_qwen_projection
from memwmlm.models.injection_indices import resolve_injection_indices


@dataclass(frozen=True)
class MemoryQwenConfig:
    memory_width: int = 512
    writer_heads: int = 8
    reader_heads: int = 8
    injection_layers: int = 4
    gate_bias_init: float = -6.0
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    source_lora_rank: int = 0
    source_lora_alpha: float = 16.0
    source_lora_dropout: float = 0.0
    injection_indices: tuple[int, ...] | None = None

    @classmethod
    def from_checkpoint(cls, memory_config: dict[str, object]) -> "MemoryQwenConfig":
        """Build from a checkpoint's ``memory_config``, ignoring keys that were since
        removed (the vestigial ``memory_cells``) so older checkpoints hydrate cleanly."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in memory_config.items() if k in known})


class FrozenQwenWithMemory(nn.Module):
    """Frozen axial Qwen with source-writer and application-reader paths.

    LoRA may be installed in selected upper attention projections. Source and
    application adapters are distinct and explicitly scoped: source LoRA makes
    committed hidden states write-friendly; application LoRA adapts memory use.
    Base Qwen weights are always frozen.
    """

    def __init__(self, qwen: nn.Module, config: MemoryQwenConfig) -> None:
        super().__init__()
        self.qwen = qwen
        self.config = config
        hidden_size = int(qwen.config.hidden_size)
        layers = _qwen_layers(qwen)
        layer_indices = resolve_injection_indices(
            len(layers), config.injection_layers, config.injection_indices
        )

        for parameter in self.qwen.parameters():
            parameter.requires_grad_(False)
        self.qwen.eval()

        self.writer = SourceWriter(
            hidden_size=hidden_size,
            memory_width=config.memory_width,
            heads=config.writer_heads,
            window=None,
        )
        self.layer_indices = layer_indices
        self.readers = nn.ModuleDict(
            {
                str(index): MemoryReader(
                    hidden_size=hidden_size,
                    memory_width=config.memory_width,
                    heads=config.reader_heads,
                    gate_bias_init=config.gate_bias_init,
                )
                for index in self.layer_indices
            }
        )
        self.lora_projections: list[ToggleableLoRALinear] = []
        if config.lora_rank or config.source_lora_rank:
            self.lora_projections = _install_lora(
                layers,
                self.layer_indices,
                application_rank=config.lora_rank,
                application_alpha=config.lora_alpha,
                application_dropout=config.lora_dropout,
                source_rank=config.source_lora_rank,
                source_alpha=config.source_lora_alpha,
                source_dropout=config.source_lora_dropout,
            )

        axial_parameter = next(self.qwen.parameters())
        self.writer.to(device=axial_parameter.device, dtype=axial_parameter.dtype)
        self.readers.to(device=axial_parameter.device, dtype=axial_parameter.dtype)
        self._active_memory: MemoryState | None = None
        self._lora_mode = "none"
        self._handles = [
            layers[index].register_forward_hook(self._make_injection_hook(index))
            for index in self.layer_indices
        ]

    def train(self, mode: bool = True) -> "FrozenQwenWithMemory":
        """Keep frozen Qwen in eval mode while sidecar/LoRA follow train mode."""

        super().train(mode)
        self.qwen.eval()
        for projection in self.lora_projections:
            projection.train(mode)
        return self

    @property
    def device(self) -> torch.device:
        return self.qwen.get_input_embeddings().weight.device

    def lora_summary(self) -> dict[str, object]:
        """Return serializable adapter counts and scopes for experiment logs."""

        parameter_map = dict(self.named_parameters())
        modes = {
            "application": (".lora_", self.config.lora_rank, self.config.lora_alpha, self.config.lora_dropout),
            "source": (
                ".source_lora_",
                self.config.source_lora_rank,
                self.config.source_lora_alpha,
                self.config.source_lora_dropout,
            ),
        }
        result: dict[str, object] = {
            "enabled": bool(self.lora_projections),
            "target_layers": list(self.layer_indices) if self.lora_projections else [],
            "target_projections": ["q_proj", "o_proj"] if self.lora_projections else [],
            "projection_modules": len(self.lora_projections),
        }
        for mode, (marker, rank, alpha, dropout) in modes.items():
            names = [
                name for name, parameter in parameter_map.items()
                if parameter.requires_grad and marker in name
            ]
            result[mode] = {
                "enabled": bool(names),
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "trainable_parameters": sum(parameter_map[name].numel() for name in names),
                "trainable_parameter_tensors": len(names),
            }
        return result

    def trainable_parameter_summary(self) -> dict[str, int]:
        """Report aggregate memory-sidecar and LoRA parameter counts."""

        total = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        application = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and ".lora_" in name and ".source_lora_" not in name
        )
        source = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and ".source_lora_" in name
        )
        return {
            "total": total,
            "application_lora": application,
            "source_lora": source,
            "memory_sidecar": total - application - source,
        }

    def write_memory(self, source_input_ids: Tensor, source_attention_mask: Tensor) -> MemoryState:
        """Commit source into runtime memory, optionally training source LoRA."""

        # With source LoRA, retain a graph through source adapter only; all base
        # Qwen parameters are frozen. Otherwise source states are constants.
        source_context = torch.enable_grad() if self.config.source_lora_rank else torch.no_grad()
        with source_context, self.read_memory(None), self.lora_mode("source"):
            outputs = self.qwen(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        source_states = outputs.hidden_states[-1]
        if not self.config.source_lora_rank:
            source_states = source_states.detach()
        return self.writer(source_states, source_attention_mask)

    @contextmanager
    def lora_mode(self, mode: str) -> Iterator[None]:
        """Select `source`, `application`, or `none` adapters for one Qwen pass."""

        if mode not in {"none", "source", "application"}:
            raise ValueError(f"Unknown LoRA mode: {mode}")
        if self._lora_mode != "none":
            raise RuntimeError("Nested LoRA scopes are not supported")
        self._lora_mode = mode
        for projection in self.lora_projections:
            projection.mode = mode
        try:
            yield
        finally:
            for projection in self.lora_projections:
                projection.mode = "none"
            self._lora_mode = "none"

    def application_lora(self, enabled: bool) -> Iterator[None]:
        """Compatibility helper for generation evaluation."""

        return self.lora_mode("application" if enabled else "none")

    @contextmanager
    def read_memory(self, memory: MemoryState | None) -> Iterator[None]:
        """Activate memory injection only for the enclosed Qwen forward pass."""

        if self._active_memory is not None:
            raise RuntimeError("Nested memory read scopes are not supported")
        self._active_memory = memory
        try:
            yield
        finally:
            self._active_memory = None

    def forward(self, *args: object, memory: MemoryState | None = None, **kwargs: object) -> object:
        with self.read_memory(memory), self.lora_mode("application" if memory is not None else "none"):
            return self.qwen(*args, **kwargs)

    def _make_injection_hook(self, layer_index: int):
        reader = self.readers[str(layer_index)]

        def inject(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            memory = self._active_memory
            if memory is None:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
                return (hidden + reader(hidden, memory), *output[1:])
            if isinstance(output, Tensor):
                return output + reader(output, memory)
            raise TypeError(f"Unexpected Qwen decoder-layer output: {type(output)!r}")

        return inject

    def trainable_state_dict(self) -> dict[str, Tensor]:
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in trainable}

    def load_trainable_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        missing = [name for name in missing if name in expected]
        if missing or unexpected:
            raise RuntimeError(f"Invalid memory checkpoint; missing={missing}, unexpected={unexpected}")

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _install_lora(
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    application_rank: int,
    application_alpha: float,
    application_dropout: float,
    source_rank: int,
    source_alpha: float,
    source_dropout: float,
) -> list[ToggleableLoRALinear]:
    """Attach scoped LoRA to q_proj/o_proj of selected Qwen attention layers."""

    projections: list[ToggleableLoRALinear] = []
    for index in layer_indices:
        attention = getattr(layers[index], "self_attn", None)
        if not isinstance(attention, nn.Module):
            raise TypeError(f"Expected Qwen layer {index} to expose self_attn")
        for name in ("q_proj", "o_proj"):
            projections.append(
                replace_qwen_projection(
                    attention,
                    name,
                    application_rank=application_rank,
                    application_alpha=application_alpha,
                    application_dropout=application_dropout,
                    source_rank=source_rank,
                    source_alpha=source_alpha,
                    source_dropout=source_dropout,
                )
            )
    return projections


def _qwen_layers(qwen: nn.Module) -> Sequence[nn.Module]:
    try:
        layers = qwen.model.layers
    except AttributeError as error:
        raise TypeError("Expected a dense Qwen CausalLM with model.layers") from error
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise TypeError("Qwen model.layers is not an ordered layer collection")
    return layers
