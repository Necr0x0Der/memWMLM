"""Frozen Qwen3 wrapper with temporary cross-attention to resettable memory."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

from memwmlm.memory.embedding_memory import MemoryReader, MemoryState, SourceWriter
from memwmlm.models.lora import ToggleableLoRALinear, replace_qwen_projection


@dataclass(frozen=True)
class MemoryQwenConfig:
    memory_width: int = 512
    memory_cells: int = 32
    writer_heads: int = 8
    reader_heads: int = 8
    injection_layers: int = 4
    gate_bias_init: float = -6.0
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0


class FrozenQwenWithMemory(nn.Module):
    """Keep axial Qwen frozen while exposing source-write and application-read paths.

    The module uses temporary forward hooks on selected decoder layers. Hooks are
    installed once, but are inactive unless a ``MemoryState`` is in scope. This
    keeps the unmodified Qwen forward path available for full-context controls.
    """

    def __init__(self, qwen: nn.Module, config: MemoryQwenConfig) -> None:
        super().__init__()
        self.qwen = qwen
        self.config = config
        hidden_size = int(qwen.config.hidden_size)
        layers = _qwen_layers(qwen)
        if not 0 < config.injection_layers <= len(layers):
            raise ValueError("injection_layers must be in [1, number of Qwen layers]")

        for parameter in self.qwen.parameters():
            parameter.requires_grad_(False)
        self.qwen.eval()

        self.writer = SourceWriter(
            hidden_size=hidden_size,
            memory_width=config.memory_width,
            cells=config.memory_cells,
            heads=config.writer_heads,
        )
        self.layer_indices = tuple(range(len(layers) - config.injection_layers, len(layers)))
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
        if config.lora_rank:
            self.lora_projections = _install_application_lora(
                layers,
                self.layer_indices,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )

        # With a normal single-device Qwen load, memory modules must follow the
        # input-embedding device. A sharded backbone is intentionally deferred.
        axial_parameter = next(self.qwen.parameters())
        self.writer.to(device=axial_parameter.device, dtype=axial_parameter.dtype)
        self.readers.to(device=axial_parameter.device, dtype=axial_parameter.dtype)
        self._active_memory: MemoryState | None = None
        self._application_lora_enabled = False
        self._handles = [
            layers[index].register_forward_hook(self._make_injection_hook(index))
            for index in self.layer_indices
        ]

    def train(self, mode: bool = True) -> "FrozenQwenWithMemory":
        """Keep frozen Qwen fixed while letting sidecar/LoRA follow train mode."""

        super().train(mode)
        self.qwen.eval()
        # qwen.eval() also visits LoRA modules because they replace projections.
        # Restore only their mode so optional LoRA dropout behaves as configured.
        for projection in self.lora_projections:
            projection.train(mode)
        return self

    @property
    def device(self) -> torch.device:
        return self.qwen.get_input_embeddings().weight.device

    def lora_summary(self) -> dict[str, object]:
        """Return a serializable assertion of installed interface-LoRA state."""

        trainable_names = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and ".lora_" in name
        ]
        trainable_parameters = sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if parameter.requires_grad and ".lora_" in name
        )
        return {
            "enabled": bool(self.lora_projections),
            "rank": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "dropout": self.config.lora_dropout,
            "target_layers": list(self.layer_indices) if self.lora_projections else [],
            "target_projections": ["q_proj", "o_proj"] if self.lora_projections else [],
            "projection_modules": len(self.lora_projections),
            "trainable_parameters": trainable_parameters,
            "trainable_parameter_tensors": len(trainable_names),
        }

    def trainable_parameter_summary(self) -> dict[str, int]:
        """Report trainable sidecar/LoRA parameter counts for experiment logs."""

        lora_parameters = sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if parameter.requires_grad and ".lora_" in name
        )
        total = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total": total,
            "lora": lora_parameters,
            "memory_sidecar": total - lora_parameters,
        }

    def write_memory(self, source_input_ids: Tensor, source_attention_mask: Tensor) -> MemoryState:
        """Run frozen source encoding and create per-example runtime memory."""

        # Application-only LoRA is explicitly disabled while committing source.
        with torch.no_grad(), self.read_memory(None), self.application_lora(False):
            outputs = self.qwen(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # Axial Qwen is frozen, but writer remains in the autograd graph so the
        # application loss can train the source-to-memory transformation.
        return self.writer(outputs.hidden_states[-1].detach(), source_attention_mask)

    @contextmanager
    def application_lora(self, enabled: bool) -> Iterator[None]:
        """Enable installed LoRA only for an application-model forward pass."""

        if self._application_lora_enabled:
            raise RuntimeError("Nested LoRA scopes are not supported")
        self._application_lora_enabled = enabled
        for projection in self.lora_projections:
            projection.enabled = enabled
        try:
            yield
        finally:
            for projection in self.lora_projections:
                projection.enabled = False
            self._application_lora_enabled = False

    @contextmanager
    def read_memory(self, memory: MemoryState | None) -> Iterator[None]:
        """Activate memory injection only for the enclosed axial forward pass."""

        if self._active_memory is not None:
            raise RuntimeError("Nested memory read scopes are not supported")
        self._active_memory = memory
        try:
            yield
        finally:
            self._active_memory = None

    def forward(self, *args: object, memory: MemoryState | None = None, **kwargs: object) -> object:
        # A caller using memory asks for the application path. Keep base-Qwen
        # controls memory=None and LoRA-free by construction.
        with self.read_memory(memory), self.application_lora(memory is not None):
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
        """Return only learned memory-sidecar weights, never frozen Qwen weights."""

        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable
        }

    def load_trainable_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load a sidecar-only checkpoint and reject missing/unexpected keys."""

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        missing = [name for name in missing if name in expected]
        if missing or unexpected:
            raise RuntimeError(f"Invalid memory checkpoint; missing={missing}, unexpected={unexpected}")

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _install_application_lora(
    layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[ToggleableLoRALinear]:
    """Attach LoRA to `q_proj` and `o_proj` in selected upper Qwen layers."""

    projections: list[ToggleableLoRALinear] = []
    for index in layer_indices:
        attention = getattr(layers[index], "self_attn", None)
        if not isinstance(attention, nn.Module):
            raise TypeError(f"Expected Qwen layer {index} to expose self_attn")
        for name in ("q_proj", "o_proj"):
            projections.append(replace_qwen_projection(attention, name, rank, alpha, dropout))
    return projections


def _qwen_layers(qwen: nn.Module) -> Sequence[nn.Module]:
    """Locate dense Qwen decoder layers and fail explicitly on unsupported models."""

    try:
        layers = qwen.model.layers
    except AttributeError as error:
        raise TypeError("Expected a dense Qwen CausalLM with model.layers") from error
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise TypeError("Qwen model.layers is not an ordered layer collection")
    return layers
