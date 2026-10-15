"""Uncompressed source-token KV as a separate, learned episodic-memory side path.

This is deliberately *not* native causal continuation: application positions
restart at zero and the source is available only via dedicated readers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn

from memwmlm.models.lora import ToggleableLoRALinear, replace_qwen_projection
from memwmlm.models.injection_indices import resolve_injection_indices


@dataclass(frozen=True)
class EpisodicConfig:
    injection_layers: int = 4
    reader_heads: int = 8
    gate_bias_init: float = -6.0
    lora_rank: int = 0
    lora_alpha: float = 16.0
    injection_indices: tuple[int, ...] | None = None


class KVReader(nn.Module):
    """Read every source position's native K/V, without a pooling writer."""

    def __init__(self, hidden: int, kv_dim: int, heads: int, gate_bias: float) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden size must be divisible by reader_heads")
        self.key = nn.Linear(kv_dim, hidden, bias=False)
        self.value = nn.Linear(kv_dim, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Parameter(torch.tensor(gate_bias))

    def forward(self, hidden: Tensor, native_keys: Tensor, native_values: Tensor) -> Tensor:
        # Native K/V: [batch, kv_heads, source_length, head_dim]. Preserve every
        # position; flatten grouped KV heads into a feature dimension per token.
        keys = native_keys.transpose(1, 2).reshape(native_keys.shape[0], native_keys.shape[2], -1)
        values = native_values.transpose(1, 2).reshape(native_values.shape[0], native_values.shape[2], -1)
        read, _ = self.attention(self.query(hidden), self.key(keys), self.value(values), need_weights=False)
        return torch.sigmoid(self.gate) * self.output(read)


class FrozenQwenWithEpisodicKV(nn.Module):
    def __init__(self, qwen: nn.Module, config: EpisodicConfig) -> None:
        super().__init__()
        self.qwen = qwen
        self.config = config
        for parameter in qwen.parameters():
            parameter.requires_grad_(False)
        qwen.eval()
        layers = qwen.model.layers
        self.indices = resolve_injection_indices(
            len(layers), config.injection_layers, config.injection_indices
        )
        kv_dim = int(qwen.config.num_key_value_heads) * int(qwen.config.head_dim)
        hidden = int(qwen.config.hidden_size)
        self.readers = nn.ModuleDict({str(i): KVReader(hidden, kv_dim, config.reader_heads, config.gate_bias_init)
                                      for i in self.indices})
        parameter = next(qwen.parameters())
        self.readers.to(device=parameter.device, dtype=parameter.dtype)
        self.projections: list[ToggleableLoRALinear] = []
        if config.lora_rank:
            for i in self.indices:
                for name in ("q_proj", "o_proj"):
                    self.projections.append(replace_qwen_projection(
                        layers[i].self_attn, name, config.lora_rank, config.lora_alpha, 0.0
                    ))
        self._active: tuple[tuple[Tensor, Tensor], ...] | None = None
        self.handles = [layers[i].register_forward_hook(self._hook(i)) for i in self.indices]

    def train(self, mode: bool = True) -> "FrozenQwenWithEpisodicKV":
        super().train(mode)
        self.qwen.eval()
        for module in self.projections:
            module.train(mode)
        return self

    @property
    def device(self) -> torch.device:
        return self.qwen.get_input_embeddings().weight.device

    @torch.no_grad()
    def encode(self, source_ids: Tensor) -> tuple[tuple[Tensor, Tensor], ...]:
        """Snapshot all source-token layer K/V in the native cache format."""
        if source_ids.shape[0] != 1:
            raise ValueError("Episodic v1 supports one unpadded source per forward")
        # Source LoRA and reader must be inactive. This stores per-example data,
        # not trainable model parameters or an optimizer state.
        output = self.qwen(input_ids=source_ids, use_cache=True, return_dict=True)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Qwen source forward returned no KV cache")
        if hasattr(cache, "layers"):
            # Recent Transformers DynamicCache is backed by per-layer cache
            # objects; older releases provide tuple-like indexing.
            pairs = [(layer.keys, layer.values) if hasattr(layer, "keys") else layer
                     for layer in cache.layers]
        else:
            pairs = [cache[i] for i in range(len(self.qwen.model.layers))]
        if len(pairs) != len(self.qwen.model.layers):
            raise RuntimeError("Source cache does not have one KV pair per layer")
        for index, (keys, values) in enumerate(pairs):
            if keys is None or values is None or keys.ndim != 4 or values.ndim != 4:
                raise RuntimeError(f"Source cache layer {index} has no dense 4D K/V")
            if keys.shape[2] != source_ids.shape[1] or values.shape[2] != source_ids.shape[1]:
                raise RuntimeError(f"Source cache layer {index} did not retain every source token")
        return tuple((keys.detach().clone(), values.detach().clone()) for keys, values in pairs)

    @contextmanager
    def activate(self, memory: tuple[tuple[Tensor, Tensor], ...] | None,
                 application: bool = True) -> Iterator[None]:
        if self._active is not None:
            raise RuntimeError("Nested memory scopes are not supported")
        self._active = memory
        for projection in self.projections:
            projection.mode = "application" if application else "none"
        try:
            yield
        finally:
            for projection in self.projections:
                projection.mode = "none"
            self._active = None

    def forward(self, input_ids: Tensor, memory: tuple[tuple[Tensor, Tensor], ...] | None,
                **kwargs: object) -> object:
        with self.activate(memory, application=True):
            return self.qwen(input_ids=input_ids, **kwargs)

    def _hook(self, index: int):
        reader = self.readers[str(index)]

        def inject(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            if self._active is None:
                return output
            keys, values = self._active[index]
            if isinstance(output, tuple):
                return (output[0] + reader(output[0], keys, values), *output[1:])
            if isinstance(output, Tensor):
                return output + reader(output, keys, values)
            raise TypeError(f"Unexpected decoder output: {type(output)!r}")

        return inject

    def trainable_state_dict(self) -> dict[str, Tensor]:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in names}

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        required = {name for name, p in self.named_parameters() if p.requires_grad}
        if unexpected or required.intersection(missing):
            raise RuntimeError(f"Invalid episodic checkpoint: missing={required.intersection(missing)}, unexpected={unexpected}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
