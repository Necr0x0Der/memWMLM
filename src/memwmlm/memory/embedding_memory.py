"""Resettable writable embedding memory used by the first sanity check."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MemoryState:
    """Per-example runtime memory; never registered as an ``nn.Parameter``."""

    keys: Tensor  # [batch, cells, memory_width]
    values: Tensor  # [batch, cells, memory_width]


class SourceWriter(nn.Module):
    """Compress frozen source hidden states into a fresh pool of memory cells."""

    def __init__(self, hidden_size: int, memory_width: int, cells: int, heads: int) -> None:
        super().__init__()
        if memory_width % heads:
            raise ValueError("memory_width must be divisible by heads")
        self.queries = nn.Parameter(torch.empty(1, cells, memory_width))
        self.source_projection = nn.Linear(hidden_size, memory_width, bias=False)
        self.attention = nn.MultiheadAttention(memory_width, heads, batch_first=True)
        self.norm = nn.LayerNorm(memory_width)
        self.mlp = nn.Sequential(
            nn.Linear(memory_width, 2 * memory_width),
            nn.SiLU(),
            nn.Linear(2 * memory_width, memory_width),
        )
        self.key_projection = nn.Linear(memory_width, memory_width, bias=False)
        self.value_projection = nn.Linear(memory_width, memory_width, bias=False)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, source_states: Tensor, source_mask: Tensor) -> MemoryState:
        """Write a distinct memory state for every source in a batch."""

        batch = source_states.shape[0]
        source = self.source_projection(source_states)
        queries = self.queries.expand(batch, -1, -1)
        pooled, _ = self.attention(
            queries,
            source,
            source,
            key_padding_mask=~source_mask.bool(),
            need_weights=False,
        )
        cells = self.norm(queries + pooled)
        cells = cells + self.mlp(cells)
        return MemoryState(keys=self.key_projection(cells), values=self.value_projection(cells))


class MemoryReader(nn.Module):
    """Query resettable memory cells from one axial hidden-state stream."""

    def __init__(
        self,
        hidden_size: int,
        memory_width: int,
        heads: int,
        gate_bias_init: float = -6.0,
    ) -> None:
        super().__init__()
        if memory_width % heads:
            raise ValueError("memory_width must be divisible by heads")
        self.query_projection = nn.Linear(hidden_size, memory_width, bias=False)
        self.attention = nn.MultiheadAttention(memory_width, heads, batch_first=True)
        self.output_projection = nn.Linear(memory_width, hidden_size, bias=False)
        # Start as almost-identical frozen Qwen; sigmoid(-6) ≈ 0.0025.
        self.gate_bias = nn.Parameter(torch.tensor(gate_bias_init))
        self.gate_projection = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.gate_projection.weight)

    def forward(self, hidden_states: Tensor, memory: MemoryState) -> Tensor:
        queries = self.query_projection(hidden_states)
        read, _ = self.attention(queries, memory.keys, memory.values, need_weights=False)
        gate = torch.sigmoid(self.gate_bias + self.gate_projection(hidden_states))
        return gate * self.output_projection(read)
