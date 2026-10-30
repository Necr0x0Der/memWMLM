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
    mask: Tensor | None = None  # [batch, cells], True = padded/ignored position (optional)


class SourceWriter(nn.Module):
    """Compress frozen source hidden states into position-aligned memory cells.

    Each source position becomes one cell (previously the whole source was pooled
    into a fixed number of concept cells via learned ``queries``). Cells are no
    longer a position-agnostic bag: cell i summarises source position i, so the
    reader can retrieve by position (reproduction) and by content (recall/QA).
    """

    def __init__(self, hidden_size: int, memory_width: int, heads: int, window: int | None = None) -> None:
        super().__init__()
        if memory_width % heads:
            raise ValueError("memory_width must be divisible by heads")
        self.source_projection = nn.Linear(hidden_size, memory_width, bias=False)
        # window=None -> one cell per source position (purely positional).
        # window=W    -> each cell also sees a causal neighbourhood of W positions.
        self.window = window
        if window:
            self.attention = nn.MultiheadAttention(memory_width, heads, batch_first=True)
        self.norm = nn.LayerNorm(memory_width)
        self.mlp = nn.Sequential(
            nn.Linear(memory_width, 2 * memory_width),
            nn.SiLU(),
            nn.Linear(2 * memory_width, memory_width),
        )
        self.key_projection = nn.Linear(memory_width, memory_width, bias=False)
        self.value_projection = nn.Linear(memory_width, memory_width, bias=False)

    def forward(self, source_states: Tensor, source_mask: Tensor) -> MemoryState:
        """Write one memory cell per source position (a distinct state per example)."""

        projected = self.source_projection(source_states)  # [batch, src_len, memory_width]
        if self.window:
            src_len = projected.shape[1]
            causal_mask = torch.triu(
                torch.full((src_len, src_len), float("-inf"), device=projected.device),
                k=1,
            )
            windowed, _ = self.attention(
                projected, projected, projected,
                key_padding_mask=~source_mask.bool(),
                attn_mask=causal_mask,
                need_weights=False,
            )
            cells = self.norm(projected + windowed)
        else:
            cells = self.norm(projected)
        cells = cells + self.mlp(cells)
        # Pad positions become neutral cells; the mask travels with the state so
        # the reader can exclude them (mirrors how the old pooled writer masked).
        pad_mask = ~source_mask.bool()
        cells = cells * source_mask.unsqueeze(-1).to(cells.dtype)
        return MemoryState(
            keys=self.key_projection(cells),
            values=self.value_projection(cells),
            mask=pad_mask,
        )


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
        read, _ = self.attention(
            queries, memory.keys, memory.values,
            key_padding_mask=memory.mask,
            need_weights=False,
        )
        gate = torch.sigmoid(self.gate_bias + self.gate_projection(hidden_states))
        return gate * self.output_projection(read)
