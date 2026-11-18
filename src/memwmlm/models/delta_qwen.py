"""Variant 1 of issue #6: delta-rule memory on Qwen3's *existing* attention heads.

Each selected Qwen3 attention layer is patched so that it can run in one of
three modes:

* ``off``     -- unmodified frozen Qwen attention;
* ``capture`` -- additionally record pre-RoPE, ``k_norm``-ed keys and values per
  KV head (used to write a passage bank into memory);
* ``read``    -- add ``gate * read(S, q)`` to every query head's attention output
  before ``o_proj``; ``q`` is the pre-RoPE ``q_norm``-ed query and ``S`` is the
  delta-rule state of that head's KV group.

Keys/queries are taken *before* RoPE so stored associations carry content, not
absolute position. No Qwen weight is modified; the memory state is runtime data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from memwmlm.memory import delta_rule


@dataclass
class DeltaMemoryConfig:
    rule: delta_rule.Rule = "delta"
    beta: float = 1.0
    alpha: float = 1.0
    gate: float = 1.0
    layers: tuple[int, ...] | None = None  # None = every layer
    skip_first_token: bool = True  # first position acts as an attention sink
    center_keys: bool = False  # subtract per-head mean key (calibration, no training)
    # also subtract the mean key from read queries, so query and key live in the same
    # (centred) space; needed for key_expand, where DPFP is not shift-equivariant
    center_queries: bool = False
    beta_norm: str = "none"  # "tokens": effective beta = beta / (tokens written into the state)
    key_expand: int = 0  # DPFP order nu (Schlag et al. 2021): key/query dim 128 -> 2*128*nu; 0 = off
    write_chunk: int = 256  # token chunk for expanded writes


def prepend_sink(input_ids: Tensor, attention_mask: Tensor, sink_id: int) -> tuple[Tensor, Tensor]:
    """Prepend one sink token to every right-padded passage.

    Qwen3's tokenizer adds no BOS, so with ``skip_first_token`` the first
    *content* token would be dropped from the write. With a sink token in
    front, the skipped position is that token and every content token is
    written.
    """

    column = torch.full_like(input_ids[:, :1], sink_id)
    ones = torch.ones_like(attention_mask[:, :1])
    return torch.cat([column, input_ids], dim=1), torch.cat([ones, attention_mask], dim=1)


def dpfp(x: Tensor, nu: int) -> Tensor:
    """Deterministic parameter-free projection: ``[..., D] -> [..., 2 D nu]`` (non-negative)."""

    x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
    return torch.cat([x * torch.roll(x, shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)


class _Capture:
    def __init__(self) -> None:
        self.keys: list[Tensor] = []  # per call: [B, Hkv, T, D]
        self.values: list[Tensor] = []
        self.queries: list[Tensor] = []  # per call: [B, H, T, D]


class DeltaMemoryQwen:
    """Attach delta-rule capture/read paths to a frozen Qwen3 causal LM."""

    def __init__(self, qwen: nn.Module, config: DeltaMemoryConfig) -> None:
        self.qwen = qwen
        self.config = config
        for parameter in qwen.parameters():
            parameter.requires_grad_(False)
        layers = qwen.model.layers
        indices = range(len(layers)) if config.layers is None else config.layers
        self.layer_indices = tuple(indices)
        text_config = qwen.config
        self.num_kv_heads = int(text_config.num_key_value_heads)
        self.num_heads = int(text_config.num_attention_heads)
        self.head_dim = int(getattr(text_config, "head_dim", text_config.hidden_size // self.num_heads))
        self.groups = self.num_heads // self.num_kv_heads
        self.key_dim = 2 * self.head_dim * config.key_expand if config.key_expand else self.head_dim

        self.mode = "off"
        self.capture_queries = False
        self.state: dict[int, Tensor] | None = None  # layer -> [Hkv, D, D]
        self.key_mean: dict[int, Tensor] | None = None  # layer -> [Hkv, D]
        self.gates: dict[int, Tensor | float] = {}
        self._captures: dict[int, _Capture] = {}
        self._originals: dict[int, Callable] = {}
        for index in self.layer_indices:
            attention = layers[index].self_attn
            self._originals[index] = attention.forward
            attention.forward = self._make_forward(index, attention)

    # ------------------------------------------------------------------ modes
    @contextmanager
    def capturing(self) -> Iterator[dict[int, _Capture]]:
        self._captures = {index: _Capture() for index in self.layer_indices}
        self.mode = "capture"
        try:
            yield self._captures
        finally:
            self.mode = "off"

    @contextmanager
    def reading(self, state: dict[int, Tensor] | None) -> Iterator[None]:
        """Read ``state`` during the enclosed passes; ``None`` = plain Qwen."""

        previous_state, previous_mode = self.state, self.mode
        self.state = state
        self.mode = "read" if state is not None else "off"
        try:
            yield
        finally:
            self.state, self.mode = previous_state, previous_mode

    # ------------------------------------------------------------------ write
    @torch.no_grad()
    def capture_passages(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a padded batch of passages independently; return stacked k, v, mask.

        Returns keys/values ``[L, Hkv, B*T, D]`` (L = selected layers) and a
        write mask ``[B*T]`` that excludes padding (and the first real token if
        ``skip_first_token``). Passages are right-padded and each starts at
        position 0, so no cross-passage positional binding occurs.
        """

        with self.capturing() as captures:
            self.qwen(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        mask = attention_mask.bool().clone()
        if self.config.skip_first_token:
            first = mask.float().argmax(dim=1)
            mask[torch.arange(mask.shape[0]), first] = False
        keys = torch.stack([torch.cat(captures[i].keys, dim=0) for i in self.layer_indices])
        values = torch.stack([torch.cat(captures[i].values, dim=0) for i in self.layer_indices])
        # [L, B, Hkv, T, D] -> [L, Hkv, B*T, D]
        L, B, H, T, D = keys.shape
        keys = keys.permute(0, 2, 1, 3, 4).reshape(L, H, B * T, D)
        values = values.permute(0, 2, 1, 3, 4).reshape(L, H, B * T, D)
        return keys, values, mask.reshape(B * T)

    @torch.no_grad()
    def build_state(
        self, keys: Tensor, values: Tensor, mask: Tensor,
        initial: dict[int, Tensor] | None = None, total_tokens: int | None = None,
    ) -> dict[int, Tensor]:
        """Write captured keys/values (``[L, Hkv, N, D]``) into per-layer states.

        ``initial`` continues writing into existing states (streaming a corpus in
        chunks); ``total_tokens`` is the corpus-wide token count used by
        ``beta_norm="tokens"`` when writing in chunks.
        """

        keys = keys[:, :, mask].float()
        values = values[:, :, mask].float()
        if self.config.center_keys:
            if self.key_mean is None:
                raise RuntimeError("center_keys=True requires calibrate_key_mean() first")
            mean = torch.stack([self.key_mean[i] for i in self.layer_indices]).to(keys)
            keys = keys - mean[:, :, None, :]
        if self.config.key_expand:
            return self._build_expanded(keys, values, initial, total_tokens)
        keys = F.normalize(keys, dim=-1)
        beta = self.config.beta
        if self.config.beta_norm == "tokens":
            beta = beta / max(int(total_tokens if total_tokens is not None else keys.shape[2]), 1)
        elif self.config.beta_norm != "none":
            raise ValueError(f"unknown beta_norm {self.config.beta_norm!r}")
        if initial is None:
            state0 = delta_rule.empty_state(keys.shape[:2], self.head_dim, self.head_dim, device=keys.device)
        else:
            state0 = torch.stack([initial[i] for i in self.layer_indices]).float()
        if self.config.rule == "hebbian" and self.config.alpha == 1.0:
            state = state0 + beta * torch.einsum("lhtk,lhtv->lhkv", keys, values)
        else:
            state = delta_rule.write(
                state0, keys, values,
                beta=beta, alpha=self.config.alpha, rule=self.config.rule,
            )
        return {index: state[n] for n, index in enumerate(self.layer_indices)}

    def _build_expanded(self, keys: Tensor, values: Tensor, initial, total_tokens) -> dict[int, Tensor]:
        """Expanded-key write, streamed in token chunks (expanded keys are large)."""

        cfg = self.config
        if cfg.alpha != 1.0:
            raise ValueError("expanded keys support alpha = 1 only")
        beta = cfg.beta
        if cfg.beta_norm == "tokens":
            beta = beta / max(int(total_tokens if total_tokens is not None else keys.shape[2]), 1)
        if initial is None:
            state = torch.zeros(*keys.shape[:2], self.key_dim, self.head_dim, device=keys.device)
        else:
            state = torch.stack([initial[i] for i in self.layer_indices]).float()
        for start in range(0, keys.shape[2], cfg.write_chunk):
            k = F.normalize(dpfp(keys[:, :, start : start + cfg.write_chunk], cfg.key_expand), dim=-1)
            v = values[:, :, start : start + cfg.write_chunk]
            if cfg.rule == "hebbian":
                state = state + beta * torch.einsum("lhtk,lhtv->lhkv", k, v)
            else:
                b = torch.full(k.shape[:-1], beta, device=k.device)
                state = delta_rule.write_chunked(state, k, v, b, chunk_size=cfg.write_chunk)
        return {index: state[n] for n, index in enumerate(self.layer_indices)}

    @torch.no_grad()
    def calibrate_key_mean(self, keys: Tensor, mask: Tensor) -> None:
        mean = keys[:, :, mask].float().mean(dim=2)  # [L, Hkv, D]
        self.key_mean = {index: mean[n] for n, index in enumerate(self.layer_indices)}

    # ------------------------------------------------------------------ patch
    def _make_forward(self, index: int, attention: nn.Module):
        from transformers.models.qwen3 import modeling_qwen3 as mq

        owner = self

        def forward(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attention.head_dim)
            q_pre = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            k_pre = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            if owner.mode == "capture":
                owner._captures[index].keys.append(k_pre.detach())
                owner._captures[index].values.append(value_states.detach())
                if owner.capture_queries:
                    owner._captures[index].queries.append(q_pre.detach())

            cos, sin = position_embeddings
            query_states, key_states = mq.apply_rotary_pos_emb(q_pre, k_pre, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, attention.layer_idx)

            interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(
                attention.config._attn_implementation, mq.eager_attention_forward
            )
            attn_output, attn_weights = interface(
                attention, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=attention.scaling,
                sliding_window=attention.sliding_window, **kwargs,
            )  # [B, T, H, D]

            if owner.mode == "read" and owner.state is not None and index in owner.state:
                attn_output = attn_output + owner._read(index, q_pre).to(attn_output.dtype)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return attention.o_proj(attn_output), attn_weights

        return forward

    def _read(self, index: int, q_pre: Tensor) -> Tensor:
        state = self.state[index]  # [Hkv, D, D]
        # Key centring (if enabled) is a key-side calibration; queries are used as is
        # unless center_queries is set (then they are centred with the same mean).
        queries = q_pre.float()
        if self.config.center_queries:
            if not self.config.center_keys or self.key_mean is None:
                raise RuntimeError("center_queries=True requires center_keys and calibrate_key_mean()")
            mean = self.key_mean[index].to(queries).repeat_interleave(self.groups, dim=0)  # [H, D]
            queries = queries - mean[None, :, None, :]
        if self.config.key_expand:
            queries = dpfp(queries, self.config.key_expand)
        queries = F.normalize(queries, dim=-1)  # [B, H, T, D or expanded D]
        expanded = state.repeat_interleave(self.groups, dim=0)  # [H, D, D]
        read = torch.einsum("bhtk,hkv->bhtv", queries, expanded)
        gate = self.gates.get(index, self.config.gate)
        if isinstance(gate, Tensor):
            gate = gate.view(1, -1, 1, 1)
        return (gate * read).transpose(1, 2)  # [B, T, H, D]

    def close(self) -> None:
        layers = self.qwen.model.layers
        for index, original in self._originals.items():
            layers[index].self_attn.forward = original
        self._originals = {}


def null_state(states: dict[int, Tensor]) -> dict[int, Tensor]:
    return {index: torch.zeros_like(value) for index, value in states.items()}


def layer_subset(n_layers: int, spec: str | None) -> tuple[int, ...] | None:
    """Parse 'all', 'a-b', or 'i,j,k'."""

    if spec in (None, "", "all"):
        return None
    if "-" in spec and "," not in spec:
        start, end = (int(x) for x in spec.split("-"))
        return tuple(range(start, end + 1))
    values: Sequence[int] = tuple(int(x) for x in spec.split(","))
    if any(not 0 <= v < n_layers for v in values):
        raise ValueError(f"layer index out of range in {spec!r}")
    return tuple(values)
