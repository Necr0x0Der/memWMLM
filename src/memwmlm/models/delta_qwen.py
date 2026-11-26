"""Variant 1 of issue #6: delta-rule memory on Qwen3's *existing* attention heads.

Each selected Qwen3 attention layer is patched so that it can run in one of
three modes:

* ``off``     -- unmodified frozen Qwen attention;
* ``capture`` -- additionally record pre-RoPE, ``k_norm``-ed keys and values per
  KV head (used to write a passage bank into memory);
* ``read``    -- add ``gate * read(S, q)`` to every query head's attention output
  before ``o_proj``; ``q`` is the pre-RoPE ``q_norm``-ed query and ``S`` is the
  delta-rule state of that head's KV group.

With ``DeltaMemoryConfig(read="knn")`` the "state" is instead the uncompressed
stored keys/values per KV head (an upper bound, no delta rule): each query
attends with one joint softmax over its normal context and its ``knn_top_k``
best stored keys (no gate; empty memory = plain Qwen). Stored keys are
RoPE-rotated *right-aligned*: token ``j`` of a passage of ``L`` tokens gets
position ``j - L``, so every passage sits immediately before the prompt
(which starts at position 0), as a pasted passage would; all passages share
that slot ("parallel context windows"). Scoring stored keys without position
(distance 0) was tried first and swamps attention (calibration smoke test).

With a trained ``FeatureMaps`` attached (``memory.feature_maps``), the linear
delta-rule memory writes ``phi_k(k)`` (L2-normalised) and reads with
``phi_q(q)``: learned per-head maps (Hedgehog-style, Zhang et al. 2024)
distilled so that the read reproduces what the softmax memory adds to each
attention output (``train_feature_map.py``). Centring / whitening / DPFP are
not used with them.

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
    read: str = "linear"  # "linear" = delta-rule state S; "knn" = stored keys/values, joint softmax (upper bound)
    knn_top_k: int = 32  # stored keys per query head and token in the joint softmax (0 = all)
    knn_chunk: int = 8192  # stored keys scored per chunk when selecting the top k
    # whiten keys and read queries per head with calibration key statistics:
    # x -> W (x - mean), W = (C + eps * mean_eig * I)^(-1/2); needs center_keys and
    # calibrate_key_mean() (which then also estimates W). Spreads keys over all dims.
    whiten: bool = False
    whiten_eps: float = 0.1


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


def _checkpoint(fn, *args):
    from torch.utils.checkpoint import checkpoint

    return checkpoint(fn, *args, use_reentrant=False)


def dpfp(x: Tensor, nu: int) -> Tensor:
    """Deterministic parameter-free projection: ``[..., D] -> [..., 2 D nu]`` (non-negative)."""

    x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
    return torch.cat([x * torch.roll(x, shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)


class _Capture:
    def __init__(self) -> None:
        self.keys: list[Tensor] = []  # per call: [B, Hkv, T, D]
        self.values: list[Tensor] = []
        self.queries: list[Tensor] = []  # per call: [B, H, T, D]
        self.hidden: list[Tensor] = []  # per call: [B, T, hidden] attention input (only if capture_hidden)


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
        self.key_whiten: dict[int, Tensor] | None = None  # layer -> [Hkv, D, D] (symmetric)
        self.gates: dict[int, Tensor | float] = {}
        self.feature_maps: FeatureMaps | None = None  # learned phi_q / phi_k (linear read only)
        self.record: dict[int, tuple[Tensor, Tensor]] | None = None  # knn teacher: layer -> (q_pre, memory contribution)
        # Sanity check 6: if set, memory is read with these pre-RoPE queries (layer -> [B, H, D], e.g. the
        # mean query of the question's passage) at every position, instead of the token's own query.
        self.read_queries: dict[int, Tensor] | None = None
        # Learned write strength (2026-11-24, option 1): beta = sigmoid(beta_logits) per (layer, KV head),
        # [L, Hkv]; replaces config.beta in the plain (D 128) delta write, differentiable.
        self.beta_logits: Tensor | None = None
        # Option 2: per-token beta = sigmoid(w . x_t + b), x_t = the written token's attention input
        # (post input_layernorm) at that layer; module with weight [L, Hkv, hidden] and bias [L, Hkv].
        self.beta_proj: nn.Module | None = None
        self.last_hidden: Tensor | None = None  # [L, B*T, hidden] from the last capture_passages (if needed)
        # Step B (2026-11-25): trainable writer (``WriterLoRA``) -> written keys/values are computed from the
        # captured attention inputs with Qwen's k/v projections + a low-rank update (Qwen's own pass unchanged).
        self.writer: nn.Module | None = None
        self._captures: dict[int, _Capture] = {}
        self._originals: dict[int, Callable] = {}
        for index in self.layer_indices:
            attention = layers[index].self_attn
            self._originals[index] = attention.forward
            attention.forward = self._make_forward(index, attention)

    @property
    def needs_hidden(self) -> bool:
        return self.beta_proj is not None or self.writer is not None

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
    def capture_passages(
        self, input_ids: Tensor, attention_mask: Tensor, rope: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        if self.needs_hidden:
            hidden = torch.stack([torch.cat(captures[i].hidden, dim=0) for i in self.layer_indices])  # [L, B, T, X]
            self.last_hidden = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])
        if self.config.read == "knn" if rope is None else rope:
            keys = self._rope_right_aligned(keys, attention_mask)
        # [L, B, Hkv, T, D] -> [L, Hkv, B*T, D]
        L, B, H, T, D = keys.shape
        keys = keys.permute(0, 2, 1, 3, 4).reshape(L, H, B * T, D)
        values = values.permute(0, 2, 1, 3, 4).reshape(L, H, B * T, D)
        return keys, values, mask.reshape(B * T)

    @torch.no_grad()
    def passage_queries(self, input_ids: Tensor, attention_mask: Tensor) -> dict[int, Tensor]:
        """Mean pre-RoPE query of each passage over its written tokens: layer -> ``[B, H, D]``.

        Same encoding and write mask as ``capture_passages`` (padding and, if
        ``skip_first_token``, the first token excluded).
        """

        previous = self.capture_queries
        self.capture_queries = True
        try:
            with self.capturing() as captures:
                self.qwen(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            self.capture_queries = previous
        mask = attention_mask.bool().clone()
        if self.config.skip_first_token:
            mask[torch.arange(mask.shape[0]), mask.float().argmax(dim=1)] = False
        w = mask.float()[:, None, :, None]  # [B, 1, T, 1]
        out = {}
        for index in self.layer_indices:
            q = torch.cat(captures[index].queries, dim=0).float()  # [B, H, T, D]
            out[index] = (q * w).sum(2) / w.sum(2).clamp_min(1.0)
        return out

    def _rope_right_aligned(self, keys: Tensor, attention_mask: Tensor) -> Tensor:
        """Rotate pre-RoPE keys ``[L, B, Hkv, T, D]`` to positions ``j - len`` (right-padded rows)."""

        from transformers.models.qwen3.modeling_qwen3 import rotate_half

        lengths = attention_mask.sum(dim=1, keepdim=True)  # [B, 1]
        positions = torch.arange(attention_mask.shape[1], device=keys.device)[None, :] - lengths  # [B, T]
        cos, sin = self.qwen.model.rotary_emb(keys[0, :, 0], positions)  # [B, T, D]
        cos, sin = cos[None, :, None].to(keys.dtype), sin[None, :, None].to(keys.dtype)
        return keys * cos + rotate_half(keys) * sin

    def build_state(
        self, keys: Tensor, values: Tensor, mask: Tensor,
        initial: dict[int, Tensor] | None = None, total_tokens: int | None = None,
        hidden: Tensor | None = None,
    ) -> dict[int, Tensor]:
        """Write captured keys/values (``[L, Hkv, N, D]``) into per-layer states.

        ``initial`` continues writing into existing states (streaming a corpus in
        chunks); ``total_tokens`` is the corpus-wide token count used by
        ``beta_norm="tokens"`` when writing in chunks.
        """

        if self.config.read == "knn":
            if initial is not None:
                raise ValueError("knn memory is built in one call")
            # raw k_norm-ed, right-aligned RoPE keys (capture_passages): softmax is invariant
            # to a shared key offset, and the scaling expects Qwen's key scale (no centring / L2)
            return {index: (keys[n][:, mask], values[n][:, mask]) for n, index in enumerate(self.layer_indices)}
        if self.feature_maps is not None:
            return self.build_feature_state(keys[:, :, mask], values[:, :, mask])
        if self.needs_hidden:
            if hidden is None:
                raise ValueError("beta_proj / writer need the written tokens' hidden states (capture_passages -> last_hidden)")
            hidden = hidden[:, mask]
        if self.writer is not None:
            if self.config.read == "knn" or self.feature_maps is not None:
                raise ValueError("writer supports the linear delta-rule read only")
            keys, values = self.writer(hidden)  # [L, Hkv, T, D] for the written tokens only
            mask = torch.ones(keys.shape[2], dtype=torch.bool, device=keys.device)
        if self.beta_logits is None and self.beta_proj is None and self.writer is None:
            with torch.no_grad():
                return self._build_plain(keys, values, mask, initial, total_tokens)
        return self._build_plain(keys, values, mask, initial, total_tokens, hidden)

    def _build_plain(self, keys, values, mask, initial, total_tokens, hidden=None) -> dict[int, Tensor]:
        keys = keys[:, :, mask].float()
        values = values[:, :, mask].float()
        if self.config.center_keys:
            if self.key_mean is None:
                raise RuntimeError("center_keys=True requires calibrate_key_mean() first")
            mean = torch.stack([self.key_mean[i] for i in self.layer_indices]).to(keys)
            keys = keys - mean[:, :, None, :]
        if self.config.whiten:
            if not self.config.center_keys or self.key_whiten is None:
                raise RuntimeError("whiten=True requires center_keys and calibrate_key_mean() first")
            w = torch.stack([self.key_whiten[i] for i in self.layer_indices]).to(keys)
            keys = torch.einsum("lhtd,lhed->lhte", keys, w)
        if self.config.key_expand:
            return self._build_expanded(keys, values, initial, total_tokens, hidden)
        keys = F.normalize(keys, dim=-1)
        if self.beta_logits is not None or self.beta_proj is not None:
            if self.config.rule != "delta" or self.config.alpha != 1.0 or self.config.beta_norm != "none" or initial:
                raise ValueError("learned beta supports the plain delta rule (alpha 1, one write) only")
            if self.beta_proj is not None:
                beta = self.beta_proj(hidden)  # [L, Hkv, T]
            else:
                beta = torch.sigmoid(self.beta_logits.float())[:, :, None].expand(keys.shape[:3])
            state0 = delta_rule.empty_state(keys.shape[:2], self.head_dim, self.head_dim, device=keys.device)
            state = delta_rule.write_chunked(state0, keys, values, beta, chunk_size=self.config.write_chunk,
                                             checkpoint=torch.is_grad_enabled())
            return {index: state[n] for n, index in enumerate(self.layer_indices)}
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

    def build_feature_state(self, keys: Tensor, values: Tensor) -> dict[int, Tensor]:
        """Delta-rule write of ``phi_k(k)`` -> v per layer (differentiable; used in training)."""

        fm, cfg = self.feature_maps, self.config
        out = {}
        for n, index in enumerate(self.layer_indices):
            k = fm.keys(n, keys[n].float())  # [Hkv, T, 2D]
            state = torch.zeros(*k.shape[:-2], k.shape[-1], self.head_dim, device=k.device)
            beta = torch.full(k.shape[:-1], cfg.beta, device=k.device)
            out[index] = delta_rule.write_chunked(state, k, values[n].float(), beta, chunk_size=cfg.write_chunk,
                                                  checkpoint=torch.is_grad_enabled())
        return out

    def _build_expanded(self, keys: Tensor, values: Tensor, initial, total_tokens, hidden=None) -> dict[int, Tensor]:
        """Expanded-key write, streamed in token chunks (expanded keys are large)."""

        cfg = self.config
        if cfg.alpha != 1.0:
            raise ValueError("expanded keys support alpha = 1 only")
        if self.beta_logits is not None or self.beta_proj is not None:
            return self._build_expanded_learned(keys, values, initial, hidden)
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

    def _build_expanded_learned(self, keys: Tensor, values: Tensor, initial, hidden) -> dict[int, Tensor]:
        """Expanded-key delta write with learned beta (2026-11-24 amendment: per-token beta + DPFP).

        Differentiable in beta. With grad enabled each layer is checkpointed (and each
        write chunk inside it), so only the final [Hkv, 2*D*nu, D] states are kept.
        """

        cfg = self.config
        if cfg.rule != "delta" or cfg.beta_norm != "none" or initial:
            raise ValueError("learned beta supports the plain delta rule (alpha 1, one write) only")
        if self.beta_proj is not None:
            beta = self.beta_proj(hidden)  # [L, Hkv, T]
        else:
            beta = torch.sigmoid(self.beta_logits.float())[:, :, None].expand(keys.shape[:3])
        grad = torch.is_grad_enabled()

        def chunk(state, k, v, b):
            k = F.normalize(dpfp(k, cfg.key_expand), dim=-1)
            return delta_rule.write_chunked(state, k, v, b, chunk_size=cfg.write_chunk)

        def layer(k, v, b):
            state = torch.zeros(k.shape[0], self.key_dim, self.head_dim, device=k.device)
            for start in range(0, k.shape[1], cfg.write_chunk):
                part = (state, k[:, start:start + cfg.write_chunk], v[:, start:start + cfg.write_chunk],
                        b[:, start:start + cfg.write_chunk])
                state = _checkpoint(chunk, *part) if grad else chunk(*part)
            return state

        out = {}
        for n, index in enumerate(self.layer_indices):
            args = (keys[n], values[n], beta[n].float())
            out[index] = _checkpoint(layer, *args) if grad else layer(*args)
        return out

    @torch.no_grad()
    def calibrate_key_mean(self, keys: Tensor, mask: Tensor) -> None:
        k = keys[:, :, mask].float()
        mean = k.mean(dim=2)  # [L, Hkv, D]
        self.key_mean = {index: mean[n] for n, index in enumerate(self.layer_indices)}
        if self.config.whiten:
            self.key_whiten = {index: w for index, w in zip(
                self.layer_indices, whitening_matrix(k - mean[:, :, None], self.config.whiten_eps))}

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
                if owner.needs_hidden:
                    owner._captures[index].hidden.append(hidden_states.detach())

            cos, sin = position_embeddings
            query_states, key_states = mq.apply_rotary_pos_emb(q_pre, k_pre, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, attention.layer_idx)

            interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(
                attention.config._attn_implementation, mq.eager_attention_forward
            )
            q_read = q_pre  # query used for the memory read only
            if owner.read_queries is not None:
                fixed = owner.read_queries[index].to(q_pre.dtype)  # [B, H, D]
                q_read = fixed[:, :, None, :].expand_as(q_pre)
            if (owner.mode == "read" and owner.config.read == "knn" and owner.state is not None
                    and index in owner.state):
                mem_k, mem_v = owner.state[index]
                q_mem = query_states
                if owner.read_queries is not None:  # rotate to the reading token's position, like its own query
                    q_mem, _ = mq.apply_rotary_pos_emb(q_read, k_pre, cos, sin)
                attn_output = joint_memory_attention(
                    query_states, key_states, value_states, attention_mask, q_mem, mem_k, mem_v,
                    attention.scaling, owner.config.knn_top_k, owner.config.knn_chunk,
                )  # [B, T, H, D]
                if owner.record is not None:  # distillation target: what memory adds to the plain output
                    plain, _ = interface(
                        attention, query_states, key_states, value_states, attention_mask,
                        dropout=0.0, scaling=attention.scaling, sliding_window=attention.sliding_window, **kwargs,
                    )
                    owner.record[index] = (q_pre.detach(), (attn_output.float() - plain.float()).detach())
                attn_output = attn_output.to(hidden_states.dtype).reshape(*input_shape, -1).contiguous()
                return attention.o_proj(attn_output), None

            attn_output, attn_weights = interface(
                attention, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=attention.scaling,
                sliding_window=attention.sliding_window, **kwargs,
            )  # [B, T, H, D]

            if owner.mode == "read" and owner.state is not None and index in owner.state:
                attn_output = attn_output + owner._read(index, q_read).to(attn_output.dtype)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return attention.o_proj(attn_output), attn_weights

        return forward

    def _read(self, index: int, q_pre: Tensor) -> Tensor:
        state = self.state[index]  # [Hkv, D, D] (or [B, Hkv, K, V] per example)
        if self.feature_maps is not None:
            queries = self.feature_maps.queries(self.layer_indices.index(index), q_pre.float())  # [B, H, T, 2D]
            gdim = 1 if state.dim() == 4 else 0
            spec = "bhtk,bhkv->bhtv" if state.dim() == 4 else "bhtk,hkv->bhtv"
            read = torch.einsum(spec, queries, state.repeat_interleave(self.groups, dim=gdim))
            gate = self.gates.get(index, self.config.gate)
            gate = gate.view(1, -1, 1, 1) if isinstance(gate, Tensor) else gate
            return (gate * read).transpose(1, 2)
        if self.config.key_expand and torch.is_grad_enabled() and state.dim() == 4:
            # training: recompute the expanded queries in backward instead of keeping them
            read = _checkpoint(lambda q, s: self._linear_read(index, q, s), q_pre, state)
        else:
            read = self._linear_read(index, q_pre, state)
        gate = self.gates.get(index, self.config.gate)
        if isinstance(gate, Tensor):
            gate = gate.view(1, -1, 1, 1)
        return (gate * read).transpose(1, 2)  # [B, T, H, D]

    def _linear_read(self, index: int, q_pre: Tensor, state: Tensor) -> Tensor:
        """Ungated linear read -> [B, H, T, D]."""

        # Key centring (if enabled) is a key-side calibration; queries are used as is
        # unless center_queries is set (then they are centred with the same mean).
        queries = q_pre.float()
        if self.config.center_queries:
            if not self.config.center_keys or self.key_mean is None:
                raise RuntimeError("center_queries=True requires center_keys and calibrate_key_mean()")
            mean = self.key_mean[index].to(queries).repeat_interleave(self.groups, dim=0)  # [H, D]
            queries = queries - mean[None, :, None, :]
        if self.config.whiten:
            if not self.config.center_queries or self.key_whiten is None:
                raise RuntimeError("whiten=True requires center_queries (same transform on both sides)")
            w = self.key_whiten[index].to(queries).repeat_interleave(self.groups, dim=0)  # [H, D, D]
            queries = torch.einsum("bhtd,hed->bhte", queries, w)
        if self.config.key_expand:
            queries = dpfp(queries, self.config.key_expand)
        queries = F.normalize(queries, dim=-1)  # [B, H, T, D or expanded D]
        b, h, t, k = queries.shape
        grouped = queries.view(b, h // self.groups, self.groups, t, k)  # query heads of one KV head together
        if state.dim() == 4:  # one state per batch row: [B, Hkv, K, V]
            read = torch.einsum("bhgtk,bhkv->bhgtv", grouped, state)
        else:
            read = torch.einsum("bhgtk,hkv->bhgtv", grouped, state)
        return read.reshape(b, h, t, -1)

    def close(self) -> None:
        layers = self.qwen.model.layers
        for index, original in self._originals.items():
            layers[index].self_attn.forward = original
        self._originals = {}


class TokenBeta(nn.Module):
    """Per-token write strength: beta[l, h, t] = sigmoid(w[l, h] . x[l, t] + b[l, h]).

    ``w`` starts at 0, so at initialisation every token is written with ``sigmoid(b) = beta_init``.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, hidden_size: int, beta_init: float = 0.95) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_layers, n_kv_heads, hidden_size))
        self.bias = nn.Parameter(torch.full((n_layers, n_kv_heads), float(torch.logit(torch.tensor(beta_init)))))

    def forward(self, hidden: Tensor) -> Tensor:  # [L, T, X] -> [L, Hkv, T]
        return torch.sigmoid(torch.einsum("ltx,lhx->lht", hidden.float(), self.weight) + self.bias[:, :, None])


class FeatureMaps(nn.Module):
    """Learned per-head feature maps for the delta-rule memory (Hedgehog-style, Zhang et al. 2024).

    ``phi(x) = [softmax(W x + b), softmax(-(W x + b))]`` (dim ``2D``), ``W``
    initialised to the identity. Keys use one map per (layer, KV head) and are
    L2-normalised (delta rule with beta 1 needs unit keys); queries use one map
    per (layer, query head), unnormalised (a sharp query reads strongly, a flat
    one weakly), times a learned per-head scale. Layer ``n`` = position in
    ``DeltaMemoryQwen.layer_indices``.
    """

    def __init__(self, n_layers: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        eye = torch.eye(head_dim)
        self.wq = nn.Parameter(eye.repeat(n_layers, n_heads, 1, 1))
        self.bq = nn.Parameter(torch.zeros(n_layers, n_heads, head_dim))
        self.wk = nn.Parameter(eye.repeat(n_layers, n_kv_heads, 1, 1))
        self.bk = nn.Parameter(torch.zeros(n_layers, n_kv_heads, head_dim))
        self.log_scale = nn.Parameter(torch.zeros(n_layers, n_heads))

    @staticmethod
    def phi(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        z = torch.einsum("...htd,hed->...hte", x, w) + b[:, None, :]
        return torch.cat([z.softmax(-1), (-z).softmax(-1)], dim=-1)

    def keys(self, n: int, k: Tensor) -> Tensor:
        """``[..., Hkv, T, D]`` -> unit ``[..., Hkv, T, 2D]``."""
        return F.normalize(self.phi(k, self.wk[n], self.bk[n]), dim=-1)

    def queries(self, n: int, q: Tensor) -> Tensor:
        """``[..., H, T, D]`` -> ``[..., H, T, 2D]``."""
        return self.phi(q, self.wq[n], self.bq[n]) * self.log_scale[n].exp()[:, None, None]


def load_feature_maps(memory: "DeltaMemoryQwen", path) -> FeatureMaps:
    """Attach trained ``FeatureMaps`` (``train_feature_map.py`` checkpoint) to ``memory``."""

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    fm = FeatureMaps(*ckpt["shape"])
    fm.load_state_dict(ckpt["state_dict"])
    device = next(memory.qwen.parameters()).device
    memory.feature_maps = fm.to(device).eval().requires_grad_(False)
    return memory.feature_maps


def whitening_matrix(centred: Tensor, eps: float) -> Tensor:
    """``[..., N, D]`` centred samples -> symmetric ``[..., D, D]`` ``(C + eps * mean_eig * I)^(-1/2)``.

    Shrinkage ``eps`` (relative to the mean eigenvalue) keeps nearly empty
    directions from being amplified into noise.
    """

    cov = centred.transpose(-1, -2) @ centred / max(centred.shape[-2] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.clamp_min(0) + eps * evals.mean(-1, keepdim=True)
    return ((evecs * evals.rsqrt()[..., None, :]) @ evecs.transpose(-1, -2)).to(centred.dtype)


def null_state(states: dict[int, Tensor]) -> dict[int, Tensor]:
    if states and isinstance(next(iter(states.values())), tuple):  # knn memory: empty = plain Qwen
        return {index: (k[:, :0], v[:, :0]) for index, (k, v) in states.items()}
    return {index: torch.zeros_like(value) for index, value in states.items()}


def joint_memory_attention(
    query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor | None,
    q_mem: Tensor, mem_k: Tensor, mem_v: Tensor, scaling: float, top_k: int, chunk: int = 8192,
) -> Tensor:
    """One softmax over the normal context and the best stored memory tokens.

    ``query`` ``[B, H, T, D]`` and ``key``/``value`` ``[B, Hkv, S, D]`` are the
    usual (RoPE-ed) attention inputs; ``attention_mask`` is the additive or
    boolean ``[B, 1, T, S]`` mask (``None`` = causal). ``q_mem`` ``[B, H, T, D]``
    is the query scored against the stored keys ``mem_k`` / values ``mem_v``
    ``[Hkv, M, D]``. Returns ``[B, T, H, D]``.
    """

    b, h, t, d = query.shape
    hkv, s = key.shape[1], key.shape[2]
    g = h // hkv
    q = query.float().view(b, hkv, g, t, d)
    logits_c = torch.einsum("bhgtd,bhsd->bhgts", q, key.float()) * scaling
    if attention_mask is None:
        causal = torch.ones(t, s, dtype=torch.bool, device=query.device).tril(s - t)
        logits_c = logits_c.masked_fill(~causal, float("-inf"))
    elif attention_mask.dtype == torch.bool:
        logits_c = logits_c.masked_fill(~attention_mask[:, :, None, :, :s], float("-inf"))
    else:
        logits_c = logits_c + attention_mask[:, :, None, :, :s].float()
    m = mem_k.shape[1]
    if m == 0:
        weights = torch.softmax(logits_c, dim=-1)
        out = torch.einsum("bhgts,bhsd->bhgtd", weights, value.float())
        return out.reshape(b, h, t, d).transpose(1, 2)

    qp = q_mem.view(b, hkv, g, t, d)
    k_sel = m if top_k <= 0 or top_k >= m else top_k
    if k_sel == m:
        idx = torch.arange(m, device=query.device).expand(b, hkv, g, t, m)
    else:  # rank in the model dtype, chunked; recompute the kept logits in float32 below
        best_val, best_idx = None, None
        for start in range(0, m, chunk):
            part = torch.einsum("bhgtd,hmd->bhgtm", qp, mem_k[:, start : start + chunk].to(qp.dtype))
            val, ix = part.topk(min(k_sel, part.shape[-1]), dim=-1)
            ix = ix + start
            if best_val is not None:
                val, keep = torch.cat([best_val, val], -1).topk(k_sel, dim=-1)
                ix = torch.cat([best_idx, ix], -1).gather(-1, keep)
            best_val, best_idx = val, ix
        idx = best_idx
    heads = torch.arange(hkv, device=query.device).view(1, hkv, 1, 1, 1)
    k_mem = mem_k[heads, idx].float()  # [B, Hkv, G, T, K, D]
    v_mem = mem_v[heads, idx].float()
    logits_m = torch.einsum("bhgtd,bhgtkd->bhgtk", qp.float(), k_mem) * scaling
    weights = torch.softmax(torch.cat([logits_c, logits_m], dim=-1), dim=-1)
    out = torch.einsum("bhgts,bhsd->bhgtd", weights[..., :s], value.float())
    out = out + torch.einsum("bhgtk,bhgtkd->bhgtd", weights[..., s:], v_mem)
    return out.reshape(b, h, t, d).transpose(1, 2)


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


class WriterLoRA(nn.Module):
    """Trainable writer of step B: per selected layer, written key = k_norm(W_K x + s B_K A_K x) and
    value = W_V x + s B_V A_V x, x = the token's attention input (captured; Qwen's own pass is unchanged).
    W_K, W_V, k_norm are Qwen's frozen modules (not owned); A random, B zero -> starts as frozen Variant 1."""

    def __init__(self, memory: "DeltaMemoryQwen", rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        layers = memory.qwen.model.layers
        self._attn = [layers[i].self_attn for i in memory.layer_indices]  # plain list: not saved
        self.heads, self.dim = memory.num_kv_heads, memory.head_dim
        L, X, out = len(self._attn), memory.qwen.config.hidden_size, memory.num_kv_heads * memory.head_dim
        device = next(memory.qwen.parameters()).device
        self.scale = alpha / rank
        self.down_k = nn.Parameter(torch.empty(L, rank, X, device=device))
        self.down_v = nn.Parameter(torch.empty(L, rank, X, device=device))
        for w in (self.down_k, self.down_v):
            for n in range(L):
                nn.init.kaiming_uniform_(w.data[n], a=5 ** 0.5)
        self.up_k = nn.Parameter(torch.zeros(L, out, rank, device=device))
        self.up_v = nn.Parameter(torch.zeros(L, out, rank, device=device))

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:  # [L, T, X] -> keys, values [L, Hkv, T, D]
        keys, values = [], []
        for n, attn in enumerate(self._attn):
            x = hidden[n]
            xf = x.float()
            k = attn.k_proj(x).float() + self.scale * (xf @ self.down_k[n].T) @ self.up_k[n].T
            v = attn.v_proj(x).float() + self.scale * (xf @ self.down_v[n].T) @ self.up_v[n].T
            k = attn.k_norm(k.view(-1, self.heads, self.dim)).float()
            keys.append(k.transpose(0, 1))
            values.append(v.view(-1, self.heads, self.dim).transpose(0, 1))
        return torch.stack(keys), torch.stack(values)
