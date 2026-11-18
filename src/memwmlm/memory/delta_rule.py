"""Write-once associative matrix memory updated by the delta rule (issue #6).

State convention: ``S`` has shape ``[..., d_k, d_v]`` and is read as
``read(q) = q @ S`` (i.e. ``sum_i q_i S[i, :]``).

Two update rules are provided:

* ``"delta"`` (Widrow-Hoff / DeltaNet):
  ``S <- alpha * S + beta * k (v - k^T (alpha * S))^T``.
  With unit-norm keys and ``beta = 1`` writing ``(k, v)`` makes ``read(k) == v``
  exactly and *overwrites* the previous value stored under ``k``.
* ``"hebbian"`` (outer-product / linear attention with decay):
  ``S <- alpha * S + beta * k v^T``. This is the formula written in the issue
  text; it accumulates rather than overwrites, so repeated or correlated keys
  interfere additively.

The state is runtime memory produced from the content; it is never an
``nn.Parameter`` and is not trained by gradient descent.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Rule = Literal["delta", "hebbian"]


def empty_state(
    batch_shape: tuple[int, ...], d_k: int, d_v: int, *, device=None, dtype=torch.float32
) -> Tensor:
    return torch.zeros(*batch_shape, d_k, d_v, device=device, dtype=dtype)


def write(
    state: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    beta: float | Tensor = 1.0,
    alpha: float = 1.0,
    rule: Rule = "delta",
    mask: Tensor | None = None,
) -> Tensor:
    """Sequentially write ``T`` key/value pairs into ``state``.

    Args:
        state: ``[..., d_k, d_v]``.
        keys: ``[..., T, d_k]`` (callers normally L2-normalise them).
        values: ``[..., T, d_v]``.
        beta: write strength; scalar or ``[..., T]``.
        alpha: per-write decay of the existing state (1.0 = no decay).
        rule: ``"delta"`` or ``"hebbian"``.
        mask: optional ``[..., T]`` bool, True = write this position.

    Returns the new state; the input is not modified. Differentiable.
    """

    if rule not in ("delta", "hebbian"):
        raise ValueError(f"Unknown rule {rule!r}")
    if keys.shape[:-1] != values.shape[:-1]:
        raise ValueError("keys and values must share leading/time dimensions")
    if keys.shape[-1] != state.shape[-2] or values.shape[-1] != state.shape[-1]:
        raise ValueError("key/value widths must match the state")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    keys = keys.to(state.dtype)
    values = values.to(state.dtype)
    steps = keys.shape[-2]
    beta_t = torch.as_tensor(beta, device=state.device, dtype=state.dtype)
    if beta_t.dim() == 0:
        beta_t = beta_t.expand(keys.shape[:-1])
    else:
        beta_t = beta_t.to(state.dtype).expand(keys.shape[:-1])
    if mask is not None:
        beta_t = beta_t * mask.to(state.dtype).expand(keys.shape[:-1])
        decay = torch.where(mask.expand(keys.shape[:-1]).bool(), alpha, 1.0).to(state.dtype)
    else:
        decay = None

    for t in range(steps):
        k = keys[..., t, :]
        v = values[..., t, :]
        b = beta_t[..., t, None, None]
        a = alpha if decay is None else decay[..., t, None, None]
        state = a * state
        if rule == "delta":
            predicted = torch.einsum("...k,...kv->...v", k, state)
            update = v - predicted
        else:
            update = v
        state = state + b * k.unsqueeze(-1) * update.unsqueeze(-2)
    return state


def read(state: Tensor, queries: Tensor) -> Tensor:
    """Read ``[..., T, d_k]`` queries from ``[..., d_k, d_v]`` state -> ``[..., T, d_v]``."""

    return torch.einsum("...tk,...kv->...tv", queries.to(state.dtype), state)


def write_chunked(
    state: Tensor,
    keys: Tensor,
    values: Tensor,
    beta: Tensor,
    chunk_size: int = 256,
    checkpoint: bool = False,
) -> Tensor:
    """Exact delta-rule write (``alpha = 1``) computed chunk-parallel.

    Equivalent to :func:`write` with ``rule="delta"``, ``alpha=1`` and per-token
    ``beta`` ``[..., T]``. Within a chunk, the corrections
    ``u_i = beta_i (v_i - S_0^T k_i - sum_{j<i} (k_i . k_j) u_j)`` solve the
    unit-lower-triangular system ``(I + diag(beta) tril(K K^T, -1)) U =
    diag(beta) (V - K S_0)``; then ``S = S_0 + K^T U``. ``checkpoint`` recomputes
    each chunk in the backward pass so only chunk-boundary states are kept.
    """

    def step(s: Tensor, k: Tensor, v: Tensor, b: Tensor) -> Tensor:
        c = k.shape[-2]
        eye = torch.eye(c, device=k.device, dtype=k.dtype)
        a = torch.tril(k @ k.transpose(-1, -2), diagonal=-1) * b.unsqueeze(-1)
        r = b.unsqueeze(-1) * (v - k @ s)
        u = torch.linalg.solve_triangular(eye + a, r, upper=False, unitriangular=True)
        return s + k.transpose(-1, -2) @ u

    keys = keys.to(state.dtype)
    values = values.to(state.dtype)
    beta = beta.to(state.dtype)
    for start in range(0, keys.shape[-2], chunk_size):
        args = (state, keys[..., start : start + chunk_size, :], values[..., start : start + chunk_size, :],
                beta[..., start : start + chunk_size])
        if checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as ckpt

            state = ckpt(step, *args, use_reentrant=False)
        else:
            state = step(*args)
    return state
