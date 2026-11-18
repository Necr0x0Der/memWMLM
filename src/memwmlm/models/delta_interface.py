"""Variant 2 of issue #6: a trained delta-rule writer/reader around frozen Qwen3.

One associative state ``S`` of shape ``[d, d]`` is written from the hidden
states of a chosen decoder layer ``l`` and read at the same layer:

* writer: ``x = rms(h)``, ``k = norm(W_K x)``, ``v = W_V x``,
  ``beta = sigmoid(w_b . x + b)``; delta rule with ``alpha = 1``
  (:func:`memwmlm.memory.delta_rule.write_chunked`);
* reader: ``q = norm(W_Q rms(h))``, ``r = q^T S``; the output of layer ``l``
  becomes ``h + W_O r``. ``W_O`` is zero-initialised, so an untrained
  interface (or ``S = 0``) leaves Qwen exactly unchanged.

The Qwen weights stay frozen. Hidden states for writing are taken from a
forward pass truncated after layer ``l`` with no read active.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from memwmlm.memory import delta_rule


class _Stop(Exception):
    pass


def _rms(h: Tensor) -> Tensor:
    h = h.float()
    return h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)


class DeltaInterface(nn.Module):
    def __init__(self, hidden_size: int, width: int, beta_bias: float = 0.0, shift_values: bool = False,
                 bias_state: bool = False) -> None:
        super().__init__()
        self.width = width
        self.shift_values = shift_values
        # learned content-independent state added to every read (absorbs generic adaptation)
        self.s0 = nn.Parameter(torch.zeros(width, width)) if bias_state else None
        # fixed per-dimension input standardisation (Variant 2b); identity until set_input_stats
        self.register_buffer("in_mean", torch.zeros(hidden_size))
        self.register_buffer("in_std", torch.ones(hidden_size))
        self.register_buffer("standardise", torch.zeros((), dtype=torch.bool))
        self.w_k = nn.Linear(hidden_size, width, bias=False)
        self.w_v = nn.Linear(hidden_size, width, bias=False)
        self.w_q = nn.Linear(hidden_size, width, bias=False)
        self.w_o = nn.Linear(width, hidden_size, bias=False)
        self.w_b = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_b.weight)
        nn.init.constant_(self.w_b.bias, beta_bias)

    def set_input_stats(self, mean: Tensor, std: Tensor) -> None:
        """Replace RMS normalisation by ``(h - mean) / std`` (removes Qwen's massive-activation dims)."""

        self.in_mean.copy_(mean)
        self.in_std.copy_(std.clamp_min(1e-3))
        self.standardise.fill_(True)

    def _norm(self, h: Tensor) -> Tensor:
        if bool(self.standardise):
            return (h.float() - self.in_mean) / self.in_std
        return _rms(h)

    def empty_state(self, device: torch.device | str) -> Tensor:
        return torch.zeros(self.width, self.width, device=device)

    def write(self, hidden: Tensor, state: Tensor | None = None, chunk_size: int = 512,
              checkpoint: bool = False, lengths: list[int] | None = None) -> Tensor:
        """Write token hidden states ``[T, hidden]`` (in order) into ``state``.

        With ``shift_values`` (Variant 2b) the pairs are (token t, token t+1)
        within each passage; ``lengths`` gives the passage lengths in tokens.
        """

        x = self._norm(hidden)
        if self.shift_values:
            if lengths is None:
                raise ValueError("shift_values needs passage lengths")
            starts = torch.tensor([0] + list(lengths[:-1]), device=hidden.device).cumsum(0)
            last = starts + torch.tensor(lengths, device=hidden.device) - 1
            keep = torch.ones(x.shape[0], dtype=torch.bool, device=hidden.device)
            keep[last] = False
            idx = keep.nonzero().squeeze(-1)
            kx, vx = x[idx], x[idx + 1]
        else:
            kx = vx = x
        keys = F.normalize(self.w_k(kx), dim=-1)
        values = self.w_v(vx)
        beta = torch.sigmoid(self.w_b(kx)).squeeze(-1)
        if state is None:
            state = self.empty_state(hidden.device)
        return delta_rule.write_chunked(state, keys, values, beta, chunk_size=chunk_size, checkpoint=checkpoint)

    def read(self, hidden: Tensor, state: Tensor) -> Tensor:
        """``state`` is ``[d, d]`` (shared) or ``[B, d, d]`` (one per batch row)."""

        q = F.normalize(self.w_q(self._norm(hidden)), dim=-1)
        if self.s0 is not None:
            state = state + self.s0
        return self.w_o(q @ state)


class InterfaceQwen:
    """Attach a :class:`DeltaInterface` to decoder layer ``layer`` of a Qwen3 causal LM."""

    def __init__(self, qwen: nn.Module, interface: DeltaInterface, layer: int) -> None:
        self.qwen = qwen
        self.interface = interface
        self.layer = layer
        self.state: Tensor | None = None
        self._stop = False
        self._captured: Tensor | None = None
        for parameter in qwen.parameters():
            parameter.requires_grad_(False)
        self._handle = qwen.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: tuple, output: Tensor) -> Tensor | None:
        if self._stop:
            self._captured = output
            raise _Stop
        if self.state is None:
            return None
        return output + self.interface.read(output, self.state).to(output.dtype)

    @contextmanager
    def reading(self, state: Tensor | None) -> Iterator[None]:
        previous, self.state = self.state, state
        try:
            yield
        finally:
            self.state = previous

    @torch.no_grad()
    def layer_hidden(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Output of decoder layer ``layer`` for a padded batch (no read, truncated pass)."""

        previous, self.state, self._stop = self.state, None, True
        try:
            self.qwen.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _Stop:
            pass
        finally:
            self.state, self._stop = previous, False
        captured, self._captured = self._captured, None
        if captured is None:
            raise RuntimeError("layer hook did not fire")
        return captured

    def passage_tokens(self, input_ids: Tensor, attention_mask: Tensor, skip_first: bool = True,
                       return_lengths: bool = False):
        """Hidden states of all real tokens of right-padded passages, concatenated in order."""

        hidden = self.layer_hidden(input_ids, attention_mask)
        mask = attention_mask.bool().clone()
        if skip_first:
            mask[:, 0] = False
        if return_lengths:
            return hidden[mask], mask.sum(1).tolist()
        return hidden[mask]

    def close(self) -> None:
        self._handle.remove()
