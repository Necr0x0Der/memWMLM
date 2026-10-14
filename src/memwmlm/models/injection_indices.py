"""Resolve memory-reader hook positions; independent of the model runtime."""

from __future__ import annotations

from collections.abc import Sequence


def resolve_injection_indices(
    layer_count: int, injection_layers: int, injection_indices: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Explicit indices override the legacy final-N setting; all indices are 0-based."""

    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if injection_indices is None:
        if not 0 < injection_layers <= layer_count:
            raise ValueError("injection_layers must be in [1, number of Qwen layers]")
        return tuple(range(layer_count - injection_layers, layer_count))
    if isinstance(injection_indices, (str, bytes)) or not isinstance(injection_indices, Sequence):
        raise ValueError("injection_indices must be a nonempty sequence of 0-based integers")
    indices = tuple(injection_indices)
    if not indices or any(type(index) is not int or index < 0 or index >= layer_count for index in indices):
        raise ValueError("injection_indices must be nonempty 0-based integers within the model layers")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("injection_indices must be strictly increasing without duplicates")
    return indices
