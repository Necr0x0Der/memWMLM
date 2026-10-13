"""Task-adapted full-context bAbI control: frozen Qwen, attention-projection LoRA.

No story is hidden and no writable-memory components are used. Prompts and
candidate scoring are identical to the frozen full-context baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from memwmlm.data.babi import BabiExample
from memwmlm.models.lora import ToggleableLoRALinear, replace_qwen_projection
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.babi import BabiVariant, render_babi_prompt
from memwmlm.training.reset_pair import causal_answer_loss


@dataclass(frozen=True)
class FullContextBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


def make_full_context_batch(
    examples: Sequence[BabiExample], tokenizer: object, formatter: QwenChatPromptFormatter, device: torch.device
) -> FullContextBatch:
    """Right-pad full-context prompts + answer + EOS; mask all non-answer loss."""
    if not examples:
        raise ValueError("Cannot batch zero examples")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must provide an EOS token")
    sequences: list[list[int]] = []
    starts: list[int] = []
    for example in examples:
        prefix = formatter(render_babi_prompt(example, BabiVariant.FULL_CONTEXT).prefix)
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        answer_ids = tokenizer(example.answer, add_special_tokens=False).input_ids
        if not prefix_ids or not answer_ids:
            raise ValueError(f"Empty prompt or answer tokens for {example.example_id}")
        starts.append(len(prefix_ids))
        sequences.append([*prefix_ids, *answer_ids, int(eos_id)])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    ids = torch.full((len(sequences), max(map(len, sequences))), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for row, (sequence, start) in enumerate(zip(sequences, starts, strict=True)):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, : len(sequence)] = 1
        labels[row, start : len(sequence)] = ids[row, start : len(sequence)]
    return FullContextBatch(ids, mask, labels)


class FullContextLoRA(nn.Module):
    """Freeze the complete base and adapt q_proj/o_proj in the final N blocks."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float, layers: int):
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires rank > 0, alpha > 0 and dropout in [0, 1)")
        blocks = base.model.layers
        if layers <= 0 or layers > len(blocks):
            raise ValueError(f"layers must be in [1, {len(blocks)}]")
        self.base = base
        self.rank, self.alpha, self.dropout, self.layers = rank, alpha, dropout, layers
        self.target_layer_indices = tuple(range(len(blocks) - layers, len(blocks)))
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.adapters: list[ToggleableLoRALinear] = []
        for index in self.target_layer_indices:
            attention = blocks[index].self_attn
            for name in ("q_proj", "o_proj"):
                adapter = replace_qwen_projection(attention, name, rank, alpha, dropout)
                adapter.mode = "application"
                self.adapters.append(adapter)
        self.base.eval()

    def train(self, mode: bool = True) -> FullContextLoRA:
        super().train(mode)
        # Dropout in the pretrained backbone must remain disabled in training.
        self.base.eval()
        for adapter in self.adapters:
            adapter.dropout.train(mode)
        return self

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.base(*args, **kwargs)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {name: parameter.detach().cpu() for name, parameter in self.named_parameters() if parameter.requires_grad}

    def load_adapter_state_dict(self, state: dict[str, Tensor]) -> None:
        parameters = dict(self.named_parameters())
        expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError(f"LoRA keys mismatch: missing={sorted(expected - set(state))}, unexpected={sorted(set(state) - expected)}")
        with torch.no_grad():
            for name in expected:
                if state[name].shape != parameters[name].shape:
                    raise ValueError(f"LoRA shape mismatch: {name}")
                parameters[name].copy_(state[name])

    def summary(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_layer_indices": list(self.target_layer_indices),
            "projections": ["q_proj", "o_proj"],
            "trainable_parameters": sum(p.numel() for p in self.trainable_parameters()),
            "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }


__all__ = ["FullContextLoRA", "FullContextBatch", "make_full_context_batch", "causal_answer_loss"]
