"""Frozen causal-LM scoring of a finite set of textual answer candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class CandidatePrediction:
    candidate: str
    log_likelihood: float


class AnswerScorer(Protocol):
    """Common interface for candidate likelihood and free-generation scoring."""

    def predict(self, prefix: str, candidates: Sequence[str]) -> CandidatePrediction: ...


class CausalCandidateScorer:
    """Score continuations exactly, without relying on free-form generation.

    The model and tokenizer are supplied by the caller so this class is also usable
    by later memory-enabled axial-model wrappers.
    """

    def __init__(self, model: object, tokenizer: object) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def score(self, prefix: str, candidates: Sequence[str]) -> list[CandidatePrediction]:
        """Return summed token log-likelihood for each continuation.

        A leading space is deliberately included before each candidate: the prompt
        ends in ``Answer:`` and bAbI answers are lexical continuations, not part of
        the prompt's final token. Candidate-specific tokenization is preserved.
        """

        if not candidates:
            raise ValueError("At least one candidate is required")

        prefix_ids = self.tokenizer(prefix, add_special_tokens=True).input_ids
        if not prefix_ids:
            raise ValueError("Tokenizer produced an empty prompt")

        encoded: list[tuple[str, list[int]]] = []
        for candidate in candidates:
            suffix_ids = self.tokenizer(" " + candidate, add_special_tokens=False).input_ids
            if not suffix_ids:
                raise ValueError(f"Tokenizer produced an empty candidate: {candidate!r}")
            encoded.append((candidate, suffix_ids))

        max_length = max(len(prefix_ids) + len(suffix) for _, suffix in encoded)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

        input_ids = torch.full(
            (len(encoded), max_length), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)

        for row, (_, suffix) in enumerate(encoded):
            sequence = prefix_ids + suffix
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, device=self.device)
            attention_mask[row, :length] = 1
            labels[row, len(prefix_ids) : length] = torch.tensor(suffix, device=self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        log_probs = outputs.logits[:, :-1].log_softmax(dim=-1)
        target_ids = labels[:, 1:]
        target_mask = target_ids.ne(-100)
        gathered = log_probs.gather(-1, target_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        scores: Tensor = (gathered * target_mask).sum(dim=-1)

        return [
            CandidatePrediction(candidate=candidate, log_likelihood=float(score))
            for (candidate, _), score in zip(encoded, scores, strict=True)
        ]

    def predict(self, prefix: str, candidates: Sequence[str]) -> CandidatePrediction:
        """Return the candidate with maximum teacher-forced log-likelihood."""

        return max(self.score(prefix, candidates), key=lambda result: result.log_likelihood)
