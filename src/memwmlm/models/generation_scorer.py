"""Free-generation scoring for finite-answer tasks such as bAbI."""

from __future__ import annotations

import re
from typing import Sequence

import torch

from memwmlm.models.candidate_scorer import CandidatePrediction


class FreeGenerationScorer:
    """Generate an answer, then accept only an exact normalized candidate.

    Candidate likelihood remains the preferred diagnostic for the first memory
    study. Free generation shows whether the frozen model follows the requested
    one-word output contract.
    """

    def __init__(self, model: object, tokenizer: object, max_new_tokens: int = 16) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def predict(self, prefix: str, candidates: Sequence[str]) -> CandidatePrediction:
        if not candidates:
            raise ValueError("At least one candidate is required")

        encoded = self.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        output_ids = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        continuation = self.tokenizer.decode(
            output_ids[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        normalized = _normalize_answer(continuation)
        candidate_lookup = {_normalize_answer(candidate): candidate for candidate in candidates}
        # A non-candidate deliberately remains wrong instead of being extracted
        # from explanatory prose after generation.
        predicted = candidate_lookup.get(normalized, continuation.strip())
        return CandidatePrediction(candidate=predicted, log_likelihood=float("nan"))


def _normalize_answer(text: str) -> str:
    """Normalize only surrounding formatting; do not extract answers from prose."""

    return re.sub(r"\s+", " ", text.strip().strip(".,;:!?`'\"“”")).casefold()
