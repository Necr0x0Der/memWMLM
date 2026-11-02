"""Selective-reproduction checks for the frozen-base baseline evaluator.

Hermetic (no model download): a char-level tokenizer plus a tiny deterministic
mock model exercise the per-example ``sentence_indices`` path and the
``per_position`` breakdown added to ``evaluate_repro_baseline``. Mirrors the
selective mode the memory sidecar gained on the ``selective-repro`` branch.
"""

import unittest

import torch

from memwmlm.data.repro import ReproExample
from memwmlm.evaluation.repro_baseline import (
    _index_key,
    evaluate_repro_baseline,
    render_baseline_prefix,
    target_text,
)

_TEXT_A = "Alpha beta gamma."
_TEXT_B = "One two three four."


class CharTokenizer:
    """One token per character (ids offset by +2 to keep pad=0 / eos=1 free);
    no ``chat_template`` so the formatter returns the raw text."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.chat_template = None

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        if isinstance(text, str):
            ids = [ord(c) + 2 for c in text]
            if add_special_tokens:
                ids = [self.eos_token_id] + ids
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids])}
            return _Encoded(ids)
        batch = [[self.eos_token_id] + [ord(c) + 2 for c in t] for t in text]
        return {"input_ids": torch.tensor(batch)}

    def decode(self, ids, skip_special_tokens=True) -> str:
        return "".join(
            chr(int(i) - 2) for i in ids if not (skip_special_tokens and i == self.eos_token_id)
        )


class _Encoded:
    def __init__(self, input_ids) -> None:
        self.input_ids = input_ids


class _Out:
    def __init__(self, logits) -> None:
        self.logits = logits
        self.past_key_values = None


class MockQwen:
    """Deterministic mock for the full-mode baseline evaluator: ``generate``
    echoes the prefix then appends 4 repeat tokens; ``__call__`` returns uniform
    (zero) logits so the target log-likelihood is computable. vocab is large
    enough that char ids never exceed it (no gather out-of-range)."""

    def __init__(self, device="cpu", repeat=4, vocab=256) -> None:
        self.device = device
        self.repeat = repeat
        self.vocab = vocab

    def generate(self, **encoded):
        prefix = encoded["input_ids"]
        extra = torch.full((prefix.shape[0], self.repeat), 5, dtype=torch.long, device=self.device)
        return torch.cat([prefix, extra], dim=1)

    def __call__(self, input_ids, attention_mask, use_cache, return_dict):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab, device=self.device)
        return _Out(logits)


class BaselineIndexKeyTests(unittest.TestCase):
    def test_index_key(self):
        self.assertEqual(_index_key(None), "whole")
        self.assertEqual(_index_key(1), "1")
        self.assertEqual(_index_key(3), "3")


class BaselineSelectiveRenderingTests(unittest.TestCase):
    def test_full_context_with_sentence_index(self):
        ex = ReproExample("e", "heldout", "First sentence. Second sentence.")
        prefix = render_baseline_prefix(ex, "full_context", sentence_index=2)
        self.assertIn("Sentence 2:", prefix)                          # the selective question
        self.assertIn("First sentence. Second sentence.", prefix)     # passage still visible
        self.assertEqual(target_text(ex, sentence_index=2), "Second sentence.")

    def test_application_only_with_sentence_index(self):
        ex = ReproExample("e", "heldout", "First sentence. Second sentence.")
        prefix = render_baseline_prefix(ex, "application_only", sentence_index=1)
        self.assertIn("Sentence 1:", prefix)
        self.assertNotIn("First sentence. Second sentence.", prefix)  # passage absent


class BaselineSelectiveEvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()
        self.exA = ReproExample("a", "train", _TEXT_A)
        self.exB = ReproExample("b", "train", _TEXT_B)

    def test_per_example_indices_group_per_position(self):
        model = MockQwen()
        # row0 -> sentence 1, row1 -> whole passage
        result = evaluate_repro_baseline(
            model, self.tok, [self.exA, self.exB],
            variant="full_context", sentence_indices=[1, None],
            max_new_tokens=8, progress_every=1000,
        )
        self.assertEqual(result.examples, 2)
        self.assertIsNotNone(result.per_position)
        keys = set(result.per_position)
        self.assertIn("1", keys)      # sentence 1
        self.assertIn("whole", keys)  # whole passage
        self.assertEqual(result.per_position["1"]["examples"], 1)
        self.assertEqual(result.per_position["whole"]["examples"], 1)

    def test_scalar_matches_per_example(self):
        model = MockQwen()
        scalar = evaluate_repro_baseline(
            model, self.tok, [self.exA], variant="full_context",
            sentence_index=1, max_new_tokens=8, progress_every=1000,
        )
        per = evaluate_repro_baseline(
            model, self.tok, [self.exA], variant="full_context",
            sentence_indices=[1], max_new_tokens=8, progress_every=1000,
        )
        self.assertEqual(
            scalar.per_position["1"]["examples"], per.per_position["1"]["examples"])
        self.assertEqual(
            scalar.per_position["1"]["mean_token_acc"], per.per_position["1"]["mean_token_acc"])


if __name__ == "__main__":
    unittest.main()
