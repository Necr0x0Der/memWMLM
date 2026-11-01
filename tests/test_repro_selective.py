"""Selective-reproduction checks: per-sentence targets and per-example batching.

Hermetic (no model download): a character-level mock tokenizer stands in for the
real one. Covers the new selective-repro logic added on the ``selective-repro``
branch — ``target_text`` / ``choose_sentence_index`` and per-example
``make_repro_batch`` targets.
"""

import random
import unittest

import torch

from memwmlm.data.repro import ReproExample
from memwmlm.models.qwen import QwenChatPromptFormatter
from memwmlm.tasks.repro_memory import choose_sentence_index, target_text
from memwmlm.training.repro_reset_pair import make_repro_batch


class CharTokenizer:
    """Minimal hermetic tokenizer: one token per character (ids offset by +2 to
    leave pad=0 / eos=1 free). Supports the call signatures make_repro_batch uses
    and has no ``chat_template`` so the formatter returns the raw text."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.chat_template = None

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = [ord(c) + 2 for c in text]
        if add_special_tokens:
            ids = [self.eos_token_id] + ids
        return ids

    def __call__(self, text, add_special_tokens=True, padding=False,
                 return_tensors=None, max_length=None, truncation=False):
        if isinstance(text, str):
            ids = self._encode(text, add_special_tokens)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids])}
            return _Encoded(ids)
        batch = [self._encode(t, add_special_tokens) for t in text]
        if padding:
            maxlen = max(len(b) for b in batch)
            batch = [b + [self.pad_token_id] * (maxlen - len(b)) for b in batch]
        input_ids = torch.tensor(batch)
        attention_mask = torch.tensor(
            [[0 if x == self.pad_token_id else 1 for x in row] for row in batch]
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True) -> str:
        chars = []
        for i in ids:
            if skip_special_tokens and i == self.eos_token_id:
                continue
            chars.append(chr(i - 2))
        return "".join(chars)


class _Encoded:
    def __init__(self, input_ids) -> None:
        self.input_ids = input_ids


_TEXT = "Alice is in the garden. Bob went to the kitchen. The cat sat on the mat."


class SelectiveReproLogicTests(unittest.TestCase):
    def test_target_text_whole_and_sentence(self):
        self.assertEqual(target_text(_TEXT, None), _TEXT)
        self.assertEqual(target_text(_TEXT, 1), "Alice is in the garden.")
        self.assertEqual(target_text(_TEXT, 3), "The cat sat on the mat.")

    def test_choose_sentence_index_excludes_last_by_default(self):
        rng = random.Random(0)
        seen = set()
        for _ in range(4000):
            idx = choose_sentence_index(_TEXT, rng, whole_prob=0.5, exclude_last=True)
            seen.add(idx)
        # The final (possibly truncated) sentence (index 3) must never be sampled.
        self.assertNotIn(3, seen)
        self.assertIn(None, seen)  # whole-passage option is present

    def test_choose_sentence_index_include_last_and_whole_prob(self):
        rng = random.Random(1)
        counts = {}
        for _ in range(4000):
            idx = choose_sentence_index(_TEXT, rng, whole_prob=0.0, exclude_last=False)
            counts[idx] = counts.get(idx, 0) + 1
        # With whole_prob=0 and include_last, all three sentences appear roughly evenly.
        self.assertEqual(set(counts), {1, 2, 3})
        for value in counts.values():
            self.assertAlmostEqual(value, 4000 / 3, delta=300)


class SelectiveReproBatchTests(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()
        self.fmt = QwenChatPromptFormatter(self.tok)

    def _target_region_text(self, labels_row):
        real = [int(i) for i in labels_row if int(i) != -100]
        return self.tok.decode(real, skip_special_tokens=True)

    def test_make_repro_batch_mixed_indices(self):
        examples = [
            ReproExample("w1", "train", _TEXT),
            ReproExample("w2", "train", "The sky is blue. Grass is green. Birds fly high."),
        ]
        # row0 -> whole passage, row1 -> sentence 1 (max_tokens large enough not to truncate)
        batch = make_repro_batch(
            examples, self.tok, self.fmt, torch.device("cpu"),
            sentence_indices=[None, 1], max_tokens=256,
        )
        self.assertEqual(batch.application_input_ids.shape[0], 2)
        self.assertEqual(self._target_region_text(batch.labels[0]), _TEXT)
        self.assertEqual(self._target_region_text(batch.labels[1]), "The sky is blue.")

    def test_make_repro_batch_scalar_matches_per_example(self):
        examples = [ReproExample("w", "train", _TEXT)]
        scalar = make_repro_batch(
            examples, self.tok, self.fmt, torch.device("cpu"),
            sentence_index=3, max_tokens=256,
        )
        per = make_repro_batch(
            examples, self.tok, self.fmt, torch.device("cpu"),
            sentence_indices=[3], max_tokens=256,
        )
        self.assertEqual(
            self._target_region_text(scalar.labels[0]),
            self._target_region_text(per.labels[0]),
        )
        self.assertEqual(self._target_region_text(per.labels[0]), "The cat sat on the mat.")

    def test_make_repro_batch_rejects_length_mismatch(self):
        examples = [ReproExample("w1", "train", _TEXT), ReproExample("w2", "train", _TEXT)]
        with self.assertRaises(ValueError):
            make_repro_batch(
                examples, self.tok, self.fmt, torch.device("cpu"),
                sentence_indices=[1],  # only one index for two examples
                max_tokens=256,
            )


if __name__ == "__main__":
    unittest.main()
