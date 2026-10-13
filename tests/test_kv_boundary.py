"""Tokenizer-boundary checks require neither CUDA nor model downloads."""

import unittest

from memwmlm.evaluation.kv_boundary import split_at_question


class OffsetTokenizer:
    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(self.offsets))), "offset_mapping": self.offsets}


class SplitTests(unittest.TestCase):
    def test_preserves_full_token_sequence_and_special_tokens(self):
        text = "Story: A.\n\nQuestion: Where?\nAnswer:"
        boundary = text.index("Question: ")
        offsets = [(0, 0), (0, 6), (6, boundary), (boundary, boundary + 9),
                   (boundary + 9, len(text)), (0, 0)]
        left, right = split_at_question(OffsetTokenizer(offsets), text)
        self.assertEqual(left + right, list(range(len(offsets))))
        self.assertEqual((left, right), ([0, 1, 2], [3, 4, 5]))

    def test_moves_cross_boundary_token_whole_to_application(self):
        text = "Story: A.\n\nQuestion: Where?"
        boundary = text.index("Question: ")
        left, right = split_at_question(OffsetTokenizer([(0, 6),
                                                          (6, boundary + 1),
                                                          (boundary + 1, len(text))]), text)
        self.assertEqual((left, right), ([0], [1, 2]))

    def test_rejects_missing_question(self):
        with self.assertRaisesRegex(ValueError, "no application question"):
            split_at_question(OffsetTokenizer([(0, 5)]), "Story")


if __name__ == "__main__":
    unittest.main()
