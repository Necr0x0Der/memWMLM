"""Rendering checks for the frozen-base reproduction baseline; no model download."""

import unittest

from memwmlm.data.repro import ReproExample
from memwmlm.evaluation.repro_baseline import (
    render_baseline_prefix,
    target_text,
)


class ReproBaselineRenderingTests(unittest.TestCase):
    def setUp(self):
        self.example = ReproExample(
            example_id="example-1",
            split="heldout",
            text="First sentence. Second sentence.",
        )

    def test_full_context_contains_the_passage(self):
        prefix = render_baseline_prefix(self.example, "full_context")
        self.assertIn("Read and retain the following passage", prefix)
        self.assertIn("First sentence. Second sentence.", prefix)
        self.assertIn("Reproduce the passage from memory.", prefix)

    def test_application_only_omits_the_passage(self):
        prefix = render_baseline_prefix(self.example, "application_only")
        self.assertNotIn("First sentence. Second sentence.", prefix)
        self.assertIn("Reproduce the passage from memory.", prefix)

    def test_sentence_index_changes_prefix_and_target(self):
        prefix = render_baseline_prefix(self.example, "application_only", sentence_index=2)
        self.assertIn("Sentence 2:", prefix)
        self.assertEqual(target_text(self.example, sentence_index=2), "Second sentence.")

    def test_whole_passage_target_is_unchanged(self):
        self.assertEqual(target_text(self.example), self.example.text)


if __name__ == "__main__":
    unittest.main()
