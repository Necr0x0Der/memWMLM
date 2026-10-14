"""Pure configuration tests for reset-pair reader placement."""

import unittest

from memwmlm.models.injection_indices import resolve_injection_indices


class InjectionIndicesTests(unittest.TestCase):
    def test_legacy_last_four_and_last_one(self):
        self.assertEqual(resolve_injection_indices(28, 4), (24, 25, 26, 27))
        self.assertEqual(resolve_injection_indices(28, 1), (27,))

    def test_explicit_early_and_spaced_override_final_n(self):
        self.assertEqual(resolve_injection_indices(28, 4, [0, 9, 18, 27]), (0, 9, 18, 27))
        self.assertEqual(resolve_injection_indices(28, 4, [0, 1, 2, 3]), (0, 1, 2, 3))
        self.assertEqual(resolve_injection_indices(28, 4, [27]), (27,))

    def test_rejects_invalid_indices(self):
        for indices in ([], [-1, 2], [0, 28], [0, 0], [9, 0], [1.0], [True], "0,9,18,27"):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                resolve_injection_indices(28, 4, indices)

    def test_legacy_validation(self):
        for count in (0, -1, 29):
            with self.subTest(count=count), self.assertRaises(ValueError):
                resolve_injection_indices(28, count)


if __name__ == "__main__":
    unittest.main()
