"""Dependency-free tests for the deterministic QA1 train extension."""

import tempfile
import unittest
from pathlib import Path

from scripts.extend_babi_qa1 import (
    build_dataset, episode_key, example_keys, generate_episode, split_episodes,
    statistics,
)


class ExtensionTests(unittest.TestCase):
    def test_generated_episode_supports_latest_queried_location(self):
        from random import Random
        episode = generate_episode(Random(12))
        self.assertEqual(len(episode), 15)
        self.assertEqual(len(example_keys(episode)), 5)
        last_location = {}
        last_id = {}
        for line in episode:
            line_id, content = line.split(" ", 1)
            if "\t" in content:
                question, answer, support = (part.strip() for part in content.split("\t"))
                actor = question.removeprefix("Where is ").removesuffix("?")
                self.assertEqual((answer, int(support)), (last_location[actor], last_id[actor]))
            else:
                tokens = content.split()
                self.assertNotEqual(last_location.get(tokens[0]), tokens[-1].rstrip("."))
                last_location[tokens[0]] = tokens[-1].rstrip(".")
                last_id[tokens[0]] = int(line_id)
        self.assertEqual(sum(statistics([episode])["question_support_movement_age"].values()), 5)

    def test_extension_preserves_original_and_avoids_test_overlap(self):
        from random import Random
        train_episode = generate_episode(Random(1))
        test_episode = generate_episode(Random(2))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.txt"
            test = root / "test.txt"
            train.write_text("\n".join(train_episode) + "\n", encoding="utf-8")
            test.write_text("\n".join(test_episode) + "\n", encoding="utf-8")
            result = build_dataset(train, test, root / "out", additional_examples=10, seed=123)
            merged_path = Path(result["merged_train"]["path"])
            generated_path = Path(result["generated_train"]["path"])
            self.assertTrue(merged_path.read_bytes().startswith(train.read_bytes()))
            generated = split_episodes(generated_path.read_text().splitlines())
            self.assertEqual(len(generated), 2)
            self.assertEqual(result["merged_train"]["stats"]["questions"], 15)
            self.assertFalse({episode_key(e) for e in generated} &
                             {episode_key(train_episode), episode_key(test_episode)})
            self.assertFalse(set().union(*(example_keys(e) for e in generated)) &
                             (example_keys(train_episode) | example_keys(test_episode)))
            with self.assertRaises(FileExistsError):
                build_dataset(train, test, root / "out", additional_examples=10, seed=123)

    def test_same_seed_larger_extension_preserves_smaller_prefix(self):
        from random import Random
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = root / "train.txt", root / "test.txt"
            train.write_text("\n".join(generate_episode(Random(1))) + "\n", encoding="utf-8")
            test.write_text("\n".join(generate_episode(Random(2))) + "\n", encoding="utf-8")
            small = build_dataset(train, test, root / "small", additional_examples=10, seed=123)
            large = build_dataset(train, test, root / "large", additional_examples=20, seed=123)
            small_bytes = Path(small["merged_train"]["path"]).read_bytes()
            large_bytes = Path(large["merged_train"]["path"]).read_bytes()
            self.assertTrue(large_bytes.startswith(small_bytes))
            self.assertEqual(large["merged_train"]["stats"]["questions"], 25)


if __name__ == "__main__":
    unittest.main()
