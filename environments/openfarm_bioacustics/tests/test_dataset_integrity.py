"""Regression checks for data integrity and unambiguous evaluation answers.

Run from environments/openfarm_bioacustics:
    python -m unittest discover -s tests -v

Requires the bioacoustics environment's dependencies to be installed.
These tests mock Hub access and do not need network or model inference.
"""

import unittest
from unittest.mock import patch

from openfarm_bioacustics.openfarm_bioacustics import (
    load_source_dataset,
    normalize_answer,
    normalized_text,
)
from openfarm_bioacustics.task_registry import TASK_CONFIGS


class FakeDataset:
    def __init__(self, columns):
        self.column_names = columns
        self.casts = []

    def cast_column(self, name, feature):
        self.casts.append((name, feature))
        return self


class TestSplitLoading(unittest.TestCase):
    @patch("datasets.load_dataset")
    def test_missing_test_split_never_falls_back_to_train(self, load):
        load.side_effect = ValueError("Unknown split 'test'")
        with self.assertRaisesRegex(RuntimeError, "requested split 'test'"):
            load_source_dataset("example/has-only-train", "test", "revision-1")

        load.assert_called_once_with(
            "example/has-only-train", split="test", revision="revision-1"
        )

    @patch("datasets.load_dataset")
    def test_transient_error_never_falls_back_to_train(self, load):
        load.side_effect = ConnectionError("Hub unavailable")
        with self.assertRaises(RuntimeError) as cm:
            load_source_dataset("example/data", "test", None)

        self.assertIsInstance(cm.exception.__cause__, ConnectionError)
        load.assert_called_once_with("example/data", split="test", revision=None)

    @patch("datasets.load_dataset")
    def test_success_preserves_requested_split_and_casts_media(self, load):
        ds = FakeDataset(["audio", "spectrogram", "label"])
        load.return_value = ds

        actual = load_source_dataset("example/data", "test_raw", "a1b2c3")
        self.assertIs(actual, ds)
        load.assert_called_once_with(
            "example/data", split="test_raw", revision="a1b2c3"
        )
        self.assertEqual([name for name, _ in ds.casts], ["audio", "spectrogram"])
        self.assertFalse(ds.casts[0][1].decode)
        self.assertTrue(ds.casts[1][1].decode)

    @patch("datasets.load_dataset")
    def test_dataset_without_media_is_unchanged(self, load):
        ds = FakeDataset(["label", "id"])
        load.return_value = ds

        self.assertIs(load_source_dataset("example/text", "train", None), ds)
        self.assertEqual(ds.casts, [])


class TestExactAnswers(unittest.TestCase):
    def setUp(self):
        self.config = TASK_CONFIGS["soundwel_valence"]

    def test_source_aliases_still_work(self):
        self.assertEqual(normalize_answer("Neg", self.config), "Negative")
        self.assertEqual(normalize_answer("Positive", self.config), "Positive")
        self.assertEqual(normalize_answer("positive", self.config), "Positive")

    def test_multiple_answers_are_not_accepted_as_one(self):
        ambiguous = normalize_answer("Positive or Negative", self.config)
        self.assertNotEqual(normalized_text(ambiguous), normalized_text("Positive"))
        self.assertNotEqual(normalized_text(ambiguous), normalized_text("Negative"))

    def test_extra_unrequested_text_is_not_accepted(self):
        prediction = normalize_answer("I think the answer is Negative", self.config)
        self.assertNotEqual(normalized_text(prediction), normalized_text("Negative"))


if __name__ == "__main__":
    unittest.main()
