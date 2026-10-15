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
    load_environment,
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


class TestKnownContaminatedEval(unittest.TestCase):
    def test_ungulate_quarantined_without_override(self):
        with self.assertRaisesRegex(ValueError, "retired"):
            load_environment(task="ungulate_valence")

    def test_multi_task_quarantined_without_override(self):
        with self.assertRaisesRegex(ValueError, "retired"):
            load_environment(task=["catmeows_context", "ungulate_valence"])

    def test_invalid_train_test_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid evaluation setup"):
            load_environment(task="catmeows_context", train_split="train", test_split="train")


class TestRetirementBoundaries(unittest.TestCase):
    def test_retired_six_not_registered(self):
        self.assertEqual(set(TASK_CONFIGS), {"catmeows_context"})

    def test_metadata_mode_is_retired(self):
        with self.assertRaisesRegex(ValueError, "recording/spectrogram"):
            load_environment(task="catmeows_context", modality="text")

    def test_mixed_splits_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported split combination"):
            load_environment(task="catmeows_context", train_split="train", test_split="test_raw")

    def test_unreviewed_override_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unreviewed dataset_name"):
            load_environment(task="catmeows_context", dataset_name="other/dataset")


class TestExactAnswers(unittest.TestCase):
    def setUp(self):
        self.config = TASK_CONFIGS["catmeows_context"]

    def test_source_aliases_still_work(self):
        self.assertEqual(normalize_answer("waiting", self.config), "waiting_for_food")
        self.assertEqual(normalize_answer("brushing", self.config), "brushing")
        self.assertEqual(normalize_answer("isolation", self.config), "isolation_unfamiliar_environment")

    def test_multiple_answers_are_not_accepted_as_one(self):
        ambiguous = normalize_answer("brushing or waiting", self.config)
        self.assertNotEqual(normalized_text(ambiguous), normalized_text("brushing"))
        self.assertNotEqual(normalized_text(ambiguous), normalized_text("waiting_for_food"))

    def test_extra_unrequested_text_is_not_accepted(self):
        prediction = normalize_answer("I think the answer is brushing", self.config)
        self.assertNotEqual(normalized_text(prediction), normalized_text("brushing"))


if __name__ == "__main__":
    unittest.main()
