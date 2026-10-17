"""Offline unit tests for the GrimACE L2 library (no archive, no network needed)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import index_l2_grimace_lib as lib


class TestLabelParsing(unittest.TestCase):
    def test_parse_label_cell(self):
        self.assertIsNone(lib.parse_label_cell(""))
        self.assertIsNone(lib.parse_label_cell(" "))
        self.assertIsNone(lib.parse_label_cell("NA"))
        self.assertIsNone(lib.parse_label_cell("NR"))
        self.assertIsNone(lib.parse_label_cell("2.0"))  # non-integer -> not rateable
        for v in (0, 1, 2):
            self.assertEqual(lib.parse_label_cell(str(v)), v)
        with self.assertRaises(ValueError):
            lib.parse_label_cell("3")

    def test_externals_csv_roundtrip(self):
        text = ("folder,image,eye,nose,cheek,ear,whisker\n"
                "psFC3346day1_4h,img02922.png,2,1,1,2,\n"
                "psFC3346day1_4h,img02923.png,0,0,0,0,1\n")
        rows = lib.parse_externals_csv(text, "clara")
        self.assertEqual(rows[0], ("psFC3346day1_4h", 2922,
                                   {"eye": 2, "nose": 1, "cheek": 1, "ear": 2, "whisker": None}))
        self.assertEqual(rows[1][2], {"eye": 0, "nose": 0, "cheek": 0, "ear": 0, "whisker": 1})

    def test_gt_train_csv_splits_raters_and_drops_automatic(self):
        text = (
            "video_name,frame_index,eye,nose,cheek,ear,whisker,average\n"
            "automatic,gt1_psFC3340day1,101,0,0.9,0.0,0,0,0.4\n"
            "gt1_psFC3340day1,101,1,,0,1,1,0.6\n"
            "gt2_a3694_m_0h_face0,55,0,0,0,0,2,0.4\n")
        rows = lib.parse_gt_train_csv(text)
        self.assertEqual(rows, [
            ("gt1", "psFC3340day1", 101, {"eye": 1, "nose": None, "cheek": 0, "ear": 1, "whisker": 1}),
            ("gt2", "a3694_m_0h_face0", 55, {"eye": 0, "nose": 0, "cheek": 0, "ear": 0, "whisker": 2}),
        ])

    def test_gt3_hohlbaum_rows(self):
        text = (
            "annotator,video_name,animal_id,treatment,timepoint,frame_index,eye,nose,cheek,ear,whisker,average\n"
            "automatic,a12_c_4h_front_39,12,c,4h,39,0,0,0,0,0,0.0\n"
            "hohlbaum,a12_c_4h_front_39,12,c,4h,217,1,0,0,2,1,0.8\n")
        rows = lib.parse_gt3_hohlbaum_individual(text)
        self.assertEqual(rows[0], ("a12_c_4h_front_39", 217,
                                   {"eye": 1, "nose": 0, "cheek": 0, "ear": 2, "whisker": 1}))

    def test_averaged_csv_strict_unknown_session(self):
        text = ("annotator,video_name,animal_id,treatment,timepoint,num_frames\n"
                "automatic,gt1_psFC3340day1,3340,m,0h,40\n")
        meta = lib.parse_averaged_csv(text, {"gt1_psFC3340day1"})
        self.assertEqual(meta["gt1_psFC3340day1"]["animal_id"], "3340")
        with self.assertRaises(ValueError):
            lib.parse_averaged_csv(text, {"other"})


class TestAnimalCanonicalization(unittest.TestCase):
    def test_folder_tag_anomaly_uses_averaged_id(self):
        self.assertEqual(lib.canonical_animal_exp1("psFC3440day0", "3340"), "3340")
        self.assertEqual(lib.canonical_animal_exp1("psFC3441day1", "3341"), "3341")
        self.assertEqual(lib.canonical_animal_exp1("psFC3346day2", "3346"), "3346")
        self.assertEqual(lib.canonical_animal_exp1("psFC3346day2", None), "3346")
        with self.assertRaises(ValueError):
            lib.canonical_animal_exp1("psFCx", None)

    def test_exp3_session_parse(self):
        self.assertEqual(lib.session_to_exp3_mouse("a12_c_4h_front_39"), "12")
        self.assertEqual(lib.session_to_exp3_timepoint("a12_c_4h_front_39"), ("c", "4h"))
        self.assertEqual(lib.session_to_exp3_timepoint("a26_c_p24_front_3"), ("c", "p24"))
        with self.assertRaises(ValueError):
            lib.session_to_exp3_timepoint("a12_d_4h_front_39")


class TestSplits(unittest.TestCase):
    def test_numeric_aware_deterministic_split(self):
        animals = ["12", "3", "21", "9", "26", "17", "18", "13", "4"]
        split = lib.stratified_animal_split(animals, {a: "c" for a in animals}, n_test=2, n_cal=2)
        self.assertEqual(split, {"train": ["3", "4", "9", "12", "13"],
                                 "cal": ["17", "18"], "test": ["21", "26"]})
        again = lib.stratified_animal_split(list(reversed(animals)), {a: "c" for a in animals},
                                            n_test=2, n_cal=2)
        self.assertEqual(split, again)

    def test_treatment_stratified_split_matches_frozen_exp1_layout(self):
        animals = ["3340", "3341", "3342", "3343", "3346", "3347", "3348", "3349", "3354",
                   "3355", "3367", "3369", "3371", "3376", "3379"]
        tr = {a: ("m" if a in {"3340", "3342", "3346", "3349", "3354", "3369", "3371", "3379"} else "b")
              for a in animals}
        split = lib.stratified_animal_split(animals, tr, n_test=2, n_cal=1)
        lib.validate_split(split)
        self.assertEqual(split["test"], ["3367", "3371", "3376", "3379"])
        self.assertEqual(split["cal"], ["3355", "3369"])
        self.assertEqual(split["train"], ["3340", "3341", "3342", "3343", "3346", "3347",
                                          "3348", "3349", "3354"])

    def test_split_digests_stable(self):
        split = {"train": ["a", "b"], "cal": ["c"], "test": ["d"]}
        self.assertEqual(lib.split_membership_digest(split),
                         lib.split_membership_digest({"test": ["d"], "train": ["a", "b"], "cal": ["c"]}))


class TestConsensus(unittest.TestCase):
    def test_strict_majority(self):
        self.assertEqual(lib.per_au_consensus({"gt1": 2, "clara": 2, "yvonne": 1, "hohlbaum": 2}), 2)
        self.assertEqual(lib.per_au_consensus({"gt1": 0, "clara": 0, "yvonne": 0}), 0)
        self.assertEqual(lib.per_au_consensus({"clara": 2, "yvonne": 1}), "NO_CONSENSUS")

    def test_not_rateable_when_all_blank_never_zero(self):
        self.assertEqual(lib.per_au_consensus({"gt1": None, "clara": None}), "NOT_RATEABLE")
        self.assertEqual(lib.per_au_consensus({}, min_raters=1), "NO_CONSENSUS")

    def test_min_raters(self):
        self.assertEqual(lib.per_au_consensus({"gt1": 1, "clara": None}, min_raters=2), "NO_CONSENSUS")
        self.assertEqual(lib.per_au_consensus({"gt1": 1}, min_raters=1), 1)


class TestDigests(unittest.TestCase):
    def test_manifest_digest_order_independent(self):
        a = [{"key": ["exp1", "s", 5], "labels": {"gt1": {"eye": 2, "nose": None}}}]
        b = [{"key": ["exp1", "s", 5], "labels": {"gt1": {"nose": None, "eye": 2}}}]
        self.assertEqual(lib.manifest_digest(a), lib.manifest_digest(b))
        self.assertNotEqual(lib.manifest_digest(a), lib.manifest_digest([{"key": ["exp1", "s", 6],
                                                                          "labels": a[0]["labels"]}]))

    def test_sha256_bytes(self):
        self.assertEqual(lib.sha256_bytes(b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
