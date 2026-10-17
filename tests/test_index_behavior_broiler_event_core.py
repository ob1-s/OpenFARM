"""Offline unit tests for the Big Broiler feather-pecking event task core.

Covers: label parsing (missingness preserved), causal window semantics (no
future-split leakage), observation-quality masks (unannotated != negative),
whole-video split determinism/disjointness/availability repair, episode merge
and overlap matching, geometry feature correctness, label-feature isolation
(changing source labels must NOT change features).
"""
from __future__ import annotations

import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import index_behavior_broiler_event_core as core


class LabelParsing(unittest.TestCase):
    def test_multilabel_and_missing(self):
        self.assertEqual(core.parse_action_labels("Eat,Drink"), ("Eat", "Drink"))
        self.assertEqual(core.parse_action_labels("Eat;Drink|Preening"), ("Eat", "Drink", "Preening"))
        self.assertEqual(core.parse_action_labels(float("nan")), ())
        self.assertEqual(core.parse_action_labels(None), ())
        self.assertEqual(core.parse_action_labels(""), ())
        self.assertEqual(core.parse_action_labels(" Feather_Pecking ,"), ("Feather_Pecking",))

    def test_missing_is_not_negative_surface(self):
        n = 2
        flags = core.per_frame_label_flags(
            pd.Series([np.nan, "Feather_Pecking"]), np.array([0, 1]), (core.FP_LABEL,), n)
        self.assertEqual(flags[core.FP_LABEL].tolist(), [False, True])

    def test_frame_flags_indexed_by_frame_not_row(self):
        # two rows at frame 1, none at frame 0/2 -> flags False at 0 and 2
        actions = pd.Series([np.nan, "Feather_Pecking", np.nan, "Feather_Pecking"])
        frames = np.array([1, 1, 2, 2])
        flags = core.per_frame_label_flags(
            actions, frames, (core.FP_LABEL,), 4)[core.FP_LABEL]
        self.assertEqual(flags.tolist(), [False, True, True, False])

    def test_row_frame_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            core.per_frame_label_flags(pd.Series(["Eat"]), np.array([7]),
                                       ("Eat",), 7)  # frame 7 invalid in a 7-frame video


class WindowSemantics(unittest.TestCase):
    def test_state_indices_stride_and_full_window(self):
        idx = core.state_indices(1000)
        self.assertEqual(idx[0], core.WINDOW_FRAMES - 1)
        self.assertTrue(np.all(np.diff(idx) == core.STRIDE_FRAMES))
        self.assertTrue(np.all(idx < 1000))
        self.assertEqual(len(core.state_indices(100)), 0)

    def test_window_any_is_inclusive_and_no_future_lookahead(self):
        idx = np.array([499, 450, 475])         # windows [250..499], [201..450], [226..475]
        for frame, expected in {
            460: [True, False, True],   # inside 499 and 475 windows, after 450
            201: [False, True, False],  # exactly the first frame of state 450
            200: [False, False, False],  # one frame before that window opens
            450: [True, True, True],    # inside both state 499 and 475 windows
            226: [False, True, True],   # exactly the first frame of state 475
            501: [False, False, False],  # future beyond all states
            499: [True, False, False],  # at state 499 exactly
        }.items():
            flags = np.zeros(1000, bool)
            flags[frame] = True
            self.assertEqual(core.window_any(idx, 250, flags).tolist(), expected,
                             f"flag frame {frame}")

    def test_window_count_and_coverage(self):
        rows = np.zeros(1000, int)
        rows[300:400] = 3  # tracked on exactly frames 300..399 (100 frames)
        idx = np.array([474, 549])
        cov = core.window_coverage(idx, rows, 250)
        self.assertAlmostEqual(cov[0], 100 / 250)
        self.assertAlmostEqual(cov[1], 100 / 250)
        cnt = core.window_count(idx, 250, (rows > 0))
        self.assertEqual(cnt.tolist(), [100, 100])

    def test_sliding_extreme_window_alignment(self):
        arr = np.zeros(1000, float)
        arr[300] = 5.0    # inside window of state 300..549 inclusive variants
        arr[100] = 7.0    # exactly the final frame of the window for state 349
        arr[901] = 9.0    # at the state frame itself
        idx = np.array([349, 899])
        mx = core._sliding_extreme(arr, idx, 250, "max")
        self.assertEqual(mx.tolist(), [7.0, 0.0])
        mn = core._sliding_extreme(np.where(arr == 0, 2.0, arr), idx, 250, "min")
        self.assertEqual(mn.tolist(), [2.0, 2.0])


class FrameGeometryAndFeatures(unittest.TestCase):
    def make_df(self):
        rows = []
        # frames 0..99: bird A static at (0.15,0.15) half-size 0.1x0.1;
        # bird B drifts +0.01 x/frame; frames 40..59 completely untracked
        for f in range(100):
            if 40 <= f < 60:
                continue
            rows.append((f, 1, 0.1, 0.1, 0.2, 0.2))
            shift = 0.01 * f
            rows.append((f, 2, 0.4 + shift, 0.4, 0.5 + shift, 0.5))
        return pd.DataFrame(rows, columns=["frame", "track", "x1", "y1", "x2", "y2"])

    def test_geometry_stats_and_duplicate_pruning(self):
        df = self.make_df()
        # duplicate (frame,track) with smaller box -> deterministic keep-last
        extra = pd.DataFrame([(10, 2, 0.9, 0.9, 0.95, 0.95)],
                             columns=["frame", "track", "x1", "y1", "x2", "y2"])
        df = pd.concat([df, extra], ignore_index=True)
        geom = core.build_frame_geometry(df, 100)
        self.assertEqual(geom["dropped_duplicate_boxes"], 1)
        self.assertEqual(int(geom["rows_per_frame"][10]), 2)
        # sorted and pruned: kept bird B box x1=0.5,x2=0.6 (shift 0.1)
        cx_a, cy_a = 0.15, 0.15
        cx_b, cy_b = 0.55, 0.45
        self.assertAlmostEqual(geom["nnd_min"][10], np.hypot(0.4, 0.3), places=5)
        self.assertAlmostEqual(geom["nnd_mean"][10], np.hypot(0.4, 0.3), places=5)
        self.assertEqual(geom["close_pair_frac"][10], 0.0)
        self.assertTrue(np.isnan(geom["nnd_min"][55]))
        self.assertEqual(int(geom["tracks_per_frame"][10]), 2)

    def test_duplicate_with_larger_area_keeps_larger_box(self):
        df = self.make_df()
        bigger = pd.DataFrame([(10, 2, 0.3, 0.3, 0.9, 0.9)],
                              columns=["frame", "track", "x1", "y1", "x2", "y2"])
        df = pd.concat([df, bigger], ignore_index=True)
        g = core.build_track_pivot(df)
        block = g[g["frame"] == 10]
        self.assertEqual(len(block), 2)
        self.assertAlmostEqual(float(block[block["track"] == 2]["cx"].iloc[0]), 0.6)

    def test_speed_median_std_and_moving_fraction(self):
        df = self.make_df()
        g = core.build_track_pivot(df)
        speed = core.compute_frame_speed(g, 100)
        self.assertIsNone(np.testing.assert_allclose(speed["speed_median"][30], 0.005, atol=1e-9))
        self.assertIsNone(np.testing.assert_allclose(speed["speed_std"][30], 0.005, atol=1e-9))
        self.assertEqual(speed["moving_frac"][30], 0.0)

    def test_features_shape_finiteness_and_label_isolation(self):
        df_base = self.make_df().assign(actions_ground=pd.Series([None] * 100, dtype="object"))
        X, idx, ctx = core.compute_features(df_base, 100)
        self.assertEqual(X.shape, (len(idx), len(core.FEATURE_NAMES)))
        self.assertTrue(np.isfinite(X).all())
        df_variant = df_base.copy()
        df_variant["actions_ground"] = None
        df_variant.loc[df_variant["frame"] >= 50, "actions_ground"] = "Feather_Pecking"
        X2, _, _ = core.compute_features(df_variant, 100)
        self.assertTrue(np.allclose(X, X2), "source action labels must not alter features")

    def test_untracked_gap_covers_partial_window(self):
        df = self.make_df()
        geom = core.build_frame_geometry(df, 100)
        idx = np.arange(49, 100, 25)  # with window=50: [0..49], [25..74], [50..99]
        cov = core.window_coverage(idx, geom["rows_per_frame"], 50)
        self.assertAlmostEqual(cov[1], 0.6, places=6)  # frames 40..59 untracked
        self.assertTrue(cov[1] < core.MIN_COVER <= cov[0])

    def test_more_than_cap_boxes_rejected(self):
        df = self.make_df()
        flood = pd.DataFrame(
            [(0, t, 0.0, 0.0, 0.01, 0.01) for t in range(core.MAX_TRACKS_PER_FRAME + 1)],
            columns=["frame", "track", "x1", "y1", "x2", "y2"])
        df = pd.concat([df, flood], ignore_index=True)
        with self.assertRaises(ValueError):
            core.build_frame_geometry(df, 100)


class EpisodeLogic(unittest.TestCase):
    def test_merge_runs_with_gap(self):
        flags = np.zeros(30, bool)
        flags[[0, 1, 5, 8, 20]] = True
        runs = core.merge_state_runs(flags, gap_states=2)
        self.assertEqual(runs, [(0, 1), (5, 8), (20, 20)])

    def test_episode_metrics_greedy_unique_reference(self):
        detected = [(0, 3), (10, 12), (20, 21)]
        reference = [(0, 2), (20, 30)]
        m = core.episode_metrics(detected, reference)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 0)
        m2 = core.episode_metrics([(0, 9), (9, 11)], [(0, 30)])
        self.assertEqual(m2["tp"], 1)
        self.assertEqual(m2["fp"], 1)


class SplitAssignment(unittest.TestCase):
    def stems_setup_fp(self, sizes=(49, 37, 6)):
        stems, setup_of, fp = [], {}, {}
        for s, n in enumerate(sizes, start=1):
            for i in range(n):
                stem = f"{i:03d}" + chr(96 + s)  # e.g. 000a, 000b - unique per setup
                stems.append(stem)
                setup_of[stem] = s
                fp[stem] = 100 if (s == 3 and i >= 1) else 0
        return stems, setup_of, fp

    def test_disjoint_deterministic_and_available(self):
        stems, setup_of, fp = self.stems_setup_fp()
        # balance FP availability into the fixed slots so no repair is triggered
        for stem, setup in setup_of.items():
            if setup in (1, 2) and stem[1].isdigit():
                i = int(stem[:3])
                if i in (3, 8, 15, 20, 27, 32, 39):
                    fp[stem] = 100
        a = core.assign_split(stems, setup_of, fp)
        b = core.assign_split(list(reversed(stems)), setup_of, fp)
        self.assertEqual(a, b)
        # setup 3 contributes one slotted test video; the documented availability
        # repair promotes one more FP-positive video into test and one into cal
        self.assertEqual(a[1], ["setup3:test", "setup3:cal"])
        assignment = a[0]
        self.assertEqual(len(assignment), 92)
        self.assertEqual(set(assignment.values()), {"train", "cal", "test"})
        test = sorted(k for k, v in assignment.items() if v == "test")
        cal = sorted(k for k, v in assignment.items() if v == "cal")
        self.assertEqual(len(test), 9)
        self.assertEqual(len(cal), 7)
        self.assertTrue(sum(1 for s in test if fp[s] > 0) >= 2)
        self.assertTrue(sum(1 for s in cal if fp[s] > 0) >= 1)

    def test_repair_moves_whole_videos_and_is_deterministic(self):
        stems, setup_of, fp = self.stems_setup_fp()
        fp_rich = {k: (500 if k.endswith("000a") else v) for k, v in fp.items()}
        rule = core.SplitRule(test_slots=(5,), cal_slots=(6,),
                              test_min_fp_videos=2, cal_min_fp_videos=1)
        assignment, repairs, totals = core.assign_split(list(stems), setup_of, fp_rich, rule)
        test = sorted(k for k, v in assignment.items() if v == "test")
        cal = sorted(k for k, v in assignment.items() if v == "cal")
        self.assertGreaterEqual(sum(1 for s in test if fp_rich[s] > 0), 2)
        self.assertGreaterEqual(sum(1 for s in cal if fp_rich[s] > 0), 1)
        self.assertEqual(len(set(test) & set(cal)), 0)
        self.assertTrue(repairs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
