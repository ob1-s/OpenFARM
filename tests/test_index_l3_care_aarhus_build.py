"""Offline adversarial tests for the Aarhus L2->L3 historical builder.

Covers: monotone time causality (no later label may enter an earlier state),
true-negatives have no pseudo-onset dates, censoring is never recast into a
negative, first-day-flagged animals are excluded from the at-risk cohort,
(batch, pen) cohort handling with PenID reuse across batches, ambiguous
non-modal pen rows quarantined from both states and every pen history,
deterministic disjoint group splits, duplicate-key failure, and checksum
enforcement.
"""
from __future__ import annotations
import io, json, sys, unittest, zipfile
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import index_l3_care_build_aarhus_historical as B

TAIL_COLUMNS = ["Batch", "PenID", "Treatment", "PigsInPen", "Period", "Week", "Weekday",
                "Day", "Observer", "AnimalID", "Sex", "TailStatus_arrival",
                "IntactTail_arrival", "DamagedTail_arrival", "RegDateTime",
                "taildamage_fresh", "eardamage_fresh"]
FOULING_COLUMNS = ["Batch", "PenID", "Week", "Day", "Score", "PenFouling"]


def scan_row(batch, pen, day, animal, ts, tail=0, ear=0, treatment="Control",
             arrival_intact=1, arrival_damaged=0, sex="Castrated"):
    return {"Batch": batch, "PenID": pen, "Treatment": treatment,
            "PigsInPen": 2, "Period": 1, "Week": 1 + day // 7,
            "Weekday": "Monday", "Day": day, "Observer": "CKC",
            "AnimalID": animal, "Sex": sex, "TailStatus_arrival": "Intact_notDamaged",
            "IntactTail_arrival": arrival_intact, "DamagedTail_arrival": arrival_damaged,
            "RegDateTime": ts.isoformat(), "taildamage_fresh": tail,
            "eardamage_fresh": ear}


def fouling_row(batch, pen, day, score=0, fouled=0):
    return {"Batch": batch, "PenID": pen, "Week": 1 + day // 7, "Day": day,
            "Score": score, "PenFouling": fouled}


def make_tail(frame_rows):
    return B.read_tail_csv(pd.DataFrame(frame_rows, columns=TAIL_COLUMNS)
                           .to_csv(index=False).encode())


def make_fouling(frame_rows):
    return B.read_fouling_csv(pd.DataFrame(frame_rows, columns=FOULING_COLUMNS)
                              .to_csv(index=False).encode())


def run_builder(tail_rows, fouling_rows, horizon=7):
    tail = make_tail(tail_rows)
    fouling = make_fouling(fouling_rows)
    pmap, amb = B.primary_pen_map(tail)
    tail = tail.assign(primary_pen=[pmap[(int(b), int(a))] for b, a in
                                    zip(tail["Batch"], tail["AnimalID"])])
    keep = tail["PenID"] == tail["primary_pen"]
    quarantine = {"rows_quarantined": int((~keep).sum()),
                  "detail": tail.loc[~keep, ["Batch", "PenID", "AnimalID", "Day"]]
                  .to_dict(orient="records")}
    tail = tail[keep].copy()
    pen_animals = B.cohort_pen_animal_map(tail, pmap)
    states, prov = B.build_states(tail, fouling, pen_animals, horizon_days=horizon)
    prov["quarantine"] = quarantine
    if not states.empty:
        states, membership = B.attach_split(states)
    else:
        membership = {}
    return states, prov, membership, amb


class DuplicateAndChecksumGuards(unittest.TestCase):
    def test_duplicate_batch_animal_day_fails(self):
        day = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 50577, day), scan_row(1, 801, 5, 50577, day)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            make_tail(rows)

    def test_member_checksum_mismatch_fails(self):
        import hashlib
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as zf:
            zf.writestr("unrelated/extra.csv", b"x")
        original = B.ZIP_MD5
        try:
            B.ZIP_MD5 = hashlib.md5(raw.getvalue()).hexdigest()
            with self.assertRaisesRegex(ValueError, "missing archive member"):
                B.load_members(raw.getvalue())
        finally:
            B.ZIP_MD5 = original


class CausalContractTests(unittest.TestCase):
    def setUp(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        self.t0 = t0
        pen, animal = 801, 50577
        rows = [
            scan_row(1, pen, 5, animal, t0),
            scan_row(1, pen, 6, animal, t0 + timedelta(days=1)),
            scan_row(1, pen, 7, animal, t0 + timedelta(days=2), tail=1),
            scan_row(1, pen, 8, animal, t0 + timedelta(days=3)),
            scan_row(1, pen, 12, animal, t0 + timedelta(days=7)),
        ]
        self.states, self.prov, membership, amb = run_builder(rows, [])

    def test_later_scan_flag_inside_horizon_is_positive_with_event_ts(self):
        s = self.states
        self.assertEqual(len(s), 2)
        self.assertTrue((s["target_status"] == "positive").all())
        self.assertTrue((s["target"] == 1.0).all())
        event = pd.Timestamp(s["event_ts"].iloc[0])
        pred = pd.Timestamp(s["prediction_ts"].iloc[0])
        self.assertLessEqual(event - pred, timedelta(days=7))
        self.assertGreater(event - pred, timedelta(0))

    def test_window_without_later_coverage_beyond_grace_is_censored(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 42, t0), scan_row(1, 801, 8, 42, t0 + timedelta(days=3))]
        states, prov, membership, amb = run_builder(rows, [])
        self.assertEqual(len(states), 2)
        self.assertEqual(states[states["prediction_day"] == 5]["target_status"].iloc[0],
                         "censored")
        self.assertTrue(pd.isna(states[states["prediction_day"] == 5]["target"].iloc[0]))
        self.assertEqual(states[states["prediction_day"] == 8]["target_status"].iloc[0],
                         "not_measured")

    def test_unscorable_followup_window_is_not_measured_not_negative(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 77, t0)]
        states, prov, membership, amb = run_builder(rows, [])
        self.assertEqual(len(states), 1)
        self.assertEqual(states["target_status"].iloc[0], "not_measured")

    def test_positive_only_from_real_observed_row_with_event_ts(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 9, t0),
                scan_row(1, 801, 10, 9, t0 + timedelta(days=5), tail=1)]
        states, prov, membership, amb = run_builder(rows, [])
        pos = states[states["target_status"] == "positive"]
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos["target"].iloc[0], 1.0)
        event = pd.Timestamp(pos["event_ts"].iloc[0])
        pred = pd.Timestamp(pos["prediction_ts"].iloc[0])
        self.assertGreater(event, pred)
        self.assertLessEqual(event, pred + timedelta(days=7))

    def test_animal_flagged_on_first_observed_day_is_excluded(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 5, t0, tail=1),
                scan_row(1, 801, 6, 5, t0 + timedelta(days=1))]
        states, prov, membership, amb = run_builder(rows, [])
        self.assertEqual(len(states), 0)
        self.assertEqual(prov["animals_flagged_at_first_observed_day"], 1)
        self.assertEqual(prov["onset_animals_with_clean_prior"], 0)

    def test_after_first_flag_no_further_candidates_in_onset_cohort(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 11, t0),
                scan_row(1, 801, 8, 11, t0 + timedelta(days=3), tail=1),
                scan_row(1, 801, 9, 11, t0 + timedelta(days=4)),
                scan_row(1, 801, 16, 11, t0 + timedelta(days=11))]
        states, prov, membership, amb = run_builder(rows, [])
        self.assertEqual(len(states), 1)
        self.assertEqual(states["target_status"].iloc[0], "positive")
        self.assertNotIn(states["prediction_ts"].iloc[0],
                         {pd.Timestamp(t0 + timedelta(days=4)).isoformat(),
                          pd.Timestamp(t0 + timedelta(days=11)).isoformat()})

    def test_onset_after_horizon_is_negative_then_later_positive(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 3, t0),
                scan_row(1, 801, 11, 3, t0 + timedelta(days=6)),
                scan_row(1, 801, 16, 3, t0 + timedelta(days=11), tail=1)]
        states, _, _, _ = run_builder(rows, [])
        day5 = states[states["prediction_day"] == 5]
        day11 = states[states["prediction_day"] == 11]
        self.assertEqual(len(day5), 1)
        self.assertEqual(day5["target_status"].iloc[0], "negative")
        self.assertEqual(day5["target"].iloc[0], 0.0)
        self.assertEqual(day11["target_status"].iloc[0], "positive")


class CohortAndSplitTests(unittest.TestCase):
    def setUp(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = []
        pens = (801, 802, 803, 804)
        for b in (1, 2):
            for pen in pens:
                for k in range(4):
                    rows.append(scan_row(b, pen, 5, 1000 * b + 10 * pen + k, t0))
        fouling = [fouling_row(1, 801, 4, 125, 1)]
        states, prov, membership, amb = run_builder(rows, fouling)
        self.states = states
        self.membership = membership

    def test_penid_reuse_across_batches_stays_separate(self):
        self.assertEqual(set(self.states["split_group"]),
                         {"1-801", "1-802", "1-803", "1-804",
                          "2-801", "2-802", "2-803", "2-804"})
        self.assertEqual(len(self.membership), 8)

    def test_splits_are_batch_stratified(self):
        for batch in (1, 2):
            batch_keys = [k for k in self.membership if k.startswith("1-")] if batch == 1 else \
                         [k for k in self.membership if k.startswith("2-")]
            got = {self.membership[k] for k in batch_keys}
            self.assertEqual(got, {"test", "calibration", "train"})

    def test_split_is_deterministic(self):
        again = B.cohort_split(sorted(self.membership))
        self.assertEqual(again, self.membership)

    def test_pen_feature_counts_only_earlier_scans(self):
        s = self.states
        self.assertTrue((s["pen_mate_count"] == 3).all())
        self.assertTrue((s["pen_mate_fresh_tail_days_prior7"].iloc[0] == 0))

    def test_pen_fouling_joined_only_from_prior_days_same_batch_pen(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(1, 801, 5, 1, t0), scan_row(1, 802, 5, 2, t0)]
        fouling = [fouling_row(1, 801, 4, 125, 1), fouling_row(2, 801, 4, 300, 1)]
        states, _, _, _ = run_builder(rows, fouling)
        vals = {(int(r["batch"]), int(r["primary_pen"])): r["pen_fouling_score_prior"]
                for _, r in states.iterrows()}
        self.assertEqual(vals[(1, 801)], 125.0)
        self.assertTrue(pd.isna(vals[(1, 802)]))


class AmbiguousPenTests(unittest.TestCase):
    def test_single_day_in_wrong_pen_is_reported_not_silently_considered(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        rows = [scan_row(2, 831, 3, 54605, t0),
                scan_row(2, 821, 4, 54605, t0 + timedelta(days=1)),
                scan_row(2, 821, 5, 54605, t0 + timedelta(days=2))]
        tail = make_tail(rows)
        pmap, amb = B.primary_pen_map(tail)
        self.assertEqual(len(amb), 1)
        self.assertEqual(amb[0]["ambiguous_rows"], 1)
        self.assertEqual(pmap[(2, 54605)], 821)

    def test_quarantined_nonmodal_row_excluded_from_states_and_pen_history(self):
        t0 = pd.Timestamp("2023-03-06 09:00:00", tz="UTC")
        a_id, b_id = 54605, 54699
        rows = [
            scan_row(2, 831, 3, a_id, t0, tail=1),
            scan_row(2, 821, 4, a_id, t0 + timedelta(days=1)),
            scan_row(2, 821, 5, a_id, t0 + timedelta(days=2)),
            scan_row(2, 831, 5, b_id, t0 + timedelta(days=2)),
        ]
        states, prov, membership, amb = run_builder(rows, [])
        self.assertEqual(prov["quarantine"]["rows_quarantined"], 1)
        self.assertEqual(prov["quarantine"]["detail"][0], {
            "Batch": 2, "PenID": 831, "AnimalID": 54605, "Day": 3})
        self.assertEqual(amb[0]["ambiguous_rows"], 1)
        self.assertEqual(prov["animals_flagged_at_first_observed_day"], 0)
        self.assertEqual(len(states[states["prediction_day"] == 3]), 0)
        s_a = states[states["animal_id"] == a_id]
        self.assertEqual(sorted(s_a["prediction_day"].tolist()), [4, 5])
        s_b = states[states["animal_id"] == b_id]
        self.assertEqual(len(s_b), 1)
        self.assertEqual(float(s_b["pen_mate_fresh_tail_days_prior7"].iloc[0]), 0.0)
        self.assertEqual(float(s_b["pen_scans_prior_day"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
