"""Offline adversarial test cases for as-of-time and future censoring.

No source media or model packages required.
"""
from __future__ import annotations
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_temporal_contract_v02 import (
    pre_only_features,assert_pre_only,assert_future_clock,
    future_feeder_bout_targets,PIGWEB_AVAILABILITY)

class HistoricalContractTests(unittest.TestCase):
    def test_pre_stimulus_retains_only_source_verified_input(self):
        original={key:float(i) for i,key in enumerate(PIGWEB_AVAILABILITY)}
        filtered=pre_only_features(original)
        self.assertEqual(set(filtered),{"pre_cortisol_ng_ml","pre_oxytocin_pg_ml"})
        assert_pre_only(filtered)
    def test_behavior_during_exposure_cannot_be_recast_as_pre(self):
        with self.assertRaisesRegex(ValueError,"later/unknown-time"):
            assert_pre_only({"pre_cortisol_ng_ml":2.,
                             "observed_locomotor_play":1.})
    def test_unknown_timing_must_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"Undocumented"):
            pre_only_features({"pre_cortisol_ng_ml":1.,"invented_study_field":7})
    def test_post_sample_must_be_later(self):
        with self.assertRaisesRegex(ValueError,"strictly later"):
            assert_future_clock("2026-02-14T10:00:00","2026-02-14T09:59:00")
        with self.assertRaisesRegex(ValueError,"strictly later"):
            assert_future_clock("2026-02-14T10:00:00","2026-02-14T10:00:00")
        assert_future_clock("2026-02-14T10:00:00","2026-02-14T10:50:00")
    def test_future_bout_can_be_yes_after_clean_zero(self):
        feed=[0.]*36;feed[7]=1.;feed[8]=2.
        self.assertEqual(future_feeder_bout_targets(feed,0,15),(0,1))
    def test_active_bout_not_new_bout(self):
        feed=[0.,1.,1.,1.,0.,0.]
        self.assertEqual(future_feeder_bout_targets(feed,2,3),(1,0))
    def test_missing_instrument_window_is_masked_not_negative(self):
        feed=[0.]*26;feed[8]=None;feed[11]=1.
        self.assertEqual(future_feeder_bout_targets(feed,0,15),(0,None))
    def test_record_boundary_masks_future(self):
        self.assertEqual(future_feeder_bout_targets([0.]*16,2,15),(0,None))
    def test_missing_current_masks_now_and_future(self):
        feed=[0.]*22;feed[5]=None
        self.assertEqual(future_feeder_bout_targets(feed,5,15),(None,None))
    def test_padded_or_nonfinite_reference_is_not_observed(self):
        feed=[0.]*30;feed[2]=float("nan")
        self.assertEqual(future_feeder_bout_targets(feed,0,15),(0,None))
    def test_feeder_end_boundary_just_valid(self):
        feed=[0.]*16;feed[15]=1.
        self.assertEqual(future_feeder_bout_targets(feed,0,15),(0,1))

if __name__=="__main__":unittest.main()
