"""Offline, source-free adversarial invariants for OpenFARM v0.2 compiler."""
from __future__ import annotations
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from openfarm_query_compiler_v02 import Compiler,Observation,Question

def q(qid="q",family="assay_a",value=1,**kw):
    return Question(question_id=qid,text=qid,primitive="Noul",
                    value=value,reference_family=family,
                    evidence_origin="independent_source",**kw)

def o(state="1",split="train",group="a",questions=None,features=None,**kw):
    return Observation(source="fixture",state_id=state,split=split,
            group_scope="animal",group_id=group,modalities=("sensor_numeric",),
            features=features if features is not None else {"observed_motion":10.0},
            questions=questions or [q()],**kw)

class ContractTests(unittest.TestCase):
    def test_real_independent_assays_same_state(self):
        compiler=Compiler()
        compiler.add(o(questions=[q("cort","cortisol"),q("oxyt","oxytocin")]))
        report=compiler.report()
        self.assertEqual(report["state_count"],1)
        self.assertEqual(report["observed_valid_question_labels"],2)
        self.assertEqual(report["mean_independently_measured_reference_families_per_state"],2)
        self.assertEqual(report["mean_non_derived_question_labels_per_state"],2)
    def test_one_multiclass_gt_with_many_binary_slices_counts_one(self):
        c=Compiler()
        choice=Question(question_id="behavior",text="Which observable behavior?",
             primitive="Choice",options=("grazing","resting","walking"),
             value="resting",reference_family="original_human_ethogram",
             evidence_origin="human_coder")
        slices=[q(f"is_{x}","original_human_ethogram",int(x=="resting"),
                  derived_from="behavior") for x in ("grazing","resting","walking")]
        c.add(o(questions=[choice,*slices]))
        report=c.report()
        self.assertEqual(report["observed_valid_question_labels"],4)
        self.assertEqual(report["mean_non_derived_question_labels_per_state"],1)
        self.assertEqual(report["mean_independently_measured_reference_families_per_state"],1)
        self.assertEqual(len(report["derived_question_ids"]),3)
    def test_same_sensor_horizons_do_not_fabricate_new_gt(self):
        c=Compiler()
        now=q("now","one_feeder")
        future=q("future","one_feeder",relation="future",reference_index=15)
        c.add(o(questions=[now,future],cutoff_index=0))
        self.assertEqual(c.report()["mean_independently_measured_reference_families_per_state"],1)
        self.assertEqual(c.report()["mean_non_derived_question_labels_per_state"],2)
    def test_masked_target_requires_explanation_and_absence(self):
        with self.assertRaisesRegex(ValueError,"Masked labels"):
            o(questions=[q(value=0,valid=False,mask_reason="sensor_off")]).validate()
        with self.assertRaisesRegex(ValueError,"reason"):
            o(questions=[q(value=None,valid=False)]).validate()
        c=Compiler()
        c.add(o(questions=[q(value=None,valid=False,mask_reason="sensor_off")]))
        r=c.report()
        self.assertEqual(r["masked_question_slots"],1)
        self.assertEqual(r["mean_independently_measured_reference_families_per_state"],0)
        self.assertEqual(r["masked_by_reason"]["sensor_off"],1)
    def test_prevent_animal_overlap(self):
        c=Compiler()
        c.add(o())
        with self.assertRaisesRegex(ValueError,"Leakage"):
            c.add(o(state="2",split="test"))
    def test_allow_animal_reuse_within_training(self):
        c=Compiler()
        c.add(o())
        c.add(o(state="2"))
        self.assertEqual(c.report()["state_count"],2)
        self.assertEqual(c.report()["group_count_by_source"]["fixture"],1)
    def test_duplicate_state_refused(self):
        c=Compiler();c.add(o())
        with self.assertRaisesRegex(ValueError,"Repeated"):
            c.add(o())
    def test_forbid_label_or_site_visible(self):
        for feature in ("feed","pain","labels","context","post_cortisol",
                        "future_eating","site","animal_id"):
            with self.subTest(feature=feature):
                with self.assertRaisesRegex(ValueError,"leakage"):
                    o(features={"observed_motion":1.0,feature:2.0}).validate()
    def test_future_timestamp_must_be_after_real_cutoff(self):
        obj=o(cutoff="2026-01-02T11:00:00",
           questions=[q(relation="future",reference_time="2026-01-02T10:00:00")])
        with self.assertRaisesRegex(ValueError,"Noncausal"):obj.validate()
        good=o(cutoff="2026-01-02T11:00:00",
             questions=[q(relation="future",reference_time="2026-01-02T11:30:00")])
        good.validate()
    def test_future_must_have_real_clock_or_relative_index(self):
        with self.assertRaisesRegex(ValueError,"exactly one"):
            o(questions=[q(relation="future")]).validate()
        with self.assertRaisesRegex(ValueError,"Noncausal"):
            o(questions=[q(relation="future",reference_index=5)],cutoff_index=5).validate()
    def test_reject_unknown_choice(self):
        choice=Question(question_id="choice",text="Behaviour?",primitive="Choice",
                        options=("a","b"),value="c",
                        reference_family="human",evidence_origin="coder")
        with self.assertRaisesRegex(ValueError,"Choice"):
            o(questions=[choice]).validate()
    def test_no_src_leak_between_export_files(self):
        with tempfile.TemporaryDirectory() as d:
            c=Compiler(local_dir=Path(d))
            c.add(o(questions=[q("pain_score","independent_vet")],
                    audit={"rater":"PRIVATE-RATER-XYZ"}))
            c.finish(Path(d)/"report.json")
            src=Path(d)/"fixture"/"train"
            visible=(src/"states.jsonl").read_text()
            refs=(src/"references.jsonl").read_text()
            prov=(src/"provenance.jsonl").read_text()
            self.assertNotIn('"value"',visible)
            self.assertNotIn('PRIVATE-RATER-XYZ',visible)
            self.assertIn('"value": 1',refs)
            self.assertIn('PRIVATE-RATER-XYZ',prov)
            self.assertIn('"question_ids"',visible)
    def test_opaque_row_ids_and_cutoff_hidden_from_model(self):
        with tempfile.TemporaryDirectory() as d:
            c=Compiler(local_dir=Path(d))
            state=o(state="animal_A:POSITIVE_EXPERIMENT",group="ANIMAL-PRIVATE",
                    cutoff="2026-02-10T11:59:00",questions=[
                    q("hormone_future","serum",relation="future",
                      reference_time="2026-02-10T12:45:00")])
            c.add(state)
            c.finish(Path(d)/"summary.json")
            base=Path(d)/"fixture"/"train"
            visible=(base/"states.jsonl").read_text()
            hidden=(base/"provenance.jsonl").read_text()
            for secret in ("POSITIVE_EXPERIMENT","ANIMAL-PRIVATE","2026-02-10T11:59:00"):
                self.assertNotIn(secret,visible)
                self.assertIn(secret,hidden)

    def test_distinct_modalities_are_not_new_label_families(self):
        c=Compiler()
        item=o(questions=[q("welfare_proxy","vet")])
        item.modalities=("expert_play_observation","before_cortisol_sensor")
        c.add(item)
        d=c.report()
        self.assertEqual(d["multi_modality_states"],1)
        self.assertEqual(d["mean_independently_measured_reference_families_per_state"],1)

if __name__=="__main__":unittest.main()
