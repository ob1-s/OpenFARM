"""Decision Index v0.1 scientific contract and expert-reference math regression tests."""
import copy,json,unittest
from pathlib import Path
from scripts.index_v01_validate import validate
from scripts.index_v01_wong_rater_audit import report

ROOT=Path("benchmarks/decision-index/v0.1/tasks.json")
class DecisionIndexTests(unittest.TestCase):
    def test_manifest_full(self):
        value=validate(json.loads(ROOT.read_text()))
        self.assertEqual(len(value["task_ids"]),8)
        self.assertGreaterEqual(value["approved_research_task_count"],3)

    def test_missing_expert_reference_rejected(self):
        doc=json.loads(ROOT.read_text())
        edited=copy.deepcopy(doc)
        del edited["tasks"][0]["metric_primary"]
        with self.assertRaises(AssertionError):
            validate(edited)

    def test_duplicated_task_id_rejected(self):
        doc=json.loads(ROOT.read_text());doc["tasks"][1]["id"]=doc["tasks"][0]["id"]
        with self.assertRaises(AssertionError):validate(doc)

    def test_wong_reference_three_votes_and_brier_floor(self):
        votes=[
            ["Immediate","Immediate","Immediate"],
            ["15 mins","30-60 mins","120 mins"],
            ["240 mins","240 mins","30-60 mins"]
        ]
        a=report(votes,"synthetic_fixture")
        self.assertEqual(a["n_expert_votes"],9)
        self.assertEqual(a["three_way_unanimous"],1)
        self.assertEqual(a["three_distinct_ratings"],1)
        self.assertAlmostEqual(a["empirical_vote_noise_to_one_hot_brier_floor"],(0+2/3+4/9)/3)
        self.assertEqual(a["majority_immediate"],1)

    def test_wong_rejects_unknown_label(self):
        with self.assertRaises(ValueError):
            report([["Immediate","15 mins","NONEXISTENT"]],"synthetic_fixture")
if __name__=="__main__":
    unittest.main()
