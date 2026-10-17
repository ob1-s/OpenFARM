"""Synthetic unit tests of expert-vote statistics, not real-vet accuracy claims."""
import importlib.util
import unittest
from pathlib import Path
P=Path(__file__).parents[1]/'scripts'/'index_l3_vet_evaluate.py'
spec=importlib.util.spec_from_file_location('index_l3_vet_evaluate',P)
E=importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
CASES=[dict(case_id='a',veterinarian_votes=['red','red','orange']),dict(case_id='b',veterinarian_votes=['green','blue',None])]

class Evaluate(unittest.TestCase):
    def test_always_red_has_zero_undertriage_but_overtriages(self):
        r=E.evaluate(CASES,E.always_red(CASES))
        self.assertEqual(r['n_unique_evaluable_cases'],2)
        self.assertEqual(r['veterinarian_vote_weighted_undertriage_rate'],0)
        self.assertGreater(r['veterinarian_vote_weighted_overtriage_rate'],0)
        self.assertEqual(r['sensitivity_for_majority_red_cases'],1)
    def test_majority_blue_will_undertriage_red(self):
        pred={'a':{'red':0,'orange':0,'yellow':0,'green':0,'blue':1},'b':{'red':0,'orange':0,'yellow':0,'green':0,'blue':1}}
        r=E.evaluate(CASES,pred)
        self.assertGreater(r['veterinarian_vote_weighted_undertriage_rate'],0)
        self.assertEqual(r['sensitivity_for_majority_red_cases'],0)
    def test_missing_predictions_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'no prediction'):
            E.evaluate(CASES,{'a':E.always_red(CASES)['a']})
    def test_invalid_probability_fail_closed(self):
        pp=E.always_red(CASES)
        pp['a']['red']=0.5
        with self.assertRaisesRegex(ValueError,'sum'):
            E.evaluate(CASES,pp)
    def test_duplicate_case_fail(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            E.evaluate([CASES[0],CASES[0]],E.always_red(CASES))
    def test_unmeasured_expert_votes_skipped_not_negative(self):
        r=E.evaluate(CASES+[dict(case_id='c',veterinarian_votes=[None,None,None])], E.always_red(CASES+[dict(case_id='c')]))
        self.assertEqual(r['n_unique_evaluable_cases'],2)

if __name__=='__main__':unittest.main()
