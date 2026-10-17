"""OpenFARM Decision Index scoring law regression tests (synthetic only)."""
import copy,unittest
from scripts.index_v01_score import decode,score
WONG=['Immediate','15 mins','30-60 mins','120 mins','240 mins']
def r(group,unit,votes,p=None):
    return {'task_id':'wong_er_priority_v01','unit_id':unit,'group_id':group,
      'kind':'choice','options':WONG,'priority_order':'most_urgent_first',
      'prediction':{'probabilities':p or {'Immediate':.5,'15 mins':.1,
          '30-60 mins':.2,'120 mins':.1,'240 mins':.1}},
      'reference':{'votes':votes}}
def n(group,unit,label,p=.8):
    return {'task_id':'big_broiler_feather_pecking_v01','unit_id':unit,'group_id':group,
      'kind':'noul','prediction':{'probability':p},'reference':{'label':label}}
class IndexScoringTests(unittest.TestCase):
    def test_soft_three_vet_probability_and_urgency(self):
        x=decode(r('patient_A','1',['Immediate','Immediate','120 mins']))
        self.assertAlmostEqual(x['mean_human_vote_variance'],4/9)
        self.assertAlmostEqual(x['brier'],(.5-2/3)**2+(.1-0)**2+(.2-0)**2+(.1-1/3)**2+.1**2)
        self.assertEqual(x['exact'],1)
        self.assertAlmostEqual(x['vote_overtriage'],1/3)
        self.assertEqual(x['vote_undertriage'],0)
    def test_prediction_status_masks_do_not_become_negative(self):
        x=n('clip','2',0);x['status']='unscorable'
        result=score([n('clip1','1',1),x,n('clip2','3',0)],bootstrap=30)['task_metrics']['big_broiler_feather_pecking_v01']
        self.assertEqual(result['scored'],2);self.assertEqual(result['unscorable'],1)
        self.assertEqual(result['n_independence_groups_claimed'],2)
    def test_duplicate_unit_rejected(self):
        x=r('a','1',['Immediate']*3)
        with self.assertRaises(ValueError):score([x,x])
    def test_reference_and_option_mismatch_rejected(self):
        x=r('a','1',['not_in_criteria']*3)
        with self.assertRaises(ValueError):score([x])
    def test_raw_model_states_must_not_leave_verifier(self):
        x=r('a','1',['Immediate']*3);x['state']='sensitive sample'
        with self.assertRaises(ValueError):score([x])
    def test_no_aggregate_of_different_tasks(self):
        x=score([r('hospital_A','1',['Immediate']*3),
          r('hospital_B','2',['240 mins']*3),
          n('clip1','a',1)],bootstrap=20)
        self.assertIsNone(x['cross_task_aggregate'])
        self.assertEqual(len(x['task_metrics']),2)
        self.assertEqual(x['task_metrics']['wong_er_priority_v01']['n_majority_immediate'],1)
if __name__=='__main__':unittest.main()
