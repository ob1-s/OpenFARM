"""No source download: multi-question prediction cannot invent masked truth."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
from benchmark_hotpig_shared_queries_v02 import stack
from evaluate_hotpig_feeder_candidate import FEATURES

class SharedStateTests(unittest.TestCase):
    def test_two_valid_questions_one_state(self):
        features=np.zeros(len(FEATURES))
        X,y,q=stack([(50,features,1,0)])
        self.assertEqual(X.shape,(2,len(FEATURES)+1))
        self.assertEqual(y.tolist(),[1,0])
        self.assertEqual(q.tolist(),[0,1])
        np.testing.assert_array_equal(X[0,:-1],X[1,:-1])
    def test_masked_future_is_not_no_label(self):
        X,y,q=stack([(50,np.ones(len(FEATURES)),0,None)])
        self.assertEqual(X.shape,(1,len(FEATURES)+1))
        self.assertEqual(y.tolist(),[0])
        self.assertEqual(q.tolist(),[0])
    def test_missing_current_keeps_measured_future_only(self):
        X,y,q=stack([(50,np.ones(len(FEATURES)),None,1)])
        self.assertEqual(X.shape,(1,len(FEATURES)+1))
        self.assertEqual(y.tolist(),[1]);self.assertEqual(q.tolist(),[1])
    def test_reject_invalid_binary_or_sensor(self):
        for rows in ([(0,np.ones(len(FEATURES)),2,0)],
                     [(0,np.ones(len(FEATURES)-1),1,0)],
                     [(0,np.full(len(FEATURES),float("nan")),1,0)]):
            with self.assertRaises(ValueError):stack(rows)
    def test_fully_masked_rejected(self):
        with self.assertRaisesRegex(ValueError,"No witnessed"):
            stack([(5,np.ones(len(FEATURES)),None,None)])

if __name__=="__main__":unittest.main()
