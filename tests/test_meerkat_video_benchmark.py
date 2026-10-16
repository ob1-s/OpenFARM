"""Offline unit tests: whole-video independent groups and forbidden-label feature flow."""
import io,sys,unittest,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_meerkat_video_heldout import groups,extract_one,FEATS,STEP,ALL_LABELS

class TestMeerkatWholeVideoIntegrity(unittest.TestCase):
    def test_camera_balanced_whole_video_groups(self):
        names=[f"22-11-07_C{cam}_{i:02d}.csv"
               for cam in (2,3) for i in range(10)]
        a=groups(names);b=groups(list(reversed(names)))
        self.assertEqual(a,b)
        self.assertEqual([len(a[k]) for k in ("train","calibration","test")],[12,4,4])
        for group in ("train","calibration","test"):
            self.assertEqual({name.split("_")[1] for name in a[group]},{"C2","C3"})
        self.assertEqual(len(set(a["train"]+a["calibration"]+a["test"])),20)

    def fixture(self,label_at_50=5):
        rows=[
          "0,0,0,1,0,100,130,130,160",
          f"50,0,0,{label_at_50},0,105,135,130,160",
          "100,0,0,6,0,110,145,130,170",
          "50,0,1,9,1,100,130,130,160",
          "50,1,4,14,0,200,230,130,160",
          "50,0,5,16,0,200,230,130,160",
        ]
        target=io.BytesIO()
        with zipfile.ZipFile(target,"w") as z:
            z.writestr("fake_C2_01.csv","\n".join(rows)+"\n")
        return zipfile.ZipFile(io.BytesIO(target.getvalue()))

    def test_no_label_leakage_and_quality_filter(self):
        name="fake_C2_01.csv"
        X,y,q=extract_one(self.fixture(),name)
        Y,y2,q2=extract_one(self.fixture(label_at_50=3),name)
        self.assertEqual(X.shape,(3,len(FEATS)))
        self.assertEqual(y.tolist(),[1,5,6])
        self.assertEqual(y2.tolist(),[1,3,6])
        self.assertTrue(np.array_equal(X,Y))
        self.assertEqual(q["pup_rows"],1)
        self.assertEqual(q["occluded_meerkat_rows"],1)
        self.assertEqual(q["unclassified_none_rows"],1)
        self.assertEqual(q["sampled_tracks"],1)
        # Frame 0 uses no future displacement; frame 50 only prior frame.
        self.assertEqual(X[0,FEATS.index("norm_speed")],0)
        self.assertGreater(X[1,FEATS.index("norm_speed")],0)

    def test_bad_duplicate_annotation_fails(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z:
            z.writestr("duplicate.csv",
                "0,0,0,1,0,100,130,130,160\n"
                "0,0,0,3,0,100,130,130,160\n")
        with self.assertRaisesRegex(RuntimeError,"Duplicate"):
            extract_one(zipfile.ZipFile(io.BytesIO(buf.getvalue())),"duplicate.csv")

if __name__=="__main__":unittest.main()
