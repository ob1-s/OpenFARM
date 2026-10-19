#!/usr/bin/env python3
"""Fail-closed source-authenticated OpenFARM Index alpha runner.

One independent task per invocation, no raw source rows published.
Original scripts remain canonical; this wrapper only validates expected
source provenance, summarizes equal-semantics metrics, and emits a scorecard.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CATALOG=ROOT/"benchmarks"/"index-alpha"/"tasks.json"
TASK_RUN={
 "hotpig.feeder-current.noul":("scripts/benchmark_hotpig_system1.py","--out"),
 "cattle.mi-behavior.choice":("scripts/build_public_cattle_behavior_v0.py","--out"),
 "brazil.cow-locomotion-risk.noul":("scripts/stress_test_cow_welfare_leave_one_farm_out.py","--out"),
}
def scorecard(task_id,r,task):
    if task_id=="hotpig.feeder-current.noul":
        sha=r["source_zip_sha256"];m=r["models"]
        assert set(m)>={"logistic_standardized_L2","tiny_gradient_boosted_tree"}
        assert r["valid_counts"]=={"train":368508,"calibration":90150,"test":92160}
        metrics={name:res["test_at_calibration_threshold"]["pooled"]
                 for name,res in m.items()}
        metrics["camera_eating_rule"]=r["rule"]["test"]["pooled"]
        domain="instrument_only"
    elif task_id=="cattle.mi-behavior.choice":
        sha=r["source"]["file_sha256"]
        assert r["integrity"]["raw_rows"]==9222
        assert len(r["integrity"]["animal_counts"])==24
        assert set(r["results"])=={"breed_balanced_group_holdout","jornada_to_chihuahuan","chihuahuan_to_jornada"}
        metrics={s:{m:{k:v for k,v in v.items() if k not in ("test_by_cow","confusion_gt_rows_pred_cols")}
                    for m,v in e["model_scores"].items()}
                 for s,e in r["results"].items()}
        domain="human_video_reviewed_behavior_label_only"
    else:
        sha=r["source_sha256"]
        assert r["independent_sites"]==16
        assert r["clinical_reference_families"]==2
        t=r["tasks"];assert set(t)=={"clinical_lameness","severe_lameness","low_body_condition"}
        metrics={k:{v:{q:w for q,w in z.items() if q!="per_farm"}
                    for v,z in e["feature_ablations"].items()}
                 for k,e in t.items()}
        domain="clinical_risk_from_context_not_direct_welfare_perception"
    if sha!=task["source_sha256"]:raise RuntimeError("Published source fingerprint mismatch")
    return {"task_id":task_id,"status":"DEVELOPMENTAL_REPRODUCED_NOT_BLIND",
            "original_source_sha256":sha,
            "task_kind":task["task_kind"],"reference":task["reference"],
            "group_split":task["split"],"primary_metric":task["primary_metric"],
            "validity_boundary":domain,"measurements":metrics,
            "warnings":task["cautions"]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=sorted(TASK_RUN),required=True)
    ap.add_argument("--out-dir",type=Path,default=Path("reports/index-alpha"))
    a=ap.parse_args()
    cat=json.loads(CATALOG.read_text());task=next(t for t in cat["tasks"] if t["id"]==a.task)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    raw=a.out_dir/(a.task+".raw.aggregate.json")
    cmd=[sys.executable,str(ROOT/TASK_RUN[a.task][0]),TASK_RUN[a.task][1],str(raw)]
    subprocess.run(cmd,cwd=ROOT,check=True)
    r=json.loads(raw.read_text())
    out=scorecard(a.task,r,task)
    (a.out_dir/(a.task+".scorecard.json")).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("OPENFARM_INDEX_ALPHA_OK",json.dumps({"task":a.task,"source_sha256":out["original_source_sha256"],
      "status":out["status"],"scorecard":str(a.out_dir/(a.task+".scorecard.json"))}))
if __name__=="__main__":main()
