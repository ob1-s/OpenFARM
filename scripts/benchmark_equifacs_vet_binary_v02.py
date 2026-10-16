"""Cheap non-ranking Level-2 pilot: independently rated vet pain vs expert FACS AUs.

The same FACS annotator's own veterinary scores are removed by source builder.
Model receives expert-coded facial movement occupancy/counts (not raw video).
Train/cal/test partitioned by whole media, 10/3/5 groups. This is too small
to estimate clinical or production calibration; test is frozen and untouched.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,average_precision_score,roc_auc_score,log_loss
from build_equifacs_clinical_v0 import produce

def metric(y,p):
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    pred=p>=.5
    tp=int(((pred==1)&(y==1)).sum())
    fp=int(((pred==1)&(y==0)).sum())
    fn=int(((pred==0)&(y==1)).sum())
    out={"n":len(y),"positive":int(sum(y)),"brier":round(float(brier_score_loss(y,p)),5),
         "logloss":round(float(log_loss(y,p,labels=[0,1])),5),
         "precision_at_0_5":round(tp/max(1,tp+fp),4),
         "recall_at_0_5":round(tp/max(1,tp+fn),4),
         "f1_at_0_5":round(2*tp/max(1,2*tp+fp+fn),4),
         "mean_predicted_probability":round(float(p.mean()),4)}
    if len(np.unique(y))==2:
        out["auroc"]=round(float(roc_auc_score(y,p)),4)
        out["average_precision"]=round(float(average_precision_score(y,p)),4)
    return out

def build():
    provenance,splits=produce()
    names=provenance["feature_vocabulary"]
    columns=[prefix+v for prefix in ("AU:","COUNT:") for v in names]
    X={};y={};media={}
    for k,(examples,private) in splits.items():
        X[k]=np.asarray([[e["features"][name] for name in columns] for e in examples],dtype=float)
        y[k]=np.asarray([e["label"] for e in examples],dtype=int)
        media[k]=[m["media"] for m in private]
    if any(set(media[a])&set(media[b]) for a,b in (("train","calibration"),("train","test"),("calibration","test"))):
        raise RuntimeError("Clinical video/media leaked train/eval")
    if len(y["train"])!=32 or len(y["test"])!=12 or len(set(media["test"]))!=5:
        raise RuntimeError("Pinned study heldout membership changed")
    prior=(int(y["train"].astype(np.int64).sum())+1)/(len(y["train"])+2)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.04,max_iter=400,random_state=17))
    model.fit(X["train"],y["train"])
    out={"study":provenance["source"],"pinned_annotations_sha256":provenance["sha256"],
         "level":"L2 limited independent expert-rated welfare-associated pain interpretation",
         "model_input":"human-preannotated facial AU occupancy and count per vet-scored interval",
         "not_input":["media ID","horse ID","vet rater","ordinal clinical label","video/audio"],
         "source_groups":{k:len(set(v)) for k,v in media.items()},
         "class_counts":{k:{"total":len(v),"positive":int(v.sum())} for k,v in y.items()},
         "models":{},"limits":[
            "Only five held-out media and 12 vet-rated intervals. Experimental pilot, no meaningful independent calibration or stable model ranking.",
            "Cannot claim end-to-end horse visual pain detection: FACS is already human-coded from videos.",
            "The FACS annotator's own pain judgments and unscorable segments are excluded upstream.",
            "Source media identifiers are not proven distinct horses/farms, requiring original provenance.",
            "Clinical pain rating is an independently observed proxy for welfare interpretation, not source ground truth of a human intervention.",
            "No post hoc threshold or model feature selection on 12 held-out intervals."
         ]}
    for split in ("calibration","test"):
        prior_p=np.full(len(y[split]),prior)
        fit_p=model.predict_proba(X[split])[:,1]
        out["models"][split]={"source_train_prior":metric(y[split],prior_p),
                              "predeclared_ridge_logistic":metric(y[split],fit_p)}
    out["admission"]="Nonranking tiny Level-2 interpretation pilot ONLY; no validated clinical screening or product-alert score."
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    r=build();a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(r,indent=2)+"\n")
    print("EQUIFACS_INDEPENDENT_VET_BASELINE:"+json.dumps(r))
