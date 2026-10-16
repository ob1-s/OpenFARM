"""Question-conditioned single-head System-1 vs separate heads on two HotPig decisions.

One animal-relative minute state (12 author-computed camera features), with
TWO distinct time-indexed questions based on ONE independent feeder instrument.
No leakage of future instrument readings into model features.

This *really fits one shared model*: a single logistic regression with 12
shared feature coefficients and one question-specific intercept adjustment,
compared to two separate logistic regressions and two train-only priors.

Scientific contract: whole-pig 16 train / 4 calibration / 4 test;
15-minute state stride; missing future reference is MASKED, not zero;
no clinical inference, no site generalization, no independent label inflation.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score,average_precision_score

from benchmark_hotpig_historical_v02 import (
    SPLITS,QUESTIONS,read_pig,table,metric,STRIDE,HORIZON)
from evaluate_hotpig_feeder_candidate import source_bytes,FEATURES

def stack(rows):
    """Rows (time, state, current, future) become masked question instances.

    IDs/cutoffs/labels and censoring remain separate; question indicator is
    the sole non-sensor model feature. Zero labels are never filled for masks.
    """
    X=[];y=[];qids=[]
    for t,features,now,future in rows:
        for i,label in enumerate((now,future)):
            if label is None:continue
            if label not in (0,1):raise ValueError("Nonbinary source evidence")
            v=np.asarray(features,dtype=float)
            if v.shape!=(len(FEATURES),) or not np.isfinite(v).all():
                raise ValueError("Invalid observed state shape/data")
            X.append(np.r_[v,float(i)])
            y.append(int(label))
            qids.append(i)
    if not X:raise ValueError("No witnessed labels")
    return np.stack(X).astype(np.float64),np.array(y,dtype=int),np.array(qids,dtype=int)

def score(y,p):
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    out=metric(y,p)
    out["observed_prevalence"]=round(float(y.mean()),6)
    out["brier"]=round(float(brier_score_loss(y,p)),7)
    out["logloss"]=round(float(log_loss(y,p,labels=[0,1])),7)
    if len(set(y))==2:
        out["auroc"]=round(float(roc_auc_score(y,p)),6)
        out["average_precision"]=round(float(average_precision_score(y,p)),6)
    else:
        out["auroc"]=None;out["average_precision"]=None
    return out

def run():
    raw=source_bytes()
    z=zipfile.ZipFile(io.BytesIO(raw))
    groups={name:{pig:read_pig(z,pig) for pig in pigs}
            for name,pigs in SPLITS.items()}
    # Aggregate across whole independent pigs, never leak split labels.
    allrows={name:[row for v in grouped.values() for row in v]
             for name,grouped in groups.items()}
    Xtrain,ytrain,qi_train=stack(allrows["train"])
    Xcal,ycal,qi_cal=stack(allrows["calibration"])
    Xtest,ytest,qi_test=stack(allrows["test"])
    if {int(x) for x in qi_test}!={0,1}:raise RuntimeError("Test questions incomplete")
    # A single model shares sensor coefficients. Each question has a separate
    # intercept through the question indicator, *not* a per-question model.
    joint=make_pipeline(StandardScaler(),LogisticRegression(
        C=.25,max_iter=400,random_state=17))
    joint.fit(Xtrain,ytrain)
    scored={name:joint.predict_proba(X)[:,1]
            for name,X in (("calibration",Xcal),("test",Xtest))}
    report={
      "source":"https://zenodo.org/records/17090997",
      "source_sha256":hashlib.sha256(raw).hexdigest(),
      "source_groups":{k:list(v) for k,v in SPLITS.items()},
      "state_stride_minutes":STRIDE,"future_horizon_minutes":HORIZON,
      "number_of_distinct_questions":2,"underlying_reference_families":1,
      "model":"ONE shared L2 logistic: 12 shared camera feature slopes, question-specific intercept",
      "question_text":{
        "feeder_now":"Did the electronic feeder register intake in the observed minute?",
        "new_feeder_bout_next_15min":"Did a NEW feeder bout start strictly after the observed minute, within 15 minutes?"},
      "observed_inputs":FEATURES,
      "other_fit_input":"question ID only (1-bit indicator); no input pig ID, feeder reading, treatment, future time, or post-event feature",
      "calibration_status":"No post-hoc calibration fitted; 4 calibration pigs reserved but insufficient for robust fitted calibration.",
      "masking":"All missing future labels remain unscored; source onset requires fully observed t..t+15 feeder reference.",
      "models":{},"warnings":[
       "Never average both questions as two independent welfare outcomes; both use one feeder instrument.",
       "No zero-shot multimodal encoder: precomputed 12 camera features, logistic only.",
       "Only 4 held-out pigs at one farm/trial; minutes and question instances highly correlated.",
       "This compares one shared-head mapping with per-question heads, NOT a neural joint world model."]}
    outputs={("joint", "calibration"):scored["calibration"],
             ("joint","test"):scored["test"]}
    for name in QUESTIONS:
        k=QUESTIONS.index(name)
        a,b=table(allrows["train"],name)[:2]
        train_prior=(int(np.asarray(b,dtype=np.int64).sum())+1)/(len(b)+2)
        fit=make_pipeline(StandardScaler(),LogisticRegression(
            C=.25,max_iter=400,random_state=17))
        fit.fit(a,b)
        for split in ("calibration","test"):
            X,y,_=table(allrows[split],name)
            outputs[(name+"_independent_logistic",split)]=fit.predict_proba(X)[:,1]
            outputs[(name+"_train_prior",split)]=np.full(len(y),train_prior)
    for q_id,question in enumerate(QUESTIONS):
        report["models"][question]={}
        for split,xx,yy,qq in (("calibration",Xcal,ycal,qi_cal),
                                ("test",Xtest,ytest,qi_test)):
            mask=qq==q_id
            direct_y=yy[mask]
            probs={
             "shared_one_head":scored[split][mask],
             "per_question_logistic":outputs[(question+"_independent_logistic",split)],
             "per_question_train_prior":outputs[(question+"_train_prior",split)]
            }
            for name,p in probs.items():
                if len(p)!=len(direct_y):raise RuntimeError("Question/label order mismatch")
                report["models"][question].setdefault(name,{})[split]=score(direct_y,p)
    report["joint_train_rows"]=len(ytrain)
    report["joint_masked_train_slots"]=2*len(allrows["train"])-len(ytrain)
    report["joint_test_rows"]=len(ytest)
    report["joint_masked_test_slots"]=2*len(allrows["test"])-len(ytest)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    report=run()
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n")
    print("HOTPIG_SHARED_TWO_QUESTION:"+json.dumps({
       "groups":report["source_groups"],"train_rows":report["joint_train_rows"],
       "heldout_rows":report["joint_test_rows"],
       "per_question":{q:{model:data["test"] for model,data in models.items()}
                       for q,models in report["models"].items()}
    }))
if __name__=="__main__":main()
