"""Source-verified inexpensive TWO-question HotPig System-1 historical baseline.

State: 12 source YOLO-precomputed behavior/posture counts at minute t.
Distinct reference labels from the SAME electronic feeder:
 - did feeder register intake at t?
 - did a NEW feeder-active episode start within (t, t+15]?
No post-t input, no welfare distress inference. 24 whole-pig grouped
split; training pigs 1-16, calibration 17-20, test 21-24.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse,hashlib,io,json,zipfile
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score,average_precision_score
from evaluate_hotpig_feeder_candidate import source_bytes,FEATURES

QUESTIONS=("feeder_now","new_feeder_bout_next_15min")
SPLITS={"train":range(1,17),"calibration":range(17,21),"test":range(21,25)}
STRIDE=15;HORIZON=15

def target_windows(feed):
    """Causal target needs full instrument coverage of minute t..t+15."""
    valid=np.isfinite(feed)
    onset=np.zeros(len(feed),dtype=bool)
    onset[1:]=valid[1:] & valid[:-1] & (feed[1:]>0)&(feed[:-1]<=0)
    for t in range(0,len(feed),STRIDE):
        current=int(feed[t]>0) if valid[t] else None
        good=(t+HORIZON<len(feed) and valid[t:t+HORIZON+1].all())
        future=int(onset[t+1:t+HORIZON+1].any()) if good else None
        yield t,current,future

def read_pig(archive,pig):
    raw=pd.read_csv(archive.open(f"series/P{pig}.csv"))
    if len(raw)!=23040:raise RuntimeError("Unexpected HotPig source duration")
    nums=raw[FEATURES+["feed"]].apply(pd.to_numeric,errors="coerce")
    f=nums[FEATURES].to_numpy(dtype=float)
    feed=nums["feed"].to_numpy(dtype=float)
    rows=[]
    for t,now,future in target_windows(feed):
        if not np.isfinite(f[t]).all():continue
        rows.append((t,f[t],now,future))
    return rows

def table(rows,task):
    index=0 if task=="feeder_now" else 1
    filtered=[r for r in rows if r[index+2] is not None]
    X=np.stack([r[1] for r in filtered]).astype(np.float32)
    y=np.array([r[index+2] for r in filtered],dtype=np.int8)
    return X,y,filtered

def metric(y,p,t=.5):
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    pred=p>=t
    tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum())
    fn=int(((pred==0)&(y==1)).sum())
    o={"n":len(y),"positive":int(y.sum()),"base_rate":round(float(y.mean()),6),
       "brier":round(float(brier_score_loss(y,p)),7),
       "logloss":round(float(log_loss(y,p,labels=[0,1])),7),
       "precision":round(tp/max(1,tp+fp),5),
       "recall":round(tp/max(1,tp+fn),5),
       "f1":round(2*tp/max(1,2*tp+fp+fn),5),
       "false_positive_windows":fp,
       "false_negative_windows":fn}
    if len(np.unique(y))==2:
        o["auroc"]=round(float(roc_auc_score(y,p)),6)
        o["average_precision"]=round(float(average_precision_score(y,p)),6)
    else:o["auroc"]=None;o["average_precision"]=None
    return o

def run():
    b=source_bytes()
    sha=hashlib.sha256(b).hexdigest()
    z=zipfile.ZipFile(io.BytesIO(b))
    rows={name:{pig:read_pig(z,pig) for pig in group} for name,group in SPLITS.items()}
    report={"source":"https://zenodo.org/records/17090997","sha256":sha,
            "source_pig_groups":{name:list(group) for name,group in SPLITS.items()},
            "stride":STRIDE,"horizon_minutes":HORIZON,
            "features":list(FEATURES),"reference_families":1,
            "tasks":{},"temporal_checks":{
                "label_at_t_uses_feed_t_only":True,
                "future_window_uses_feed_t_to_t_plus_15_inclusive_for_observation_coverage":True,
                "start_event_requires_prior_observed_nonpositive_minute":True,
                "future_inputs_not_exposed":True,
                "group_holdout":"whole source pig"},
            "scientific_status":"instrumental feeding activity, not physiological welfare or escalation",
            "warnings":[
                "No new independent annotation family: two time-indexed questions share ONE feeder.",
                "Same room/farm: pig heldout, not farm heldout. Multiple windows per pig are correlated.",
                "Stride 15 and horizon 15 creates adjacent nonoverlapping future horizons within a pig, except censoring; not independent pigs.",
                "No calendar-clock validated time sync across electronic feeder/camera; labels respect source relative indexes.",
                "Not caretaker alerts; 15-min prediction is an instrumental onset task, not distress."]
           }
    for task in QUESTIONS:
        per={name:table([r for group in part.values() for r in group],task)
             for name,part in rows.items()}
        Xtrain,ytrain,_=per["train"]
        Xtest,ytest,trows=per["test"]
        Xcal,ycal,_=per["calibration"]
        if len(np.unique(ytrain))!=2:raise RuntimeError(f"Class missing train: {task}")
        prior=(sum(ytrain)+1)/(len(ytrain)+2)
        models={
            "fixed_train_prior":np.full(len(ytest),prior),
            "L2_logistic":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                  LogisticRegression(C=.25,max_iter=350,random_state=17)),
            "tiny_boosted_trees":HistGradientBoostingClassifier(
                  max_leaf_nodes=9,max_depth=3,max_iter=75,min_samples_leaf=150,
                  learning_rate=.07,l2_regularization=1.,early_stopping=False,random_state=17)
        }
        scores={"counts_by_split":{name:{"n":len(y),"positive":int(y.sum())}
                  for name,(_,y,_) in per.items()},"models":{}}
        for name,model in models.items():
            if name=="fixed_train_prior":
                p=np.full(len(ytest),prior);pc=np.full(len(ycal),prior)
            else:
                model.fit(Xtrain,ytrain)
                p=model.predict_proba(Xtest)[:,1]
                pc=model.predict_proba(Xcal)[:,1]
            scores["models"][name]={
                "calibration_threshold_0_5":metric(ycal,pc),
                "test_threshold_0_5":metric(ytest,p),
                "per_test_pig":{}}
            # Per pig group analysis; no claim that every 15-min stride is IID.
            for pig,individual in rows["test"].items():
                Xp,yp,_=table(individual,task)
                pp=(np.full(len(yp),prior) if name=="fixed_train_prior"
                    else model.predict_proba(Xp)[:,1])
                scores["models"][name]["per_test_pig"][str(pig)]=metric(yp,pp)
        if task=="feeder_now":
            # Binary author-detector rule is a meaningful extra reference baseline
            # for current, unlike an arbitrary forecasting threshold.
            eating_col=FEATURES.index("eating")
            scores["models"]["fixed_camera_eating_rule"]={
              "test_threshold_0_5":metric(ytest,(Xtest[:,eating_col]>0).astype(float)),
              "calibration_threshold_0_5":metric(ycal,(Xcal[:,eating_col]>0).astype(float))}
        report["tasks"][task]=scores
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",type=Path,required=True)
    a=parser.parse_args()
    result=run()
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n")
    print("HOTPIG_MULTIQUESTION_HISTORICAL:"+json.dumps({
        name:{"counts":stats["counts_by_split"],
              "test":{model:measure["test_threshold_0_5"]
                     for model,measure in stats["models"].items()}}
        for name,stats in result["tasks"].items()
    }))
