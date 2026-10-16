"""Rigorous cheap multi-question PigWeb benchmark: PRE biomarker state -> POST assays.

Four leave-one-institute-out (LOIO) tests. At each fold another entire site
is reserved as calibration; other two sites train fixed L2 logistic heads.
Never treat behavior DURING the stimulus as a PRE-state feature.
Cortisol/oxytocin are lab measurements, not independent labels of happiness,
pain, or whether caretakers should intervene.

Never publish source rows; the CI artifact contains only aggregated metrics.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse,json,math
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score,average_precision_score
from build_pigweb_historical_v02 import build,SITE_ORDER

FEATURES=("pre_cortisol_ng_ml","pre_oxytocin_pg_ml")
ASSAYS=("cortisol","oxytocin")
def metrics(y,p):
    y=np.array(y,dtype=np.int64)
    p=np.clip(np.array(p,dtype=float),1e-6,1-1e-6)
    result={"n":int(len(y)),"positive":int(y.sum()),"positive_rate":round(float(y.mean()),5),
            "brier":round(float(brier_score_loss(y,p)),6),
            "log_loss":round(float(log_loss(y,p,labels=[0,1])),6),
            "predicted_mean":round(float(p.mean()),5),
            "accuracy_0_5":round(float(((p>=.5)==y).mean()),5)}
    if len(set(y))==2:
        result["auroc"]=round(float(roc_auc_score(y,p)),5)
        result["average_precision"]=round(float(average_precision_score(y,p)),5)
    else:
        result["auroc"]=None;result["average_precision"]=None
    return result

def run():
    source,rows=build()
    if len(rows)!=248:raise RuntimeError("Pinned PigWeb source session count changed")
    grouped={site:[r for r in rows if r["provenance"]["institute"]==site] for site in SITE_ORDER}
    if sum(len(v) for v in grouped.values())!=248:raise RuntimeError("Missing site group")
    for r in rows:
        if r["provenance"]["post"]<=r["provenance"]["pre"]:
            raise RuntimeError("Future outcomes not strictly after pre state")
        if set(FEATURES)-set(r["features"]):raise RuntimeError("PRE feature absent")
        for k in FEATURES:
            if not k.startswith("pre_"):raise RuntimeError("Hindsight behavior used as PRE input")
    out={"source":source["source"],"source_sha256":source["source_sha256"],
         "site_count":4,"pig_count":124,"paired_sessions":248,
         "model_features":list(FEATURES),
         "explicit_exclusions":["human behavior during stimulus","site ID","animal ID",
                                "experiment context/condition","realized post value","post sample delay"],
         "models":["train-site prevalence prior","pre-only L2 logistic, C=0.25"],
         "protocol":"4 leave-one-INSTITUTE-out tests, disjoint site calibration, train on remaining two; no test selection",
         "question_labels":[f"pigweb.{a}.increased" for a in ASSAYS],
         "folds":{},"warnings":[
             "Input PRE hormone values and target POST increases are independently assayed longitudinal quantities.",
             "Each pig contributes two experimental contexts; independent N=124 pigs, NOT 248 pigs.",
             "Predictions have no controlled stimulus information, and hormone increase is not an affect/alert label.",
             "The behavior data is recorded DURING experimental exposure and has NO verified completion timestamp before the PRE sample. Hindsight behavior must not be modeled as a future predictor.",
             "No source-invariant causal biomarker effect, zero-shot vision ability or cross-farm deployment is established."]}
    oof={a:{"y":[],"prior":[],"logit":[]} for a in ASSAYS}
    for i,site in enumerate(SITE_ORDER):
        calsite=SITE_ORDER[(i+1)%4]
        train=[r for s in SITE_ORDER if s not in (site,calsite) for r in grouped[s]]
        cal=grouped[calsite]
        test=grouped[site]
        for g in (train,cal,test):
            if not g:raise RuntimeError("Empty site partition")
        if (set((r["provenance"]["institute"],r["provenance"]["pig_id"]) for r in train)&
            set((r["provenance"]["institute"],r["provenance"]["pig_id"]) for r in test)):
            raise RuntimeError("Pig ID group leakage")
        out["folds"][site]={"calibration_site":calsite,
                            "n_train_sessions":len(train),"n_cal_sessions":len(cal),
                            "n_test_sessions":len(test),"assays":{}}
        for assay in ASSAYS:
            key=assay+"_increase"
            ready=lambda examples:[r for r in examples if r["targets"][key] is not None]
            a,b,c=ready(train),ready(cal),ready(test)
            if min(map(len,(a,b,c)))==0:raise RuntimeError(f"No observed labels {assay}/{site}")
            y=np.array([r["targets"][key] for r in a],dtype=int)
            ytest=np.array([r["targets"][key] for r in c],dtype=int)
            prior=(y.sum()+1)/(len(y)+2) # train-only Laplace smoothing
            predict_prior=np.full(len(c),prior,dtype=float)
            X=np.array([[r["features"][k] if r["features"][k] is not None else np.nan
                         for k in FEATURES] for r in a],dtype=float)
            Xtest=np.array([[r["features"][k] if r["features"][k] is not None else np.nan
                             for k in FEATURES] for r in c],dtype=float)
            if len(np.unique(y))!=2:
                p=np.full(len(c),prior,dtype=float)
                behavior="train_only_prior_due_to_single_class"
            else:
                model=make_pipeline(SimpleImputer(strategy="median",add_indicator=True,
                                                   keep_empty_features=True),
                                    StandardScaler(),
                                    LogisticRegression(C=.25,max_iter=500,random_state=42))
                model.fit(X,y)
                p=model.predict_proba(Xtest)[:,1]
                behavior="pre_only_L2"
            baseline=metrics(ytest,predict_prior)
            learned=metrics(ytest,p)
            oof[assay]["y"].extend(ytest.tolist());oof[assay]["prior"].extend(predict_prior.tolist());oof[assay]["logit"].extend(p.tolist())
            out["folds"][site]["assays"][assay]={
                "train_labels":len(a),"calibration_labels_unfit":len(b),
                "test_labels":len(c),"test_pigs":len({r["provenance"]["pig_id"] for r in c}),
                "fit_mode":behavior,"prior":baseline,"logistic":learned}
    out["pooled_oof_non_independent_training_folds"]={}
    for assay,d in oof.items():
        out["pooled_oof_non_independent_training_folds"][assay]={
            "prior":metrics(d["y"],d["prior"]),"logistic":metrics(d["y"],d["logit"])}
    out["decision"]="Evaluates site transfer of PRE-only biomarker forecasting, NOT valenced welfare interpretation or human-intervention recommendations."
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()
    result=run()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+"\n")
    print("PIGWEB_PRE_ONLY_BASELINES:"+json.dumps({
      "pooled":result["pooled_oof_non_independent_training_folds"],
      "by_site":{site:{assay:{"prior_brier":v["prior"]["brier"],
                              "logistic_brier":v["logistic"]["brier"],
                              "n":v["test_labels"],"prevalence":v["logistic"]["positive_rate"]}
                     for assay,v in fold["assays"].items()}
                 for site,fold in result["folds"].items()}
    }))
