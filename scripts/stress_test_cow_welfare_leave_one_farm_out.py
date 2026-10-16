"""OpenFARM Welfare Evidence Sprint: full leave-one-farm-out robustness.

Research question (retrospective, NOT animal diagnosis/forecast):
 Given source-measured Breed, Lactation, Days In Milk and MilkProduction, are
 original HUMAN gait and body-condition assessments predicted better than the
 training-farm prior *across farms*, rather than on one cherry-picked split?

Source: Bran et al., CC BY 4.0, Zenodo 3408818, Cowlevel.csv.
No source row data republished. 16 leave-one-farm-out fits per outcome, each
using strictly other farms; no same-farm fitting or threshold tuning.
Abstain from making clinically relevant claims when this check fails.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score,average_precision_score,log_loss
from audit_brazil_cow_welfare import ingest

SOURCE_SHA="3b345fa0a0c2495aab40e313b5703c7e7c5d0688393a1759684b801179e6b529"
FEATURES={
 "basic_demographic":["Breed","Lactation","DIM"],
 "with_milk_production":["Breed","Lactation","DIM","MilkProduction"]
}
TASKS={
 "clinical_lameness":{"reference":"Gait score","comparison":">=","threshold":3,
                      "family":"human gait score"},
 "severe_lameness":{"reference":"Gait score","comparison":">=","threshold":4,
                    "family":"human gait score (threshold rewrite)"},
 "low_body_condition":{"reference":"BCS","comparison":"<","threshold":3,
                       "family":"independent human body-condition score"}
}
def build_model(columns):
    cat=[c for c in columns if c=="Breed"]
    num=[c for c in columns if c!="Breed"]
    processor=ColumnTransformer([
      ("cat",Pipeline([("fill",SimpleImputer(strategy="constant",fill_value="(missing)")),
                       ("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat),
      ("num",Pipeline([("fill",SimpleImputer(strategy="median")),
                       ("std",StandardScaler())]),num)
    ])
    return Pipeline([("proc",processor),
                     ("head",LogisticRegression(C=.5,max_iter=500,random_state=20261016))])
def evaluate(y,prob):
    y=np.asarray(y,dtype=np.int8)
    p=np.clip(np.asarray(prob,dtype=float),1e-7,1-1e-7)
    out={"rows":len(y),"positive":int(y.sum()),"prevalence":round(float(y.mean()),6),
         "brier":round(float(brier_score_loss(y,p)),8),
         "log_loss":round(float(log_loss(y,p,labels=[0,1])),8)}
    if 0<y.sum()<len(y):
        out["auroc"]=round(float(roc_auc_score(y,p)),6)
        out["average_precision"]=round(float(average_precision_score(y,p)),6)
    else:
        out["auroc"]=None;out["average_precision"]=None
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    d,sha,errors=ingest("Cowlevel.csv")
    if d is None or sha!=SOURCE_SHA:raise RuntimeError(f"Pinned public source inaccessible or changed: {sha}; {errors}")
    if len(d)!=5529 or d.farm_ID.nunique()!=16:raise RuntimeError("Source cohort changed")
    if any(c not in d for c in set(sum(FEATURES.values(),[]))|{"farm_ID","Gait score","BCS"}):
        raise RuntimeError("Required independent cow covariates/reference labels not present")
    groups=sorted(int(x) for x in d.farm_ID.unique())
    report={
      "source":"https://zenodo.org/records/3408818","source_sha256":sha,
      "rights":"Original CSV CC BY 4.0; row data remain at source",
      "independent_sites":len(groups),"clinical_reference_families":2,
      "design":"LEAVE ONE WHOLE FARM OUT, 16 separate retrospective folds per task/features; study same as prior single-split analysis",
      "inputs":FEATURES,
      "excluded":"source gait/body-condition and all derived category/lameness fields, farm_ID, cow identifier and post-score summaries",
      "limitations":[
        "Retrospective, not time-anchored forecasting or prospective alerts.",
        "Milk production potentially consequence/collider of health state and is not a validated pre-clinical sensor.",
        "16 farms from one study/region, unequal farm sizes, not external replication.",
        "Two independent human label families: clinical and severe labels reuse one gait grade.",
        "No calibrated clinical/welfare intervention decision, no video, no site-sealed untouched final test.",
        "Repeated fold comparisons are exploratory; no confidence interval from treating cows as independent sites."],
      "tasks":{}
    }
    for name,task in TASKS.items():
        valid=pd.to_numeric(d[task["reference"]],errors="coerce").notna()
        df=d.loc[valid].copy()
        score=pd.to_numeric(df[task["reference"]])
        df["target"]=(score>=task["threshold"] if task["comparison"]==">=" else score<task["threshold"]).astype(int)
        candidate={}
        for variant,features in FEATURES.items():
            rows=[]
            for held in groups:
                tr=df.loc[df.farm_ID!=held]
                te=df.loc[df.farm_ID==held]
                if not len(te):continue
                if tr.target.nunique()!=2:raise RuntimeError("Single target class in train farms")
                baseline=float(tr.target.mean())
                head=build_model(features)
                head.fit(tr[features],tr.target)
                proba=head.predict_proba(te[features])[:,1]
                rule=evaluate(te.target.to_numpy(),np.full(len(te),baseline))
                learned=evaluate(te.target.to_numpy(),proba)
                rows.append({
                    "farm_id":held,"n_cows":len(te),"n_positive":int(te.target.sum()),
                    "baseline_prior":round(baseline,6),
                    "baseline":rule,"model":learned,
                    "brier_gain_over_prior":round(rule["brier"]-learned["brier"],8),
                    "log_loss_gain_over_prior":round(rule["log_loss"]-learned["log_loss"],8)
                })
            gains=np.array([x["brier_gain_over_prior"] for x in rows])
            weighted=np.average(gains,weights=[x["n_cows"] for x in rows])
            candidate[variant]={
                "farm_folds":len(rows),
                "farms_beating_train_prior_on_brier":int((gains>0).sum()),
                "farms_worse":int((gains<0).sum()),
                "unweighted_mean_brier_gain":round(float(gains.mean()),7),
                "unweighted_median_brier_gain":round(float(np.median(gains)),7),
                "cow_weighted_brier_gain":round(float(weighted),7),
                "gain_min":round(float(gains.min()),7),
                "gain_max":round(float(gains.max()),7),
                "per_farm":rows
            }
        report["tasks"][name]={
            "original_reference":task["reference"],
            "reference_family":task["family"],
            "valid_source_rows":len(df),
            "source_positive_rows":int(df.target.sum()),
            "feature_ablations":candidate
        }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    summary={k:{f:{"farms_win":v["farms_beating_train_prior_on_brier"],
                     "farm_mean_gain":v["unweighted_mean_brier_gain"],
                     "cow_weighted_gain":v["cow_weighted_brier_gain"],
                     "min":v["gain_min"],"max":v["gain_max"]}
                for f,v in val["feature_ablations"].items()}
             for k,val in report["tasks"].items()}
    print("WELFARE_COW_LOFO:"+json.dumps(summary))
if __name__=="__main__":main()
