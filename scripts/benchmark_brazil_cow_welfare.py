"""Welfare Evidence Sprint: independent clinical gait + BCS decisions.

Open, attributed source: Bran et al. Zenodo 3408818 CC BY 4.0.
Inputs: *predefined* independently recorded herd-management properties and
source cow's breed grouping; no gait, BCS, target-derived per-farm prevalence,
IDs, or assessment-order columns. Source is cross-sectional (2016).
Output: two independent clinical question families, plus derivative severe-gait
diagnostic. Generalization measured by *entire heldout farms*, not cows.

Run: python scripts/benchmark_brazil_cow_welfare.py --out report.json
Source CSVs downloaded transiently/read-only and HASH validated.
No original row-level dataset committed/re-hosted.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,io
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score,average_precision_score,log_loss,f1_score
from audit_brazil_cow_welfare import ingest

EXPECTED={
"BCS and gait score all cows.csv":"c43d3f9b8c883f6f0040ed66c9752b7441cc5c68b7603e376c1d2ef7778fabba",
"Herd.csv":"89e9c75c970f6f76cce793f343c2a371fb07cbace835f5c782d43f39d3835318",
}
CAT=[
    "Farm type","Bedding type","Alley scrapping type",
    "Heat control system type","Feeding type",
    "Pasture access to milking cows","Stall cleaning",
    "Footbath","Preventive hoof trimming","Hoof trimming freq",
    "Track covering","Track moisture","Holding area flooring",
    "Feed bunk type","cow_breed"
]
NUM=[
    "Total cows","Workers number","Milkings per day",
    "Feeding times per day","Years of milk activity"
]
TASKS={
  "clinical_lameness_gait_ge_3":{"reference":"Gait score","threshold":3,
    "definition":"Original observer assigned gait score >=3, clinical lameness flag"},
  "severe_lameness_gait_ge_4":{"reference":"Gait score","threshold":4,
    "definition":"Original observer assigned gait score >=4, severe lameness flag",
    "derived_from":"same gait score as clinical_lameness; not a new reference family"},
  "body_condition_below_3":{"reference":"BCS","threshold":3,
    "definition":"Independent original observer BCS below 3, not a diagnosis of malnutrition"}
}
def source():
    out={}
    for name,digest in EXPECTED.items():
        d,sha,err=ingest(name)
        if d is None or sha!=digest:
            raise RuntimeError(f"Cannot verify pinned source data: {name}: {sha or err}")
        out[name]=d
    return out[ "BCS and gait score all cows.csv"],out["Herd.csv"]

def split_farms(herd):
    if len(herd)!=50 or herd["farm_ID"].nunique()!=50:
        raise RuntimeError("Source herd grouping drift")
    # Only original farm type guides the split, not any measured outcome.
    rng=np.random.default_rng(20261016)
    out={"train":[],"calibration":[],"test":[]}
    for t,quotas in (("Freestall",(23,7,8)),("Compost",(7,3,2))):
        farms=sorted(map(int,herd.loc[herd["Farm type"]==t,"farm_ID"]))
        if len(farms)!=sum(quotas):raise RuntimeError(f"Farm stratum changed: {t}: {len(farms)}")
        rng.shuffle(farms)
        a,b=quotas[0],sum(quotas[:2])
        out["train"].extend(farms[:a])
        out["calibration"].extend(farms[a:b])
        out["test"].extend(farms[b:])
    for ids in out.values():ids.sort()
    if len(set(sum(out.values(),[])))!=50:raise AssertionError("Farm overlap")
    return out

def make_states(cows,herd):
    colset=set(CAT+NUM)-{"cow_breed"}
    if not colset.issubset(herd.columns):raise RuntimeError(f"Missing predefined management features: {colset-set(herd.columns)}")
    frame=cows[["farm_ID","Gait score","BCS","Breed_c","clin_lam","sev_lam","BCS_dic"]].copy()
    frame=frame.merge(herd[["farm_ID"]+list(colset)],on="farm_ID",validate="many_to_one",how="left",indicator=True)
    if (frame["_merge"]!="both").any():raise RuntimeError("Unmatched farm management metadata")
    frame=frame.drop(columns=["_merge"]).rename(columns={"Breed_c":"cow_breed"})
    if len(frame)!=13716:raise RuntimeError("Cow group/target cardinality changed")
    if int(((frame["Gait score"]>=3).astype(int)!=frame["clin_lam"]).sum())>0:raise RuntimeError("Inconsistent source clinical derived field")
    if int(((frame["Gait score"]>=4).astype(int)!=frame["sev_lam"]).sum())>0:raise RuntimeError("Inconsistent source severe field")
    if int((((frame["BCS"]<3)!=frame["BCS_dic"].astype(str).str.contains("Low",case=False))[frame["BCS"].notna()]).sum())>0:
        raise RuntimeError("Unexpected source BCS category mismatch")
    return frame

def predictor():
    c=Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="(missing)")),
                ("one_hot",OneHotEncoder(handle_unknown="ignore"))])
    n=Pipeline([("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler())])
    return ColumnTransformer([("cat",c,CAT),("num",n,NUM)])

def sampled_stats(y,p,thr=.5):
    truth=np.asarray(y,dtype=np.int8)
    prob=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    pred=prob>=thr
    result={"rows":len(y),"positives":int(truth.sum()),"positive_rate":round(float(truth.mean()),5),
         "brier":round(float(brier_score_loss(truth,prob)),6),
         "log_loss":round(float(log_loss(truth,prob,labels=[0,1])),6),
         "precision":round(float(truth[pred].mean()),5) if pred.sum() else None,
         "recall":round(float(pred[truth==1].mean()),5) if truth.sum() else None,
         "f1":round(float(f1_score(truth,pred,zero_division=0)),5)}
    if truth.min()!=truth.max():
        result["auroc"]=round(float(roc_auc_score(truth,prob)),5)
        result["average_precision"]=round(float(average_precision_score(truth,prob)),5)
    return result

def build_and_benchmark():
    cows,herd=source()
    split=split_farms(herd)
    frame=make_states(cows,herd)
    report={
      "reference_source":"https://zenodo.org/records/3408818",
      "doi":"10.5281/zenodo.3408818",
      "rights":"CC BY 4.0 per Zenodo original record",
      "file_sha256":EXPECTED,
      "source_subjects":len(frame),"independent_farms":50,
      "test_grouping":"complete farm_ID groups, 30 train / 10 calibration / 10 test farms; type-stratified; seed 20261016; no outcome stratification",
      "farm_ids":split,
      "model_inputs":CAT+NUM,
      "excluded_from_input":["farm_ID","Cow","OrderAnimal","Gait score","BCS","clin_lam","sev_lam","BCS_dic",
              "LS_1","LS_2","LS_3","LS_4","LS_5","Clinical lameness",
              "Severe lameness","Clinical lameness pen","Severe lameness prevalence",
              "Do cows fall on your farm","Cows falls category",
              "MilkingOrder","Group","Group_cat","Scored cows"],
      "task_reference_families":{"gait":"human-assessed ordinal gait score 1-5",
                                  "body_condition":"separate human-scored body condition 1.5-5"},
      "limitations":["Only cross-sectional risk association, NOT clinical symptom recognition or future prognosis.",
          "At a site, most model inputs are identical for all cows of the same breed; predictive ceiling is limited.",
          "Farm management surveyed contemporaneously, not known-before-disease longitudinal state.",
          "Source BCS<3 is a dataset threshold; not a diagnosis of hunger, deprivation or pain.",
          "Farms (50), not cows (13716), are the independent domain-shift units.",
          "10 calibration and 10 test farms yield wide uncertainty. Models see no animal gait/video/sensor signal.",
          "Severe and clinical lameness derive from SAME gait source: correlated questions not independent supervision."],
      "tasks":{}
    }
    for name,config in TASKS.items():
        source_col=config["reference"]
        valid=frame[source_col].notna()
        filtered=frame.loc[valid].copy()
        if "BCS" in source_col:yes=(filtered[source_col]<config["threshold"]).astype(int)
        else:yes=(filtered[source_col]>=config["threshold"]).astype(int)
        filtered["target"]=yes.to_numpy()
        tr=filtered[filtered.farm_ID.isin(split["train"])]
        ca=filtered[filtered.farm_ID.isin(split["calibration"])]
        te=filtered[filtered.farm_ID.isin(split["test"])]
        xtr=tr[CAT+NUM];ytr=tr.target
        xca=ca[CAT+NUM];yca=ca.target
        xte=te[CAT+NUM];yte=te.target
        prior=float(ytr.mean())
        # Predeclared constant and cheap model; model fits ONLY train farms.
        method=Pipeline([("encoding",predictor()),
                         ("head",LogisticRegression(C=.5,max_iter=400,random_state=20261016))])
        method.fit(xtr,ytr)
        prca=method.predict_proba(xca)[:,1];prte=method.predict_proba(xte)[:,1]
        report["tasks"][name]={
            "question":config["definition"],"independent_reference_family":source_col,
            "derived_from":config.get("derived_from"),
            "train":{"rows":len(tr),"positives":int(ytr.sum()),"independent_farms":int(tr.farm_ID.nunique())},
            "calibration":{"rows":len(ca),"positives":int(yca.sum()),"independent_farms":int(ca.farm_ID.nunique())},
            "test":{"rows":len(te),"positives":int(yte.sum()),"independent_farms":int(te.farm_ID.nunique())},
            "train_constant_prior":prior,
            "test_prior":sampled_stats(yte,np.full(len(yte),prior)),
            "test_logistic":sampled_stats(yte,prte),
            "calibration_logistic":sampled_stats(yca,prca),
            "per_test_farm":[{"id":int(fid),"cows":int(len(g)),
                   "observed_rate":round(float(g.target.mean()),5),
                   "predicted_rate":round(float(prte[te.farm_ID.to_numpy()==fid].mean()),5)}
                   for fid,g in te.groupby("farm_ID")]
        }
        # Report no fake per-animal independency and never use test group to tune.
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    r=build_and_benchmark()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2)+"\n")
    print("BRAZIL_COW_WELFARE_BASELINES:"+json.dumps({
        "farm_ids":r["farm_ids"],"tasks":{k:{
            "train":v["train"],"calibration":v["calibration"],"test":v["test"],
            "baseline":v["test_prior"],"logistic":v["test_logistic"],
            "calibration_logistic":v["calibration_logistic"]
        } for k,v in r["tasks"].items()}}))
if __name__=="__main__":main()
