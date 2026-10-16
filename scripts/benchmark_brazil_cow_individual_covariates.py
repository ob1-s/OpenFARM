"""Cow-level covariate ablation on SAME original clinical outcomes, 16 farms.

Exploratory, not final blinded model evaluation. Source clinical labels are
human-scored gait + BCS; inputs are contemporaneous breed, lactation, days in
milk and milk production. Not a temporal predictor, sensor/vet diagnosis,
or a new independent source. Sources pinned by checksum; whole-farm splits.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from audit_brazil_cow_welfare import ingest
from benchmark_brazil_cow_welfare import sampled_stats

EXPECTED="3b345fa0a0c2495aab40e313b5703c7e7c5d0688393a1759684b801179e6b529"
CAT=["Breed"]
NUM=["Lactation","DIM","MilkProduction"]
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    df,sha,err=ingest("Cowlevel.csv")
    if df is None or sha!=EXPECTED:raise RuntimeError("Pinned 16-farm covariate source inaccessible or changed")
    if len(df)!=5529 or df["farm_ID"].nunique()!=16:raise RuntimeError("16-farm clinical source drift")
    if not set(["Gait score","BCS",*CAT,*NUM]).issubset(df.columns):raise RuntimeError("Source features absent")
    # Original source Clinical lameness and Lameness are *derived* from gait;
    # exclude all such source prevalence and category fields from model state.
    farms=sorted(df["farm_ID"].unique())
    np.random.default_rng(20261016).shuffle(farms)
    group={"train":sorted(map(int,farms[:10])),
           "calibration":sorted(map(int,farms[10:13])),
           "test":sorted(map(int,farms[13:]))}
    prep=ColumnTransformer([
        ("c",Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="(missing)")),
                       ("onehot",OneHotEncoder(handle_unknown="ignore"))]),CAT),
        ("n",Pipeline([("impute",SimpleImputer(strategy="median")),
                       ("scale",StandardScaler())]),NUM)
    ])
    result={
      "source":"https://zenodo.org/records/3408818","source_sha":sha,
      "scope":"Same clinical cohort subset (16 farms, 5529 cows). EXPLORATORY, NOT new source or blind eval.",
      "model_inputs":CAT+NUM,
      "limitation":"Milk production and stage contemporaneous with clinician examination; retrospective association only; source dataset overlaps 50-farm cohort; no gait video.",
      "splits_by_farm":group,
      "tasks":{}
    }
    for name,condition in [
        ("source_clinical_lameness",(df["Gait score"]>=3)),
        ("source_severe_lameness",(df["Gait score"]>=4)),
        ("source_bcs_below_3",(df["BCS"]<3))
    ]:
        valid=df["BCS"].notna() if "bcs" in name else df["Gait score"].notna()
        dat=df.loc[valid].copy()
        dat["target"]=condition.loc[valid].astype(int)
        tr=dat[dat.farm_ID.isin(group["train"])]
        cal=dat[dat.farm_ID.isin(group["calibration"])]
        te=dat[dat.farm_ID.isin(group["test"])]
        m=Pipeline([("prepare",prep),("logistic",LogisticRegression(C=.5,max_iter=400,random_state=20261016))])
        m.fit(tr[CAT+NUM],tr.target)
        baseline=float(tr.target.mean())
        ptest=m.predict_proba(te[CAT+NUM])[:,1]
        by_farm=[]
        for fid in sorted(te.farm_ID.unique()):
            mask=(te.farm_ID.to_numpy()==fid)
            by_farm.append({
                "source_farm_id":int(fid),
                "baseline":sampled_stats(te.target.to_numpy()[mask],np.full(int(mask.sum()),baseline)),
                "model":sampled_stats(te.target.to_numpy()[mask],ptest[mask])
            })
        result["tasks"][name]={
          "train":{"rows":len(tr),"positive":int(tr.target.sum())},
          "calibration":{"rows":len(cal),"positive":int(cal.target.sum())},
          "test":{"rows":len(te),"positive":int(te.target.sum())},
          "baseline":sampled_stats(te.target,np.full(len(te),baseline)),
          "logistic":sampled_stats(te.target,ptest),
          "by_test_farm":by_farm
        }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("BRAZIL_COW_COVARIATE_ABLATION:"+json.dumps(result))
if __name__=="__main__":main()
