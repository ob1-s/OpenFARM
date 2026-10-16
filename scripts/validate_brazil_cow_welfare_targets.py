"""Validate two independent human-observed welfare reference families.

Source: 13,716 cows / 50 farms / source-original gait+BCS and Herd CSV,
Zenodo 3408818. Never infer animal pain, deterioration or caretaker action
from these cross-sectional 2016 ratings; no temporal claim.

Produces aggregate checks and a pinned-source data contract. No raw rows.
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,re,time
from pathlib import Path
import pandas as pd
import requests
from audit_brazil_cow_welfare import ingest
S=requests.Session()
S.headers["User-Agent"]="OpenFARM-source-rights-verifier/0.1"

def counts(s):
    return {str(k):int(v) for k,v in s.value_counts(dropna=False).items()}

def validate():
    r=S.get("https://zenodo.org/api/records/3408818",timeout=(18,38))
    r.raise_for_status()
    md=r.json().get("metadata",{})
    orig,sha,errors=ingest("BCS and gait score all cows.csv")
    herd,hsha,herr=ingest("Herd.csv")
    if orig is None or herd is None:raise RuntimeError("Required actual original clinical/source files unavailable")
    if len(orig)!=13716 or orig["farm_ID"].nunique()!=50 or len(herd)!=50:
        raise RuntimeError("Unexpected source clinical sample or site count")
    if not herd["farm_ID"].is_unique:raise RuntimeError("Farm metadata not one-to-one")
    labels=["Gait score","BCS","clin_lam","sev_lam","BCS_dic"]
    if any(c not in orig for c in labels):raise RuntimeError("Critical reference column missing")
    gait=pd.to_numeric(orig["Gait score"],errors="coerce")
    bcs=pd.to_numeric(orig["BCS"],errors="coerce")
    threshold_checks={}
    for tgt in ("clin_lam","sev_lam"):
        y=orig[tgt].astype(str).str.strip().str.lower()
        threshold_checks[tgt]={
            "source_value_counts":counts(y),
            "gait_condition_breakdown":{str(score):counts(y[gait==score])
                     for score in sorted(gait.unique())},
        }
    bcs_cat=orig["BCS_dic"].astype(str)
    report={
        "record":"https://zenodo.org/records/3408818",
        "rights_from_zenodo_api":md.get("license"),
        "access_from_api":md.get("access_right"),
        "clinical_sha256":sha,"herd_sha256":hsha,
        "cow_rows":len(orig),
        "farm_ids":int(orig.farm_ID.nunique()),
        "cow_identifier_nulls":int(orig["Cow"].isna().sum()),
        "gait_score_counts":counts(gait),
        "bcs_score_counts":counts(bcs),
        "clinical_derivative_checks":threshold_checks,
        "bcs_category_vs_threshold":{
            "measured_BCS_lt_3":int((bcs<3).sum()),
            "source_low_BCS":int(bcs_cat.str.contains("Low",case=False,na=False).sum()),
            "source_label_mismatch":int(((bcs<3) != bcs_cat.str.contains("Low",case=False,na=False))[bcs.notna()].sum())
        },
        "farm_group_sizes":{
            "min":int(orig.groupby("farm_ID").size().min()),
            "median":float(orig.groupby("farm_ID").size().median()),
            "max":int(orig.groupby("farm_ID").size().max())
        },
        "herd_management_available_fields":list(herd.columns),
        "herd_source_risk_fields":[x for x in herd.columns
                 if re.search(r"(lame|lameness|LS_\d|Scored cows|fall)",x,re.I)],
        "cross_sectional_only":True,
        "not_yet_approved":"Need ex-ante input whitelist, independent farm heldout, nullness, scoring baseline and explicit licensing."
    }
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    res=validate()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(res,indent=2,ensure_ascii=False,default=str)+"\n")
    print("BRAZIL_COW_LABEL_VALIDATION:"+json.dumps(res,default=str))
if __name__=="__main__":main()
