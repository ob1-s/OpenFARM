"""Swedish-only retrospective FIRST-TREATMENT candidate with strict causal states.

This is a limited *noncommercial case-control research candidate*, NOT an
approved population-level alert or an actual welfare-care recommendation.
The full 65-pen source has an unreconciled 28-vs-27 outbreak discrepancy:
use only the Swedish study's 21 TB + 21 K cohorts, which agree with paper.
Three Swedish treatment cohorts have 0 days of pre-treatment data -> exclude
them from positive forecasting, but retain as provenance in the audit.

Data are time-ordered by Swedish source's YYYYMMDD, compared against properly
reconstructed M[ M ]DDYY / M[ M ]DDYYYY treatment anchors. Source controls
have artificial TBSTART_ dates ONLY used to identify distinct source cohorts;
they NEVER become labels, features, or event dates.
"""
from __future__ import annotations
import argparse,collections,datetime,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_pig_tail_damage_outcomes import download,parse
from reconstruct_pig_treatment_cohorts import as_date
from openfarm_query_compiler_v02 import Compiler,Observation,Question

K=("Farm","PenID","PenType","TBSTART_")
NUM=("DFV","DFC","StdFC")
SEED=20261016

def blocks():
    b,_=download()
    d=parse(b)
    d=d.loc[d["Farm"]=="Swedish"].copy()
    d["obs_day"]=[as_date("Swedish",v,"date") for v in d["date"]]
    d["anchor"]=[as_date("Swedish",v,"outcome") for v in d["TBSTART_"]]
    return d,hashlib.sha256(b).hexdigest()

def source_cohorts(d):
    out={}
    for gk,raw in d.groupby(list(K),sort=False):
        site,pen,typ,pseudodate=gk
        assert site=="Swedish"
        key="|".join(map(str,gk))
        onset=as_date(site,pseudodate,"outcome")
        before=raw.loc[raw.obs_day<onset].copy() if typ=="TB" else raw.copy()
        if typ=="TB" and len(before)==0:
            out[key]={"status":"no_pre_event_data","type":"TB","observed_pre_days":0}
            continue
        values={col:pd.to_numeric(before[col],errors="coerce") for col in NUM}
        for col in NUM:before[col]=values[col]
        if before[list(NUM)].isna().any().any():raise RuntimeError("Source feeder feature missing")
        daily=before.groupby("obs_day").agg(
           pig_count=("ID","nunique"),
           DFV=("DFV","median"),
           DFC=("DFC","median"),
           StdFC=("StdFC","median"))
        daily=daily.sort_index()
        out[key]={"type":typ,"onset":onset,"daily":daily,"n_source_pigs":int(before.ID.nunique())}
    if len(out)!=42:raise RuntimeError("Not exactly 42 Swedish original cohorts")
    return out

def assign(cohorts):
    rng=np.random.default_rng(SEED)
    grp={"train":[],"calibration":[],"test":[]}
    for cl in ("TB","K"):
        candidates=[k for k,v in cohorts.items() if v["type"]==cl]
        if len(candidates)!=21:raise RuntimeError("Original outcome groups changed")
        rng.shuffle(candidates)
        for part,items in (("train",candidates[:13]),("calibration",candidates[13:17]),("test",candidates[17:])):
            grp[part].extend(items)
    return grp

def features(daily,day):
    # True observed past 14 calendar days only. Require >=11 known days and
    # >=5 distinct pigs observed on each used day; 7d-lag features use enough
    # corresponding observed days to be defined.
    days=[day-datetime.timedelta(days=i) for i in range(13,-1,-1)]
    rows=daily.reindex(days)
    good=rows.loc[rows.pig_count>=5]
    if len(good)<11 or len(rows.iloc[:7].loc[rows.iloc[:7].pig_count>=5])<5 or len(rows.iloc[7:].loc[rows.iloc[7:].pig_count>=5])<5:
        return None
    feat={
      "prior14_covered_days":len(good),
      "prior14_min_observed_pigs":int(good.pig_count.min()),
      "prior14_mean_observed_pigs":float(good.pig_count.mean())
    }
    for name in NUM:
        p=rows.iloc[:7][name].dropna()
        c=rows.iloc[7:][name].dropna()
        if len(p)<5 or len(c)<5:return None
        feat[f"{name.lower()}_recent7_median"]=float(c.median())
        feat[f"{name.lower()}_previous7_median"]=float(p.median())
        feat[f"{name.lower()}_seven_day_change"]=float(c.median()-p.median())
    return feat

def build(local_dir=None):
    raw,source_hash=blocks()
    cohorts=source_cohorts(raw)
    groups=assign(cohorts)
    splitby={key:split for split,items in groups.items() for key in items}
    compiler=Compiler(local_dir)
    assessment={"known_cohorts":len(cohorts),"source_label_counts":dict(collections.Counter(x["type"] for x in cohorts.values())),
       "source_hash":source_hash,"actual_source_treatment_events":21,
       "no_pre_event_cohorts":[hashlib.sha256(k.encode()).hexdigest()[:16] for k,v in cohorts.items() if v.get("status")=="no_pre_event_data"],
       "controls_pseudo_anchor_exposed_to_model":False,
       "windows_omitted_for_missing_history":0,
       "states_with_true_next7_treatment":0,
       "cohorts_with_positive_examples":set(),
       "windows_sampled":{"train":0,"calibration":0,"test":0},
       "notes":["Sweden case-control cohort deliberately samples TB and K pens; NOT natural treatment prevalence",
                "Source contains no independent affirmative caretaker review labels before observed injury treatment",
                "Selection on case status and use of same animals in repeated 14-day windows limit independence",
                "Source data are CC BY-NC 4.0 noncommercial research only",
                "3 actual TB cohorts contain only post-onset records, cannot generate true precursor positives"]}
    for key,entry in cohorts.items():
        if entry.get("status")=="no_pre_event_data":continue
        typ=entry["type"]
        daily=entry["daily"]
        first,last=min(daily.index),max(daily.index)
        # Use 7-day stride from the first day with full source history.
        t=first+datetime.timedelta(days=13)
        while t+datetime.timedelta(days=7)<=last+(datetime.timedelta(days=7) if typ=="TB" else datetime.timedelta(days=0)):
            feat=features(daily,t)
            if feat is None:
                assessment["windows_omitted_for_missing_history"]+=1
                t+=datetime.timedelta(days=7);continue
            if typ=="TB":
                lead=(entry["onset"]-t).days
                if lead<=0:break
                target=int(1<=lead<=7)
                if target:assessment["cohorts_with_positive_examples"].add(hashlib.sha256(key.encode()).hexdigest()[:16])
                # For positive onset inside future horizon the event itself
                # supplies the reference; prior history already verified.
                # For negative, real observed first-treatment date after horizon
                # supplies negative event reference.
            else:
                if t+datetime.timedelta(days=7)>last:break
                target=0  # Source K has no treatment during selected period.
            split=splitby[key]
            compiler.add(Observation(
              source="swedish_tail_treatment_case_control",
              state_id=f"swedish:{hashlib.sha256(key.encode()).hexdigest()[:16]}:{t.isoformat()}",
              split=split,
              group_scope="original_pen_cohort",group_id=hashlib.sha256(key.encode()).hexdigest()[:20],
              modalities=("electronic_feeder_past14days",),
              features=feat,
              cutoff=t.isoformat(),
              questions=[Question(
                question_id="swedish.pen.first_treatment_within_next7d",
                text="Does this pen have its FIRST recorded tail-damage treatment within the following seven days?",
                primitive="Noul",reference_family="original_pen_first_tail_damage_treatment",
                evidence_origin="original farm caretaker documented first treatment or source-matched control status",
                value=target,valid=True,
                relation="future",reference_time=(t+datetime.timedelta(days=7)).isoformat(),
                readiness="NONCOMMERCIAL_CASE_CONTROL_EXPLORATORY__NOT_DEPLOYABLE")],
              audit={"source_original_sha256":source_hash,"source_license":"CC BY-NC 4.0",
                "first_treatment_date_hidden":entry["onset"].isoformat() if typ=="TB" else None,
                "source_study_cohort_key_hidden":key,
                "selection":"case-control around first-treatment or synthetic-control anchor",
                "source_pig_count":entry["n_source_pigs"]}))
            assessment["windows_sampled"][split]+=1
            if target:assessment["states_with_true_next7_treatment"]+=1
            t+=datetime.timedelta(days=7)
    assessment["cohorts_with_positive_examples"]=len(assessment["cohorts_with_positive_examples"])
    assessment["retained_source_pen_group_counts"]={sp:len(items) for sp,items in groups.items()}
    result=compiler.report()
    result["study_audit"]=assessment
    return result,compiler

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--local-dir",type=Path)
    a=p.parse_args()
    r,c=build(a.local_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.local_dir:c.finish(a.out)
    else:a.out.write_text(json.dumps(r,indent=2,default=str)+"\n")
    print("SWEDISH_REAL_TREATMENT_HISTORICAL:"+json.dumps({
        "states":r["state_count"],"split":r["source_splits"],
        "reference_labels":r["observed_valid_question_labels"],
        "audit":r["study_audit"]},default=str))
