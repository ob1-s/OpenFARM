"""Two independently rated welfare-relevant questions per same observed cow state.

Uses frozen OpenFARM v0.2 compiler UNMODIFIED. Source 13,716 cows, 50
independent farms with human graded gait and body condition. Model-visible
state contains ONLY contemporaneous recorded farm-management data + cow breed,
NOT actual animal gait, BCS, lameness prevalence, animal IDs, site/time.

These are clinical REFERENCE TASKS, not a proven predictive welfare monitor.
The 16-farm Cowlevel experiment is an overlapping subset; it must NOT be
pooled into this dataset as independent supervision.

Produces 3 isolated files per split (state, references, provenance) only
under optional --local-dir; CI uploads ONLY aggregate manifest.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_brazil_cow_welfare import CAT,NUM,source,split_farms,make_states
from openfarm_query_compiler_v02 import Compiler,Observation,Question

def primitive(v):
    if v is None or (isinstance(v,float) and not math.isfinite(v)):return None
    if pd.isna(v):return None
    if isinstance(v,(np.integer,int)):return int(v)
    if isinstance(v,(np.floating,float)):return float(v)
    return str(v)

def build(local_dir=None):
    cows,herd=source()
    partitions=split_farms(herd)
    membership={fid:split for split,ids in partitions.items() for fid in ids}
    f=make_states(cows,herd)
    c=Compiler(local_dir=local_dir)
    for idx,row in enumerate(f.itertuples(index=False,name=None)):
        obj=dict(zip(f.columns,row))
        farm=int(obj["farm_ID"])
        if farm not in membership:raise RuntimeError("Cow farm not in heldout groups")
        state={name:primitive(obj[name]) for name in CAT+NUM}
        gait=primitive(obj["Gait score"])
        bcs=primitive(obj["BCS"])
        if not isinstance(gait,(int,float)):raise RuntimeError("Missing original gait grade")
        questions=[
          Question(question_id="brazil.cow.gait_score",
             text="What ordinal locomotion score (1 to 5) did the clinical observer assign this cow?",
             primitive="Score",reference_family="human_gait_assessment",
             evidence_origin="trained-human clinical locomotion assessment (one study visit)",
             value=gait,valid=True,readiness="reference_valid_predictor_weak"),
          Question(question_id="brazil.cow.body_condition_score",
             text="What body-condition score (1.5 to 5) did the clinical observer assign this cow?",
             primitive="Score",reference_family="human_body_condition_assessment",
             evidence_origin="independently scored body condition in same clinical survey",
             value=bcs,valid=bcs is not None,
             mask_reason=None if bcs is not None else "missing_clinical_body_condition",
             readiness="reference_valid_predictor_weak")
        ]
        observation=Observation(
           source="brazil_cow_2016_clinical",state_id=f"brancows:{idx}",
           split=membership[farm],group_scope="original_farm",
           group_id=str(farm),
           modalities=("structured_herd_and_cow_metadata",),
           features=state,
           questions=questions,
           audit={"clinical_reference_sha256":"c43d3f9b8c883f6f0040ed66c9752b7441cc5c68b7603e376c1d2ef7778fabba",
                  "herd_management_sha256":"89e9c75c970f6f76cce793f343c2a371fb07cbace835f5c782d43f39d3835318",
                  "dataset_license":"CC BY 4.0","clinical_visit_time":"single cross-sectional 2016 visit; no valid temporal ordering",
                  "disallowed_input":"gait, BCS, prevalence, source IDs, animal exam order",
                  "label_status":"independently human-observed clinical reference; prediction from metadata alone remains weak",
                  "non_independence":"rows nested in 50 farm groups; 16-farm Cowlevel subset overlaps source"})
        c.add(observation)
    report=c.report()
    report["source_doi"]="10.5281/zenodo.3408818"
    report["rights"]="CC BY 4.0 original archive, attribute Bran et al."
    report["independent_rater_families_per_cow"]=2
    report["interpretation"]="Valid clinical SCORE references; model-visible managerial metadata may be insufficient. Not a Level-3 human attention task."
    report["frozen_farm_splits"]=partitions
    return report,c

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--local-dir",type=Path)
    a=p.parse_args()
    r,c=build(a.local_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2)+"\n")
    if a.local_dir:
        c.finish(a.out)
    print("BRAZIL_CLINICAL_2QUESTION_COMPILER:"+json.dumps({
         "state_count":r["state_count"],"label_slots":r["available_question_slots"],
         "valid_labels":r["observed_valid_question_labels"],
         "masked":r["masked_question_slots"],
         "families":r["independent_reference_families_histogram"],
         "splits":r["source_splits"],
         "rights":r["rights"]}))
