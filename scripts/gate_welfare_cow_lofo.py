"""Decision admission gate for existing 16-farm leave-one-farm-out results.

This intentionally does NOT refit/tune a model. It reads an independently
generated report and asks whether gains generalize across *source farms*.
Scores: Brier / independent-clinical outcomes; strongly conservative gate.
"""
import argparse,json
from pathlib import Path

def assess(report):
    if report.get("independent_sites") != 16:
        raise RuntimeError("Expected 16 full farm holdouts; no row-random substitutes")
    expected={"clinical_lameness","severe_lameness","low_body_condition"}
    if set(report.get("tasks",{}))!=expected:
        raise RuntimeError("Unexpected clinical task family / missing results")
    decisions={}
    for task,v in report["tasks"].items():
        variants={}
        for name,run in v["feature_ablations"].items():
            if run["farm_folds"]!=16 or len(run["per_farm"])!=16:
                raise RuntimeError("Incomplete cross-farm evaluation")
            wins=run["farms_beating_train_prior_on_brier"]
            weighted=run["cow_weighted_brier_gain"]
            median=run["unweighted_median_brier_gain"]
            # A gate for *robust incremental information*: >=75% of farms
            # improve over their independent training-farm prevalence prior,
            # AND median and cow-weighted gain are strictly > 0.
            passes=(wins>=12 and median>0 and weighted>0)
            variants[name]={
                "folds":16,"farms_better_than_prior":wins,
                "farm_weighted_brier_gain":weighted,
                "median_farm_brier_gain":median,
                "genuine_cross_farm_signal_gate": "PASS" if passes else "FAIL",
            }
        decisions[task]={
            "reference_family":v["reference_family"],
            "per_feature_variant":variants,
            "promoted_for_generalizable_welfare_evaluation":False,
            "reason":"Cross-farm baseline gains are an internal robustness diagnostic; no external validation or live welfare-alert labels."
        }
    if decisions["clinical_lameness"]["reference_family"]==decisions["severe_lameness"]["reference_family"]:
        decisions["severe_lameness"]["independent_reference_family"]=False
    else:
        # The CLI still must mark this derivative distinct from independent BCS.
        decisions["severe_lameness"]["independent_reference_family"]=False
    return {"evidence":"Observed source human clinical gait and body condition, not intervention outcomes",
            "test_unit":"farm", "distinct_reference_families":2,
            "derived_rewrites_not_new_evidence":["severe_lameness from same gait score"],
            "threshold_rule":"PASS if >=12/16 whole farms beat train-only prevalence prior in Brier, positive median and cow-weighted Brier gains",
            "tasks":decisions,
            "policy":"Keep descriptive evaluations available even on FAIL; never claim a deployed clinical monitor on this evidence alone."}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.source.read_text())
    verdict=assess(report)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(verdict,indent=2)+"\n")
    print("WELFARE_COW_FARM_GATE:"+json.dumps(verdict))
