#!/usr/bin/env python3
"""Validate verifiable model scores against task schema; forbid cross-task ranking."""
import collections,json,math
from pathlib import Path
ROOT=Path("benchmarks/decision-index/v0.1")
INDEX=json.loads((ROOT/"tasks.json").read_text())
SCORES=json.loads((ROOT/"observed-results.json").read_text())
def main():
    assert SCORES["schema_version"]=="openfarm-decision-index-results-v0.1"
    assert SCORES["scientific_status"]=="EXPLORATORY_NOT_FROZEN_PUBLIC_LEADERBOARD"
    tasks={x["id"]:x for x in INDEX["tasks"]}
    seen=set()
    for line in SCORES["results"]:
        task=tasks[line["task_id"]]
        key=(line["task_id"],line["model"],line["primitive"])
        if key in seen:raise ValueError("Duplicate model/task/primitive score identity")
        seen.add(key)
        cap=task["data"]["cases"]
        assert 1<=line["n_scored"]<=cap
        assert 0<=line["urgent_caught"]<=line["urgent_reference_n"]<=cap
        assert line["urgent_reference_n"]==task["data"].get("critical_test_cases",task["data"].get("critical_cases"))
        for n in ("brier","exact_reference_agreement","undertriage_rate"):
            value=line[n]
            assert isinstance(value,(float,int)) and math.isfinite(value)
            assert 0<=value<= (2 if n=="brier" else 1)
        assert line["score_status"]=="ACTUAL_COMPLETED_EXPLORATORY"
        assert line["provenance"] and len(line["human_reference"])>15
    assert len(SCORES["results"])>=15
    k=collections.Counter(r["task_id"] for r in SCORES["results"])
    res={"n_scores":len(SCORES["results"]),"n_tasks_with_model_scores":len(k),"by_task":dict(k),
         "no_cross_task_ranking":True,"status":SCORES["scientific_status"]}
    p=Path("reports/decision-index/v0.1_observed_registry_validation.json")
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2)+"\n")
    print("OPENFARM_V01_OBSERVED_RESULTS_VALID",json.dumps(res,sort_keys=True))
if __name__=="__main__":main()
