#!/usr/bin/env python3
"""Fail-closed structural/source-provenance gate for Decision Index v0.1.

This validates task contracts; it does not pretend that upstream datasets from
other draft PRs have been merged or grant research/deployment approval.
"""
import json,re,sys,hashlib
from pathlib import Path

PATH=Path("benchmarks/decision-index/v0.1/tasks.json")
ALLOWED_L={"L1","L2","L3","L2_L3_RISK"}
ALLOWED_READY={"APPROVED_BOUNDED_RESEARCH_TRAIN_EVAL","APPROVED_BOUNDED_NONCOMMERCIAL_RESEARCH",
   "TASK_VALIDATED_REVIEW_PENDING","GEOMETRY_TASK_VALIDATED_VIDEO_NOT_VERIFIED",
   "RESEARCH_EVAL_CANDIDATE"}
REQUIRED={"id","level","species","input_modality","output_type","scope","source",
          "data","reference","metric_primary","metric_secondary","split_status","readiness","head_ref"}
def validate(doc):
    assert doc["schema_version"]=="openfarm-decision-index-v0.1"
    assert doc["status"]=="RESEARCH_INDEX_NOT_PUBLIC_LEADERBOARD"
    assert len(doc["tasks"])>=8 and len(doc["hard_gates"])>=9
    assert doc["scoring_rules"]["no_aggregate_leaderboard"]
    seen=set();species=set()
    for task in doc["tasks"]:
        assert REQUIRED.issubset(task),("missing task fields",task.get("id"))
        name=task["id"];assert re.fullmatch(r"[a-z0-9_]+_v01",name),name
        assert name not in seen,name
        seen.add(name);species.add(task["species"])
        assert task["level"] in ALLOWED_L,name
        assert task["readiness"] in ALLOWED_READY,name
        assert isinstance(task["source"],dict) and task["source"].get("name"),name
        assert task["head_ref"].startswith("research/"),name
        assert len(task["data"])>=2,name
        assert task["metric_primary"] and len(task["metric_secondary"])>=2,name
        assert "video" not in task["scope"] or task["data"],name
        for key,value in task["data"].items():
            if isinstance(value,int):assert value>=0,(name,key,value)
        if name.startswith("wong"):
            assert task["data"]["train"]+task["data"]["calibration"]+task["data"]["test"]==340
            assert "case" in task["data"]["independent_groups"]
            assert "three_vote" in task["metric_primary"]
        if name.startswith("farrow"):
            assert task["data"]["cases"]==30 and task["data"]["owner_responses"]==5316
            assert task["data"]["vet_panel_votes_individually_accessible"] is False
        if name.startswith("big_broiler"):
            assert "video_not_verified" in task["readiness"].lower(),name
    assert len(species)>=5,species
    return {"task_ids":sorted(seen),"species":sorted(species),
            "approved_research_task_count":sum(t["readiness"].startswith("APPROVED_") for t in doc["tasks"]),
            "candidate_or_review_task_count":sum(not t["readiness"].startswith("APPROVED_") for t in doc["tasks"]),
            "task_manifest_sha256":hashlib.sha256(PATH.read_bytes()).hexdigest(),
            "warning":"Schema validation, not source rights or clinical approval."}
def main():
    report=validate(json.loads(PATH.read_text()))
    f=Path("reports/decision-index/v0.1_contract_validation.json")
    f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(report,indent=2)+"\n")
    print("OPENFARM_INDEX_V01_CONTRACT",json.dumps(report,sort_keys=True))
if __name__=="__main__":
    main()
