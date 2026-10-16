"""Aggregate source-audited query manifests without inflating independence.

Input: one pinned-source compiler result per source; a source must occur ONCE
per aggregate. Multiple PigWeb leave-one-site-out protocols share observations
and must be compared as alternate protocols, not summed as new evidence.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse,glob,json

def combine(paths):
    if not paths:raise RuntimeError("No source reports found")
    by_source={}
    total=Counter()
    family_hist=Counter()
    question_valid=Counter()
    masked_reasons=Counter()
    readiness={}
    for path in paths:
        data=json.loads(Path(path).read_text())
        sources=data["source_states"]
        if len(sources)!=1:raise RuntimeError("Expected one source per manifest")
        source=next(iter(sources))
        if source in by_source:
            raise RuntimeError(f"Duplicate source {source}: do not count multiple evaluation folds as more animals")
        if int(sources[source])!=data["state_count"]:
            raise RuntimeError("State count mismatch")
        if int(data["available_question_slots"])!=int(data["observed_valid_question_labels"])+int(data["masked_question_slots"]):
            raise RuntimeError("Question mask accounting does not balance")
        for k in ("state_count","observed_valid_question_labels","available_question_slots",
                  "masked_question_slots","multi_modality_states"):
            total[k]+=int(data[k])
        for count,freq in data["independent_reference_families_histogram"].items():
            family_hist[int(count)]+=int(freq)
        question_valid.update(data["valid_by_question"])
        masked_reasons.update(data["masked_by_reason"])
        by_source[source]={
            "states":data["state_count"],
            "valid_question_labels":data["observed_valid_question_labels"],
            "question_slots":data["available_question_slots"],
            "masked":data["masked_question_slots"],
            "mean_valid_questions_per_state":data["mean_valid_question_labels_per_state"],
            "mean_primary_question_targets_per_state":data["mean_non_derived_question_labels_per_state"],
            "mean_independent_measured_references_per_state":
                 data["mean_independently_measured_reference_families_per_state"],
            "source_splits":data["source_splits"][source],
            "source_groups":data["group_count_by_source"][source],
            "derived_diagnostic_questions":data["derived_question_ids"],
            "source_evidence":data.get("source_evidence",{}),
        }
    n=total["state_count"]
    if sum(family_hist.values())!=n:raise RuntimeError("Independent family histogram cannot account for states")
    return {
        "report":"OpenFARM v0.2 source-provenance-aware multi-question compiler",
        "source_count":len(by_source),"compiled_observation_states":n,
        "model_input_multichannel_states":total["multi_modality_states"],
        "valid_typed_question_labels":total["observed_valid_question_labels"],
        "available_slots":total["available_question_slots"],
        "masked_slots":total["masked_question_slots"],
        "mean_valid_question_labels_per_state":round(total["observed_valid_question_labels"]/n,5),
        "mean_independent_reference_families_per_state":round(
            sum(k*v for k,v in family_hist.items())/n,5),
        "independent_reference_families_by_state":dict(sorted(family_hist.items())),
        "valid_question_labels_by_id":dict(question_valid),
        "masked_reasons":dict(masked_reasons),
        "per_source":by_source,
        "science_gate":"Compilation confirms label/time/group accounting, not clinical construct validity or product readiness.",
        "unit_of_independence":"Per source animal/video/site as specified by the original protocol, not rows or regenerated question strings",
        "notes":[
            "Each original cattle/meerkat Choice ethogram remains ONE ground-truth family even when compiled into 4+ binary questions.",
            "Same feeder instrument at multiple horizons does NOT create an additional independent assay.",
            "PigWeb cortisol, oxytocin and eye temperature are independently measured references; eye temperature is site-limited.",
            "A continuous biomarker delta and its binary sign are two TARGET TYPES from one original assay.",
            "All source media rights and task readiness remain per-source; this report does not lift research-only/noncommercial restrictions."
        ]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reports",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    files=[]
    for pattern in a.reports:
        found=sorted(glob.glob(pattern))
        if not found:raise RuntimeError("No results for input glob "+pattern)
        files.extend(found)
    data=combine(files)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
    print("OPENFARM_QUERY_SUITE_V02:"+json.dumps({
        k:data[k] for k in ("source_count","compiled_observation_states",
                           "model_input_multichannel_states","valid_typed_question_labels",
                           "masked_slots","mean_independent_reference_families_per_state")}))
if __name__=="__main__":main()
