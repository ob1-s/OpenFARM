"""Build audited multi-question OpenFARM v0.2 observation states from source bytes.

Examples:
  python scripts/build_openfarm_decision_queries_v02.py --source pigweb --report /tmp/pigweb.json
  python scripts/build_openfarm_decision_queries_v02.py --source hotpig --report /tmp/hotpig.json
  python scripts/build_openfarm_decision_queries_v02.py --source cattle --report /tmp/cattle.json
  python scripts/build_openfarm_decision_queries_v02.py --source equifacs --report /tmp/equifacs.json

--local-dir optionally writes model states, answer keys and hidden group
provenance to three separate files. All source rows remain local. Never
publish original third-party media/rows or present proposed endpoints as welfare
ground truth. Dataset versions are pinned in imported original audit scripts.
"""
from __future__ import annotations
import argparse,io,json,math,zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from openfarm_query_compiler_v02 import Compiler,Observation,Question
from build_pigweb_historical_v02 import build as build_pigweb,SITE_ORDER
from validate_temporal_contract_v02 import pre_only_features,assert_pre_only,assert_future_clock
from build_public_cattle_behavior_v0 import source as cattle_source,prepare as cattle_prepare,groups as cattle_groups
from evaluate_hotpig_feeder_candidate import source_bytes as hotpig_bytes,FEATURES as HOTPIG_FEATURES
from build_equifacs_clinical_v0 import produce as equifacs_source

def pigweb(c:Compiler,test_site="WUR"):
    manifest,examples=build_pigweb()
    if test_site not in SITE_ORDER:raise ValueError("Unknown PigWeb site")
    # 4 leave-one-institute-out protocols with independent calibration source.
    # Do not mix the same state across folds in the same report.
    idx=SITE_ORDER.index(test_site)
    calibration_site=SITE_ORDER[(idx+1)%len(SITE_ORDER)]
    for e in examples:
        p=e["provenance"]
        site=p["institute"]
        split=("test" if site==test_site else "calibration"
               if site==calibration_site else "train")
        # CRITICAL temporal audit 2026-10-16:
        # Human-coded behavior was recorded during the stimulus exposure;
        # source metadata supplies a PRE saliva collection time, NOT an
        # independently timestamped behavioral-observation completion time.
        # Behavior therefore cannot be assumed to exist at the PRE cutoff.
        # The future hormone decision must use PRE-only measured inputs.
        # Behavior/physiology association remains a different nonforecast
        # scientific analysis; do not silently pass it as a future state.
        assert_future_clock(p["pre"],p["post"])
        feat=pre_only_features(e["features"])
        assert_pre_only(feat)
        feat={k:(float(v) if v is not None else None) for k,v in feat.items()}
        missing=[k for k,v in feat.items() if v is None]
        qs=[]
        for assay,unit in (("cortisol","ng/mL"),("oxytocin","pg/mL"),("eye_IRT","temperature unit")):
            delta=e["targets"][assay+"_delta"]
            inc=e["targets"][assay+"_increase"]
            fid=f"pigweb.{assay}"
            # Both binary and signed delta are scored views of the SAME
            # independent laboratory assay; never count them as two assays.
            qid=f"{fid}.increased"
            kwargs={"relation":"future","reference_time":p["post"],
                    "reference_family":fid,"evidence_origin":"measured physiological assay",
                    "readiness":"site_limited_candidate" if assay=="eye_IRT" else "audited_candidate"}
            qs.append(Question(question_id=qid,text=f"Did measured {assay} increase after this observation?",
                 primitive="Noul",value=(int(inc) if inc is not None else None),
                 valid=inc is not None,mask_reason=None if inc is not None else "missing_paired_assay",
                 **kwargs))
            qs.append(Question(question_id=f"{fid}.delta",
                 text=f"What was the signed subsequent change in measured {assay} ({unit})?",
                 primitive="Value",value=float(delta) if delta is not None else None,
                 valid=delta is not None,mask_reason=None if delta is not None else "missing_paired_assay",
                 derived_from=qid,**kwargs))
        obs=Observation(source="pigweb",state_id=":".join(("pigweb",p["institute"],p["pig_id"],p["context"])),
          split=split,group_scope="pig_within_institute",
          group_id=f"{p['institute']}:{p['pig_id']}",
          modalities=("pre_session_biomarker_values",),
          features=feat,cutoff=p["pre"],questions=qs,
          audit={"source_revision":manifest["source"],"license":"CC BY 4.0",
                 "recording_context_hidden":p["context"],
                 "missing_model_input_fields":missing,
                 "post_interval_minutes":p["delta_minutes"],
                 "excluded_temporally_unavailable_features":[k for k in e["features"]
                    if k.startswith("observed_")],
                 "eligibility":"PRE_ONLY_future_biomarker_forecast; observed_behavior_is_not_available_at_pre"})
        c.add(obs)
    return {"temporal_audit":"pre_biomarker_only__behavior_during_stimulus_excluded",
            "frozen_source_sha256":manifest["source_sha256"],"site_test":test_site,
            "site_calibration":calibration_site,"source_sessions":manifest["matched_pre_to_post_sessions"]}

def cattle(c:Compiler):
    raw,source_manifest=cattle_source()
    df,integrity=cattle_prepare(raw)
    if integrity["raw_rows"]!=9222:raise RuntimeError("Unexpected source observation total")
    parts=cattle_groups(df)["breed_balanced_group_holdout"]
    membership={id:split for split,ids in parts.items() for id in ids}
    if len(membership)!=24:raise RuntimeError("Cattle split does not span all animals")
    mapping={"GR":"grazing","WA":"walking","RE":"resting","RU":"ruminating"}
    for i,r in enumerate(df.itertuples(index=False)):
        source_behavior=r.Behavior
        if source_behavior not in mapping:raise RuntimeError("Unrecognized ethogram code")
        chosen=mapping[source_behavior]
        family="cattle.human_video_ethogram"
        primary_id="cattle.behavior.choice"
        qs=[Question(question_id=primary_id,text="Which human-observed behavior dominated this minute?",
                     primitive="Choice",options=tuple(mapping.values()),value=chosen,
                     reference_family=family,evidence_origin="human_video_observed_behavior",
                     readiness="bounded_research_approved")]
        for name in mapping.values():
            qs.append(Question(question_id=f"cattle.is_{name}",
                 text=f"Was {name} the human-observed dominant behavior this minute?",
                 primitive="Noul",value=int(chosen==name),reference_family=family,
                 evidence_origin="derived_one_vs_rest_same_human_code",
                 derived_from=primary_id,readiness="diagnostic_slice_not_independent_GT"))
        c.add(Observation(source="cattle",state_id=f"cattle:source_row:{i}",
            split=membership[r.ID],group_scope="cow",group_id=str(r.ID),
            modalities=("wearable_preprocessed_motion_index",),
            features={"motion_index":float(r.MI)},questions=qs,
            audit={"license":"CC BY 4.0","source_sha256":source_manifest["file_sha256"],
                 "source_ordinal_row":i,"nonunique_source_time_key":True,
                 "no_temporal_forecast_approved":True}))
    return {"source_integrity":integrity,"sha256":source_manifest["file_sha256"],
            "groups":{k:len(v) for k,v in parts.items()}}

def hotpig(c:Compiler,stride=15,horizon=15):
    source=hotpig_bytes()
    z=zipfile.ZipFile(io.BytesIO(source))
    split={"train":range(1,17),"calibration":range(17,21),"test":range(21,25)}
    valid_samples=0
    for partition,pigs in split.items():
        for pig in pigs:
            df=pd.read_csv(z.open(f"series/P{pig}.csv"))
            if len(df)!=23040:raise RuntimeError("Unexpected 16-day minute count")
            numeric=df[HOTPIG_FEATURES+["feed"]].apply(pd.to_numeric,errors="coerce")
            feature_arr=numeric[HOTPIG_FEATURES].to_numpy(dtype=float)
            feed=numeric["feed"].to_numpy(dtype=float)
            fvalid=np.isfinite(feed)
            # Missing prior feeder reference leaves onset ambiguous; event is
            # only measured if t..t+H are ALL observed instrument minutes.
            starts=np.zeros(len(feed),dtype=bool)
            starts[1:]=fvalid[1:] & fvalid[:-1] & (feed[1:]>0)&(feed[:-1]<=0)
            for t in range(0,len(df),stride):
                if not np.isfinite(feature_arr[t]).all():continue
                features={name:float(feature_arr[t][j]) for j,name in enumerate(HOTPIG_FEATURES)}
                cur_good=bool(fvalid[t])
                future_good=(t+horizon<len(df) and
                             bool(fvalid[t:t+horizon+1].all()))
                yes=bool(starts[t+1:t+horizon+1].any()) if future_good else None
                base={"reference_family":"hotpig.electronic_feeder_series",
                      "evidence_origin":"independently_measured_electronic_feeder",
                      "readiness":"bounded_candidate_historical" }
                qs=[
                  Question(question_id="hotpig.feeder_now",
                           text="Did the instrument register feed intake in this minute?",
                           primitive="Noul",value=int(feed[t]>0) if cur_good else None,
                           valid=cur_good,mask_reason=None if cur_good else "missing_current_feeder_reference",
                           readiness="approved_bounded_research",reference_family=base["reference_family"],
                           evidence_origin=base["evidence_origin"]),
                  Question(question_id="hotpig.next15_new_feeding_bout",
                           text="Will a newly instrument-registered feeder-active period begin in the next 15 minutes?",
                           primitive="Noul",value=int(yes) if future_good else None,
                           valid=bool(future_good),
                           mask_reason=None if future_good else ("end_of_record" if t+horizon>=len(df)
                                                                 else "missing_future_feeder_reference"),
                           relation="future",reference_index=t+horizon,**base)
                ]
                c.add(Observation(source="hotpig",state_id=f"pig:{pig}:minute:{t}",
                    split=partition,group_scope="pig",group_id=str(pig),
                    modalities=("computer_vision_precomputed_behavior_counts",),
                    features=features,questions=qs,cutoff_index=t,
                    audit={"source_minute_index":t,"source_pig":pig,
                           "source_zip_sha256":z.comment.decode("utf-8","replace")[:40] if z.comment else None,
                           "license":"CC BY 4.0",
                           "source_relative_only_no_actual_calendar_time":True,
                           "future_onset_target_is_instrumental_not_welfare_alert":True}))
                valid_samples+=1
    return {"source_states_compiled":valid_samples,"stride_minutes":stride,
            "lookahead_minutes":horizon,"n_source_pigs":24,
            "source_grouping":"whole pig, 16/4/4; one experimental site"}

def meerkat(c:Compiler):
    # Human visual ethogram is the reference; model receives only human
    # bounding-box geometry and past-only movement from the original existing
    # 20-video-heldout benchmark. The source video is the evaluation group.
    from benchmark_meerkat_video_heldout import (archive,groups,extract_one,
                                                 FEATS as MEERKAT_FEATS)
    z,sha=archive()
    names=sorted(n for n in z.namelist() if n.endswith(".csv"))
    if len(names)!=20:raise RuntimeError("Expected exactly 20 source videos")
    label_codes=json.loads(z.read("behaviours.json"))
    names_by_id={int(v):str(k) for k,v in label_codes.items()}
    valid_ids=tuple(sorted(i for i in names_by_id if i in range(1,16)))
    if len(valid_ids)!=15:raise RuntimeError("Unexpected human behavior class ontology")
    label_options=tuple(names_by_id[i] for i in valid_ids)
    # Data analysis slices of a SINGLE human ethogram reference, never
    # four independently annotated extra tasks.
    probe_codes={
        name:code for name,code in label_codes.items()
        if any(t in name.casefold().replace(" ","") for t in
               ("forag","guard","groom","playfight"))
        and int(code) in valid_ids
    }
    if len(probe_codes)<3:raise RuntimeError("Expected human ethogram probes missing")
    folds=groups(names)
    lookup={video:part for part,vids in folds.items() for video in vids}
    for video in names:
        X,y,quality=extract_one(z,video)
        if len(y)==0:raise RuntimeError(f"No valid observations in {video}")
        for i,(x,source_code) in enumerate(zip(X,y)):
            source_code=int(source_code)
            if source_code not in names_by_id or source_code==16:
                raise RuntimeError("Unknown/occluded class cannot become false negative")
            chosen=names_by_id[source_code]
            family="meerkat.human_video_ethogram"
            primary_id="meerkat.behavior.choice"
            qs=[Question(question_id=primary_id,primitive="Choice",
                text="Which human-observed behavior is shown by this visible tracked meerkat?",
                options=label_options,value=chosen,
                reference_family=family,evidence_origin="human_labeled_bbox_video_ethogram",
                readiness="bounded_noncommercial_research")]
            for code_name,code in sorted(probe_codes.items()):
                slug="".join(c if c.isalnum() else "_" for c in code_name.lower())
                qs.append(Question(question_id=f"meerkat.is_{slug}",primitive="Noul",
                    text=f"Is the human behavior label {code_name}?",
                    value=int(source_code==int(code)),
                    reference_family=family,derived_from=primary_id,
                    evidence_origin="one_vs_rest_of_same_human_ethogram",
                    readiness="diagnostic_slice_only"))
            c.add(Observation(source="meerkat",
                state_id=f"video:{video}:postfilter_row:{i}",
                split=lookup[video],group_scope="original_source_video",
                group_id=video,modalities=("source_human_bbox_with_past_motion",),
                features={name:float(x[col]) for col,name in enumerate(MEERKAT_FEATS)},
                questions=qs,audit={"source_sha256":sha,"source_video_file":video,
                    "license":"CC BY-NC-SA 4.0; strictly noncommercial use",
                    "source_screen":"adult_meerkat_non_occluded_valid_human_behavior",
                    "sampling":"every 50th original frame",
                    "not_raw_video_frame_inference":True,
                    "no_animal_heldout_claim":True}))
    return {"sha256":sha,"whole_video_groups":{k:len(v) for k,v in folds.items()},
            "codebook":names_by_id,
            "derived_probe_code_names":list(sorted(probe_codes)),
            "noncommercial_only":True}

def equifacs(c:Compiler):
    manifest,source=equifacs_source()
    for split,(entries,provenance) in source.items():
        for i,(e,p) in enumerate(zip(entries,provenance)):
            row_features=e["features"]
            qs=[
             Question(question_id="equifacs.vet_ordinal_observation",primitive="Score",
               text="What ordinal rating (0 no, 1 mild, 2 moderate, 3 severe) did the independent veterinarian assign?",
               value=int(p["original_ordinal_label"]),reference_family="equifacs.independent_vet",
               evidence_origin="clinician_scored_interval",
               readiness="nonranking_pilot_sparse_ordinal"),
             Question(question_id="equifacs.vet_any_pain",primitive="Noul",
               text="Did a different veterinary assessor label pain present?",
               value=int(e["label"]),reference_family="equifacs.independent_vet",
               evidence_origin="binary_threshold_of_same_independent_clinician_score",
               derived_from="equifacs.vet_ordinal_observation",
               readiness="nonranking_pilot")
            ]
            # Model input only expert-coded FACS features, not labels/video ID.
            c.add(Observation(source="equifacs",
                state_id=f"equifacs:{p['media']}:{p['start_seconds']}:{p['rater']}:{i}",
                split=split,group_scope="media_id",group_id=str(p["media"]),
                modalities=("expert_facial_action_timeline",),
                features={k:float(v) for k,v in row_features.items()},
                questions=qs,
                audit={"license":"PLOS supplemental annotation reuse only; media rights separate",
                       "source_video_id":p["media"],"vet_annotator":p["rater"],
                       "timing":[p["start_seconds"],p["end_seconds"]],
                       "clinical_animal_uniqueness":"NOT independently proven",
                       "noncodeable_intervals_filtered":True}))
    return {"report":{k:v for k,v in manifest.items() if k in
                      ("source_media_count","accepted_intervals","splits","quality_excluded_intervals")}}

BUILDERS={"pigweb":pigweb,"cattle":cattle,"hotpig":hotpig,"equifacs":equifacs,"meerkat":meerkat}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,choices=sorted(BUILDERS))
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--local-dir",type=Path)
    p.add_argument("--pigweb-test-site",default="WUR",choices=SITE_ORDER)
    args=p.parse_args()
    compiler=Compiler(local_dir=args.local_dir)
    meta=(pigweb(compiler,test_site=args.pigweb_test_site)
          if args.source=="pigweb" else BUILDERS[args.source](compiler))
    report=compiler.finish(args.report)
    report["source_evidence"]=meta
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    slim={k:report[k] for k in ("state_count","available_question_slots","observed_valid_question_labels",
       "masked_question_slots","mean_valid_question_labels_per_state",
       "mean_non_derived_question_labels_per_state",
       "mean_independently_measured_reference_families_per_state",
       "source_splits","valid_by_question","masked_by_question")}
    print("OPENFARM_QUERY_COMPILER_V02:"+json.dumps({"source":args.source,**slim}))

if __name__=="__main__":main()
