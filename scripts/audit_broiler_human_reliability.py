"""Independent observer agreement on real Big Broiler human behavior annotations.

Access only the ten triple-annotated video CSVs within a 33 GB zip over
HTTP RANGE. No giant download, no media redistribution, no model predictions.
The three annotators' agreement is computed PAIRWISE (no dependence on a
consensus label which may itself be derived from these same annotators).

F1 is FRAME-ROW positive-label agreement, *not* segment/onset agreement.
Samples from every 25th frame only, with video unit grouping. Empty action
string treated as 'no recorded action', never as missing image video.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd
from remotezip import RemoteZip
from sklearn.metrics import f1_score,cohen_kappa_score

URL="https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
SOURCE="https://github.com/decide-ugent/big-broiler-dataset"
SAMPLE=25
ANNOTATORS=("actions_annotator_1","actions_annotator_2","actions_annotator_3")
def parse(s):
    s="" if pd.isna(s) else str(s).strip()
    return {p.strip() for p in re.split(r"[;,|]",s) if p.strip()} if s else set()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    with RemoteZip(URL,initial_buffer_size=4*1024*1024) as z:
        files=sorted([name for name in z.namelist() if "annotations/" in name and name.endswith(".csv")])
        if len(files)!=102:raise RuntimeError("Expected 102 Big Broiler videos, refuse shifted source")
        holdout=[n for n in files if int(Path(n).stem.split("_")[0]) in range(92,102)]
        if len(holdout)!=10:raise RuntimeError("Expected exactly ten triple-human holdout videos")
        video_data={}
        all_labels=set()
        for name in holdout:
            with z.open(name) as f:
                df=pd.read_csv(f,low_memory=False)
            required=set(ANNOTATORS)|{"frame","track","actions_ground"}
            if not required.issubset(df.columns):
                raise RuntimeError(f"Missing independent annotator columns: {name}")
            if df.duplicated(["frame","track"]).any():
                raise RuntimeError(f"Source repeated frame+track: {name}")
            raw_rows=len(df)
            df=df.loc[df.frame%SAMPLE==0,list(required)].copy()
            sources={}
            for key in ANNOTATORS+("actions_ground",):
                parsed=[parse(v) for v in df[key].to_numpy()]
                sources[key]=parsed
                for setv in parsed:all_labels.update(setv)
            video_data[name]={
                "rows":len(df),"raw_rows":raw_rows,"sources":sources,
                "empty_reference_action_fraction":float((df["actions_ground"].isna()|(df["actions_ground"].astype(str).str.strip()=="")).mean()),
            }
        labels=sorted(all_labels)
        if len(labels)<12:raise RuntimeError("Unusually few source behavior names")
        out={"source_archive":URL,"upstream_code":SOURCE,"license":"AGPL-3.0 (per source README)",
             "scope":"Only 10 video IDs 092–101 with three concurrent independent annotators",
             "raw_total_rows":sum(v["raw_rows"] for v in video_data.values()),
             "evaluated_rows":sum(v["rows"] for v in video_data.values()),
             "sample_stride_frames":SAMPLE,
             "video_ids":[Path(k).stem for k in holdout],
             "source_labels":labels,
             "human_pairwise_binary_agreement":{},
             "video_metadata":{Path(k).stem:{"rows":v["rows"],"raw_rows":v["raw_rows"],
                    "empty_reference_action_fraction":v["empty_reference_action_fraction"]}
                    for k,v in video_data.items()},
             "source_caveats":["Positive frame agreement differs from event-onset and segment-level F1",
                "All human annotators see the same clips; no evidence here of independent animal/site transfer",
                "No inferred positive/negative affect labels; Play/Preening are behaviors only",
                "An empty string is interpreted as 'no action recorded'; annotation coverage is source protocol dependent.",
                "These ten videos remain reserved for unseen-video evaluation; do not train models on them."]}
        counts={}
        for label in labels:
            comparisons=[]
            presentvideos=0
            reference_n=0
            for i,j in [(0,1),(0,2),(1,2)]:
                va=[];vb=[]
                for record in video_data.values():
                    va.extend(label in c for c in record["sources"][ANNOTATORS[i]])
                    vb.extend(label in c for c in record["sources"][ANNOTATORS[j]])
                aa=np.asarray(va,bool);bb=np.asarray(vb,bool)
                tp=int((aa&bb).sum()); fp=int((aa&~bb).sum());fn=int((~aa&bb).sum())
                score=2*tp/max(1,2*tp+fp+fn)
                comparisons.append({"pair":f"{i+1}-{j+1}","f1":score,"tp":tp,"disagree":fp+fn,
                                    "positive_a":int(aa.sum()),"positive_b":int(bb.sum())})
            for record in video_data.values():
                if any(label in t for t in record["sources"]["actions_ground"]):presentvideos+=1
                reference_n+=sum(label in t for t in record["sources"]["actions_ground"])
            agreement=float(np.mean([v["f1"] for v in comparisons]))
            out["human_pairwise_binary_agreement"][label]={
               "pairs":comparisons,"pairwise_mean_f1":agreement,
               "support_ground":reference_n,"ground_positive_videos":presentvideos,
               "interpretation":"HUMAN_CONSENSUS_CANDIDATE" if (
                  reference_n>=35 and presentvideos>=2 and agreement>=.65
                  and min(v["f1"] for v in comparisons)>=.5) else "RARE_OR_AMBIGUOUS"}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2)+"\n")
    print("BROILER_HUMAN_RELIABILITY:"+json.dumps({
        "video_count":len(out["video_ids"]),"n_rows":out["evaluated_rows"],
        "behaviors":[{"name":k,"mean_pairwise_f1":round(v["pairwise_mean_f1"],4),
                      "reference_rows":v["support_ground"],
                      "positive_videos":v["ground_positive_videos"],
                      "status":v["interpretation"]}
                     for k,v in sorted(out["human_pairwise_binary_agreement"].items(),
                                       key=lambda x:x[1]["pairwise_mean_f1"],reverse=True)]}))
if __name__=="__main__":main()
