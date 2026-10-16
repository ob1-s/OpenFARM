"""Track B: forensic audit of openly indexed cow gait trajectory source.

DO NOT assert source is commercially reusable: 2026 GitHub repo has NO explicit
license at the time of inspection. Training/eval admission requires rights.
Full 272 trajectories are inspected from a locally cloned pinned repo.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd

def audit(root):
    scores=root/"data/videos_lameness_scores.csv"
    files=sorted((root/"data/videos_keypoints").glob("*.csv"))
    if not scores.exists():raise RuntimeError("Lameness score reference missing")
    df=pd.read_csv(scores,dtype={"Video":str,"ID":str})
    cols={"Video","ID","hard_vote"}
    if not cols.issubset(df.columns):raise RuntimeError(f"Unexpected score schema: {df.columns}")
    if df["Video"].duplicated().any():raise RuntimeError("Duplicate score video keys")
    if not len(df)==272:raise RuntimeError("Unexpected source clip count")
    expected=set(df["Video"])
    found={f.stem for f in files}
    if expected!=found:raise RuntimeError(f"Unmatched trajectory/clinical score keys: {len(expected^found)}")
    buckets=collections.defaultdict(set)
    for r in df.itertuples(index=False):
        buckets[r.ID].add(int(r.hard_vote))
    score_counts={str(k):int(v) for k,v in df["hard_vote"].value_counts().sort_index().items()}
    frame_total=0
    hashes=set();dupe_media=0;missing=0;short=0
    frames=[]
    per_video_stats=[]
    for f in files:
        binary=f.read_bytes()
        h=hashlib.sha256(binary).hexdigest()
        if h in hashes:dupe_media+=1
        hashes.add(h)
        frame=pd.read_csv(f)
        need={"video","frame","LFHoof_x","LFHoof_y","LFHoof_likelihood"}
        if not need.issubset(frame.columns):
            raise RuntimeError(f"Unexpected trajectory feature schema in {f.name}")
        if frame["frame"].duplicated().any():raise RuntimeError(f"Repeated frame in {f.name}")
        n=len(frame);frames.append(n)
        frame_total+=n
        if n<12:short+=1
        lik=[col for col in frame if col.endswith("_likelihood")]
        coordinate=[col for col in frame if col.endswith("_x") or col.endswith("_y")]
        missing+=int(frame[lik+coordinate].isna().sum().sum())
        per_video_stats.append({"video":f.stem,"frames":n,"likelihood_below_half":int((frame[lik]<.5).sum().sum())})
    # KEY FACT: independent animal groups are the 98 unique ID values, NOT the 272 videos.
    mixed={k:sorted(v) for k,v in buckets.items() if len(v)>1}
    repeats=collections.Counter(df["ID"])
    code_license=root/"LICENSE"
    out={
     "repo":"https://github.com/hrussel/lstm-lameness-detection",
     "labels_SHA256":hashlib.sha256(scores.read_bytes()).hexdigest(),
     "n_videos":len(df),"n_cow_ids":len(buckets),
     "n_pose_csv_files":len(files),
     "score_distribution":score_counts,
     "n_animals_repeated":sum(n>1 for n in repeats.values()),
     "max_repeats_per_animal":max(repeats.values()),
     "animal_ids_with_multiple_ordinal_scores":len(mixed),
     "different_ordinal_score_examples":dict(list(mixed.items())[:12]),
     "exact_duplicate_trajectory_files":dupe_media,
     "total_frames":frame_total,"frames_min":min(frames),"frames_median":float(np.median(frames)),
     "frames_max":max(frames),"very_short_clips":short,
     "missing_pose_numeric_cells":missing,
     "sample_coordinate_schema":coordinate[:10],
     "sample_likelihood_columns":lik[:8],
     "source_license_exists":code_license.exists(),
     "rights":"No explicit source 2026 repository license found; source is publicly readable but unlicensed for copying/training by default, seek permission.",
     "task_candidate":"Score(ordinal human-rater gait category); keypoint trajectory -> expert score",
     "split_requirement":"By 98 source cow IDs, never by 272 clips. Score changes on repeated cows are acceptable if different sessions, subject to timestamp/source checks.",
     "status":"ACCESS_VERIFIED only; no model training or eval approval without usage rights and label annotation provenance."
    }
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=audit(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("COW_GAIT_POSE_AUDIT:"+json.dumps(result))
