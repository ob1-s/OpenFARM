"""Diagnose whether a leakage-free ungulate split is structurally feasible.

Reads published RAW non-pig train/test plus pig-family holdout, at one immutable
Hugging Face revision. **Diagnostic only.** Does not create or certify new splits.

Run: python scripts/diagnose_ungulate_groups.py --output audit-reports/ungulate-components.json
Requires datasets, huggingface_hub. No model inference and no dataset mutations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

DATASET="oliveirabruno01/openfarm-ungulate-valence"
SPLITS=("train_raw","test_raw","pig_family_heldout")

def components(records):
    parent=list(range(len(records)))
    size=[1]*len(records)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def join(a,b):
        x,y=root(a),root(b)
        if x==y:return
        if size[x]<size[y]:x,y=y,x
        parent[y]=x;size[x]+=size[y]

    by_audio={}
    by_group={}
    for i,r in enumerate(records):
        a=r["audio_sha256"]
        g=r.get("source_group_id")
        if not a:raise ValueError(f"Sample {i} has empty audio SHA-256")
        if a in by_audio:join(i,by_audio[a])
        else:by_audio[a]=i
        if g:
            if g in by_group:join(i,by_group[g])
            else:by_group[g]=i

    grouped=defaultdict(list)
    for i in range(len(records)):
        grouped[root(i)].append(i)
    return list(grouped.values())

def diagnose(records):
    parts=components(records)
    raw=[r for r in records if r["split"]!="pig_family_heldout"]
    h={name:{r["audio_sha256"] for r in records if r["split"]==name}
       for name in SPLITS}
    split_overlaps={}
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            both=h[a]&h[b]
            split_overlaps[f"{a}_vs_{b}"]={
                "shared_audio_hashes":len(both),
                "sha_prefixes":sorted(x[:12] for x in both)[:10],
            }

    comp_stats=[]
    for group in parts:
        rows=[records[i] for i in group]
        count=Counter(row["split"] for row in rows)
        comp_stats.append({
            "rows":len(rows),
            "distinct_audio_hashes":len({r["audio_sha256"] for r in rows}),
            "distinct_nonempty_group_ids":len({r["source_group_id"] for r in rows
                                                if r.get("source_group_id")}),
            "source_splits":dict(count),
            "labels":dict(Counter(r["valence"] for r in rows))
        })
    comp_stats.sort(key=lambda x:x["rows"],reverse=True)
    return {
        "rows":len(records),
        "raw_rows":len(raw),
        "distinct_recording_hashes":len({r["audio_sha256"] for r in records}),
        "groups_with_ids":len({r["source_group_id"] for r in records if r.get("source_group_id")}),
        "rows_without_group_id":sum(not r.get("source_group_id") for r in records),
        "old_raw_split_audio_overlap":split_overlaps,
        "connected_components":len(parts),
        "components_crossing_old_raw_train_and_test":sum(
            all(x in c["source_splits"] for x in ("train_raw","test_raw")) for c in comp_stats
        ),
        "components_mixing_pig_and_nonpig":sum(
            "pig_family_heldout" in c["source_splits"]
            and any(x in c["source_splits"] for x in ("train_raw","test_raw"))
            for c in comp_stats
        ),
        "largest_components":comp_stats[:12],
        "largest_fraction_of_nonpig_raw_rows":round(
            max((c["rows"] for c in comp_stats
                 if "train_raw" in c["source_splits"] or "test_raw" in c["source_splits"]),default=0)
            /max(1,len(raw)),4),
        "caveat":"Connected components protect exact byte-duplicate and supplied group ID only; source metadata validity and near-duplicate clips unverified. Not a repaired split."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision=HfApi().dataset_info(DATASET).sha
    if not revision:raise RuntimeError("Cannot pin dataset revision")
    records=[]
    for split in SPLITS:
        ds=load_dataset(DATASET,split=split,revision=revision,streaming=True)
        for r in ds:
            records.append({
                "split":split,
                "audio_sha256":r.get("audio_sha256"),
                "source_group_id":r.get("source_group_id"),
                "valence":r.get("valence")
            })
    out=diagnose(records);out["dataset"]=DATASET;out["revision"]=revision
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print("UNGULATE CONNECTIVITY: "+json.dumps(out,sort_keys=True))

if __name__=="__main__":main()
