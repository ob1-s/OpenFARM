"""Verify the owner-disjoint CatMeows v2 training view at an immutable HF revision.

Only metadata are read; no animal recordings downloaded or redistributed.
"""
from __future__ import annotations
import json
from collections import Counter
from datasets import load_dataset

DATASET="oliveirabruno01/openfarm-catmeows"
REVISION="57fe220b22eae46eb555af9f4169bd4677584cea"

def read(split):
    ds=load_dataset(DATASET,split=split,revision=REVISION,streaming=True)
    ds=ds.select_columns(["owner_id","cat_id","context","audio_sha256"])
    return list(ds)

def audit(tr,te):
    owners={r["owner_id"] for r in te}
    if not owners or None in owners or "" in owners:
        raise AssertionError("Unusable test owner identifiers")
    if any(not r["owner_id"] for r in tr):raise AssertionError("Missing train owner")
    cleaned=[r for r in tr if r["owner_id"] not in owners]
    if not cleaned:raise AssertionError("Empty owner-heldout training split")
    keys=("owner_id","cat_id","audio_sha256")
    overlaps={k:len({r[k] for r in cleaned} & {r[k] for r in te}) for k in keys}
    if any(overlaps.values()):raise AssertionError(f"Owner/cat/audio leakage: {overlaps}")
    classes_train=set(r["context"] for r in cleaned)
    classes_test=set(r["context"] for r in te)
    if classes_train != classes_test:
        raise AssertionError(f"Class support mismatch: train={classes_train}, test={classes_test}")
    return {
        "original_train":len(tr),"repaired_train":len(cleaned),"test":len(te),
        "removed_train_rows":len(tr)-len(cleaned),
        "train_owners":len({r["owner_id"] for r in cleaned}),
        "test_owners":len(owners),
        "overlap":overlaps,
        "train_classes":dict(Counter(r["context"] for r in cleaned)),
        "test_classes":dict(Counter(r["context"] for r in te)),
    }

def main():
    report={}
    for pair in (("train","test"),("train_raw","test_raw")):
        report[f"{pair[0]}_vs_{pair[1]}"]=audit(read(pair[0]),read(pair[1]))
    report["source"]=DATASET
    report["revision"]=REVISION
    print("CAT_OWNER_REPAIR:"+json.dumps(report,sort_keys=True))

if __name__=="__main__":main()
