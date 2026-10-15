"""Forensic check of Horse Grimace repeats at the *actual image byte* level.

Source: pinned immutable HF revision. Checks whether repeated published parent
hashes represent identical media/face-region/ordinal labels. Does not mutate HF.
"""
import hashlib,json
from collections import Counter,defaultdict
from datasets import load_dataset,Image
SOURCE="oliveirabruno01/openfarm-horse-grimace-region"
REV="ba5799d2c29886e238f8a069fb3501d1d91444f5"

def read(split):
    ds=load_dataset(SOURCE,split=split,revision=REV,streaming=True)
    ds=ds.cast_column("image",Image(decode=False))
    ans=[]
    for r in ds:
        blob=r.get("image",{}).get("bytes")
        if not isinstance(blob,bytes):raise RuntimeError(f"Missing image bytes in {split}")
        ans.append({"byte_hash":hashlib.sha256(blob).hexdigest(),
         "source_sha256":r.get("source_sha256"),"horse":r.get("horse_id"),
         "region":r.get("face_region"),"score":str(r.get("pain_score")),
         "file_id":r.get("source_file_id")})
    return ans

def report(rows):
    seen=defaultdict(list)
    for row in rows:
        seen[(row["byte_hash"],row["region"])].append(row)
    label_conflicts=sum(len({r["score"] for r in group})>1 for group in seen.values())
    source_conflicts=sum(len({r["source_sha256"] for r in group})>1 for group in seen.values())
    src=defaultdict(list)
    for row in rows: src[row["source_sha256"]].append(row)
    multi_region=sum(len({r["region"] for r in group})>1 for group in src.values())
    multi_score=sum(len({r["score"] for r in group})>1 for group in src.values())
    return {
        "rows":len(rows),"unique_bytes":len({r["byte_hash"] for r in rows}),
        "unique_media_region_pairs":len(seen),
        "redundant_media_region_rows":len(rows)-len(seen),
        "conflicting_score_for_same_media_region":label_conflicts,
        "multiple_source_hashes_same_bytes":source_conflicts,
        "source_groups_with_multiple_regions":multi_region,
        "source_groups_with_multiple_scores":multi_score,
        "published_source_sha_matches_image_bytes":sum(
            r["source_sha256"]==r["byte_hash"] for r in rows)
    }

def main():
    data={k:read(k) for k in ("train","test","train_raw","test_raw")}
    out={k:report(v) for k,v in data.items()}
    for a,b in (("train","test"),("train_raw","test_raw"),
                ("train","test_raw"),("train_raw","test")):
        out[f"{a}_vs_{b}_common_image_bytes"]=len(
            {r["byte_hash"] for r in data[a]}&
            {r["byte_hash"] for r in data[b]})
    out["source"]=SOURCE
    out["revision"]=REV
    print("HORSE MEDIA DUPLICATION AUDIT:"+json.dumps(out,sort_keys=True))
    if any(out[k]["conflicting_score_for_same_media_region"] for k in
           ("train","test","train_raw","test_raw")):
        raise SystemExit("Conflicting ordinal labels on exact same horse media+region")
    if any(out[k] for k in out if k.endswith("_common_image_bytes")):
        raise SystemExit("Cross split exact horse media leakage detected")

if __name__=="__main__":main()
