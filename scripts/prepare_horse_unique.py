"""Prepare reproducibly deduplicated horse regional-score data (source unchanged).

The published train/train_raw contain 116/1035 exact duplicate images.
The published test/test_raw have no duplicate image bytes. This script checks
identity/score consistency, retains first occurrence, and optionally saves a
local DatasetDict for training. No HF writing; no diagnosis labels invented.

Usage:
  python scripts/prepare_horse_unique.py --manifest horse-v2.json
  python scripts/prepare_horse_unique.py --manifest horse-v2.json --output-dir horse-v2
Requires datasets/huggingface_hub. Rights must be verified before redistribution.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from datasets import DatasetDict,load_dataset,Image

SOURCE="oliveirabruno01/openfarm-horse-grimace-region"
REVISION="ba5799d2c29886e238f8a069fb3501d1d91444f5"
SPLITS=("train","test","train_raw","test_raw")

def clean(ds,split):
    if not {"image","source_sha256","face_region","pain_score","horse_id"}.issubset(ds.column_names):
        raise RuntimeError(f"Required scientific source columns missing in {split}")
    ds=ds.cast_column("image",Image(decode=False))
    seen={}
    kept=[]
    members=[]
    horses=set()
    for i,row in enumerate(ds):
        blob=row["image"].get("bytes")
        if not isinstance(blob,bytes):
            raise RuntimeError(f"{split} row {i}: no image bytes")
        digest=hashlib.sha256(blob).hexdigest()
        if digest != row["source_sha256"]:
            raise RuntimeError(f"{split} row {i}: published and actual image SHA mismatch")
        region=str(row["face_region"])
        score=str(row["pain_score"])
        horse=str(row["horse_id"])
        if not region or not horse or score not in ("0","1","2"):
            raise RuntimeError(f"{split} row {i}: missing horse/region/score")
        key=(digest,region)
        if key in seen:
            previous=seen[key]
            if (score,horse)!=(previous["score"],previous["horse"]):
                raise RuntimeError(f"{split} row {i}: conflicting score/horse for same image+region")
            continue
        seen[key]={"score":score,"horse":horse}
        kept.append(i)
        members.append(digest+":"+region+":"+score+":"+horse)
        horses.add(horse)
    manifest={
      "original_rows":len(ds),
      "unique_media_region_rows":len(kept),
      "discarded_identical_rows":len(ds)-len(kept),
      "horse_count":len(horses),
      "class_counts":{level:sum(v["score"]==level for v in seen.values()) for level in ("0","1","2")},
      "ordered_member_digest":hashlib.sha256("\n".join(members).encode()).hexdigest()
    }
    return ds.select(kept), manifest, set(k[0] for k in seen), horses

def prepare(save_path=None):
    result={}
    out={}
    media={}
    groups={}
    for split in SPLITS:
        ds=load_dataset(SOURCE,split=split,revision=REVISION)
        clean_ds,manifest,identities,horses=clean(ds,split)
        out[split]=clean_ds
        result[split]=manifest
        media[split]=identities
        groups[split]=horses
    for a,b in (("train","test"),("train_raw","test_raw"),
                ("train","test_raw"),("train_raw","test")):
        if media[a]&media[b] or groups[a]&groups[b]:
            raise RuntimeError(f"Train/test media or horse identity leak: {a}/{b}")
    for small,big in (("train","train_raw"),("test","test_raw")):
        if not media[small].issubset(media[big]):
            raise RuntimeError(f"Balanced media view not subset of raw source: {small}/{big}")
    if save_path:
        DatasetDict(out).save_to_disk(str(save_path))
    return {"dataset":SOURCE,"source_revision":REVISION,
            "dedup_strategy":"identical image bytes + face region, with score/horse consistency",
            "audit":result,
            "locally_saved":bool(save_path),
            "notice":"Ordinal region cue only, not a clinical pain diagnosis. Animal count still small."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path)
    args=p.parse_args()
    data=prepare(args.output_dir)
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(data,indent=2)+"\n")
    print("HORSE DEDUP REPAIR:"+json.dumps(data))
if __name__=="__main__":main()
