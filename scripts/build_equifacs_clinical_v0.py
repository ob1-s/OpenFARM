"""Produce research-only FACS->independent vet pain binary task manifest.

Input: original, time-coded EquiFACS AUs from clinical horse videos.
Target: Johan/Camilla codeable pain rating ('any pain' vs 'no pain').
*Alina* FACS coder's own pain ratings excluded to avoid rater circularity.
Result: subject/media-disjoint train/calibration/test, not independent farms.
No raw video required for structured task, but no raw perception claim either.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,random
from pathlib import Path
from prepare_equifacs_structured import build

REQUIRED={"train":12,"calibration":4,"test":5}
AU_EXCLUDE={"non-scoreable","uncodable"}
def produce():
    info,rows=build()
    candidates=[r for r in rows
                if r["rater"].lower() in {"johan","camilla"} and r["label"] is not None]
    if len(candidates)!=64 or len(set(r["media"] for r in candidates))!=21:
        raise RuntimeError("Independent clinical annotation cohort shifted")
    # Expert FACS source has active "Non-scoreable"/"Uncodable" codes; mask
    # affected intervals rather than treating unscorable facial observations
    # as complete negative features.
    contaminated=[r for r in candidates if
                  any(name.lower() in AU_EXCLUDE for name in r["au_count"])]
    accepted=[r for r in candidates if r not in contaminated]
    media=sorted(set(r["media"] for r in accepted))
    if len(media)<18:raise RuntimeError("Too many subjects lost to quality gate")
    groups=collections.defaultdict(list)
    for r in accepted:groups[r["media"]].append(r)
    # Stratification depends only on source labels, never model accuracy. One
    # immutable split allocation is then frozen. We don't select favorable
    # heldout animals based on model performance.
    quotas={"train":round(12*len(media)/21),
            "calibration":round(4*len(media)/21)}
    quotas["test"]=len(media)-quotas["train"]-quotas["calibration"]
    assignment=None
    for seed in range(300):
        shuffled=media[:]
        random.Random(seed).shuffle(shuffled)
        a=0
        splits={}
        for name in ("train","calibration","test"):
            b=a+quotas[name]
            splits[name]=shuffled[a:b];a=b
        classes={name:{int(r["label"]>0) for mid in ids for r in groups[mid]}
                 for name,ids in splits.items()}
        if all(classes[k]=={0,1} for k in classes):
            assignment=splits;chosen_seed=seed;break
    if assignment is None:
        raise RuntimeError("Cannot construct a stratified heldout split with both outcome classes")
    codebook=sorted({code for r in accepted for code in r["au_count"]
                     if code.lower() not in AU_EXCLUDE})
    result={
        "question":"Noul(independent veterinarian rated pain present during FACS-annotated interval)",
        "source":"Rashid et al. 2020 DOI 10.1371/journal.pone.0231608, s003",
        "sha256":info["source_sha256"],
        "source_media_count":info["pain_media_count"],
        "source_interval_rows":info["pain_annotation_intervals"],
        "rater_independence":"Alina FACS coder's pain ratings excluded; Johan and Camilla only",
        "codeable_independent_rows":len(candidates),
        "non_codeable_original_rows":info["all_pain_labels"].get("non-codeable",0),
        "quality_excluded_intervals":len(contaminated),
        "accepted_intervals":len(accepted),
        "input_feature_semantics":"FACS expert AU event counts + AU temporal occupancy / vet-rated interval seconds; no media/rater/context",
        "feature_vocabulary":codebook,
        "split_seed":chosen_seed,
        "splits":{},
        "limits":[
            "No raw image/video inference is trained/evaluated; expert AU extraction is already supplied.",
            "Media IDs are taken as independent source videos; verify 21-media = 21-horse assertion from original clinical source before external transfer claims.",
            "One clinical video cohort, sparse positive events, no calibrated four-class Score.",
            "Clinical interval overlaps within source animal are not independent events; infer uncertainty clustered by media.",
            "This tests matching separate human expert observations, not an independently diagnosed veterinary condition or treatment need."
        ]
    }
    emitted={}
    for name,ids in assignment.items():
        subset=[r for mid in ids for r in groups[mid]]
        features=[]
        provenance=[]
        for r in subset:
            duration=r["to"]-r["from"]
            if duration<=0:raise RuntimeError("Nonpositive clinical segment")
            ft={f"AU:{code}":min(float(r["au_time_seconds"].get(code,0))/duration,1.0)
                for code in codebook}
            ft.update({f"COUNT:{code}":int(r["au_count"].get(code,0)) for code in codebook})
            # Media/rater IDs deliberately kept in independent provenance.
            features.append({"features":ft,"label":int(r["label"]>0)})
            provenance.append({"media":r["media"],"rater":r["rater"],
                               "start_seconds":r["from"],"end_seconds":r["to"],
                               "original_ordinal_label":r["label"]})
        counts=collections.Counter(f["label"] for f in features)
        result["splits"][name]={
            "media_ids":sorted(ids),
            "animals_assumed_from_media":len(ids),
            "codeable_rows":len(features),
            "no_pain":counts[0],"pain_associated":counts[1]
        }
        emitted[name]=(features,provenance)
    all_groups=[mid for ids in assignment.values() for mid in ids]
    if len(set(all_groups))!=len(all_groups):
        raise AssertionError("Media identity overlaps between dataset partitions")
    return result,emitted

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--local-dir",type=Path,default=None)
    a=p.parse_args()
    result,emitted=produce()
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n")
    if a.local_dir:
        a.local_dir.mkdir(parents=True,exist_ok=True)
        for name,(features,provenance) in emitted.items():
            for suffix,records in (("decision",features),("provenance",provenance)):
                f=a.local_dir/f"{name}.{suffix}.jsonl"
                f.write_text("".join(json.dumps(x)+"\n" for x in records))
    print("EQUIFACS_BOUNDED_TASK:"+json.dumps(result))

if __name__=="__main__":main()
