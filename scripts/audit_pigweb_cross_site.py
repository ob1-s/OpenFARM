"""Read real 2026 PigWeb four-institute affect-indicator data at a pinned Git SHA.

We never train on the context label as "true emotion". Test whether human-coded
play/escape observations join independently to physiological measures, and what
a genuine leave-one-institute-out split could support.

No private source rows redistributed; report aggregates + source hash only.
"""
import argparse,hashlib,io,json
from pathlib import Path
import pandas as pd
import requests

REPO="AnjaEggert/PigWeb_cross-institutes"
REV="e3dba18edba6e1b214b4bea8ae66109d0f0d1226"
FILES=("PigWeb-behavior.csv","PigWeb-physiology.csv","PigWeb-PCA.csv")
KEYS=("institute","pig_id_unique","context")

def source(name):
    url=f"https://raw.githubusercontent.com/{REPO}/{REV}/data/{name}"
    r=requests.get(url,timeout=(15,40),
                   headers={"User-Agent":"OpenFARM-PigWeb-research-integrity/0.1"})
    r.raise_for_status()
    data=r.content
    if len(data)>2_000_000:raise ValueError("unexpected source size")
    return pd.read_csv(io.BytesIO(data)),hashlib.sha256(data).hexdigest()

def audit():
    frames={};sha={}
    for name in FILES: frames[name],sha[name]=source(name)
    beh=frames[FILES[0]]
    phys=frames[FILES[1]]
    pca=frames[FILES[2]]
    for key in KEYS:
        if key not in beh or key not in phys:raise RuntimeError(f"Missing key {key}")
    for table,df in (("behavior",beh),("physiology",phys)):
        if df.duplicated(list(KEYS)+([] if table=="behavior" else ["sample"])).any():
            raise RuntimeError(f"{table} unexpectedly duplicates source animal/context/sample")
    if set(beh["context"].dropna())!={"positive","negative"}:
        raise RuntimeError("Source context schema shifted")
    if not set(phys["sample"].dropna())=={"pre","post"}:
        raise RuntimeError("Source physiology sample shifted")
    out={"source":f"https://github.com/{REPO}", "revision":REV,"license":"CC BY 4.0 per repository LICENSE",
         "file_sha256":sha,"rows":{k:len(v) for k,v in frames.items()},
         "behavior_label_origin":"source human-coded counts (verify with paper methods)",
         "warnings":[
            "Context is an experimental manipulation, NOT the animal's independently validated affect.",
            "Behavior data has no raw video in this repository; cannot authorize video-based perception evaluation.",
            "Repeated positive/negative observations are paired for the same pig and need subject+facility holdout.",
            "Physiology pre/post is not necessarily causally prior to observed behaviors; evaluate temporal ordering."
         ]}
    out["institution_site_counts"]={}
    for site in sorted(set(beh["institute"])):
        b=beh[beh.institute==site]
        p=phys[phys.institute==site]
        groups=b.groupby(["institute","pig_id_unique"],dropna=False)
        labels={}
        for context in ("positive","negative"):
            z=b[b.context==context]
            labels[context]={
              "sessions":len(z),
              "play_nonmissing":int(z.locomotor_play.notna().sum()),
              "play_positive":int((z.locomotor_play>0).sum()),
              "play_total_counts":float(z.locomotor_play.sum()),
              "escape_nonmissing":int(z.escape_events.notna().sum()),
              "escape_positive":int((z.escape_events>0).sum()),
              "escape_total_counts":float(z.escape_events.sum())
            }
        out["institution_site_counts"][site]={
          "behavior_rows":len(b),"behavior_animals":len(groups),
          "physiology_rows":len(p),"physiology_animals":int(p.pig_id_unique.nunique()),
          "behavior":labels
        }
    # Join only post physiology: one row per subject/context. No animal IDs
    # appear in final model input; here used only to verify matched provenance.
    post=phys.loc[phys["sample"]=="post"].copy()
    joined=beh.merge(post,on=list(KEYS),validate="one_to_one",
                      suffixes=("_behavior","_physiology"),how="left",indicator=True)
    out["join"]={
        "behavior_sessions":len(beh),
        "matched_with_post_physiology":int((joined["_merge"]=="both").sum()),
        "unmatched_behavior_sessions":int((joined["_merge"]!="both").sum()),
        "unmatched_animals_by_site":joined[joined._merge!="both"].groupby("institute")["pig_id_unique"].nunique().to_dict(),
        "physiology_candidate_features":[],
    }
    # These do not justify a clinical emotional ground truth; describe available
    # independent physiological measurement *only*.
    for col in ("cort_ngmL","oxyt_pgmL","eye_IRT_max","hr_min","change_CORT","change_OXT"):
        if col in joined.columns:
            out["join"]["physiology_candidate_features"].append({
                "field":col,
                "nonmissing":int(joined[col].notna().sum()),
                "source":col
            })
    out["approval"]="ACCESS_VERIFIED + TASK_SEMANTICS_CANDIDATE; NOT welfare-affect eval approved."
    return out

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--output",type=Path,required=True)
    x=a.parse_args()
    r=audit()
    x.output.parent.mkdir(parents=True,exist_ok=True)
    x.output.write_text(json.dumps(r,indent=2)+"\n")
    print("PIGWEB_SOURCE_AUDIT:"+json.dumps(r))
