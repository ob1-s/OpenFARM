"""OpenFARM v0.2 CANDIDATE — actual future physiologic outcomes from earlier observations.

Independent source: PigWeb 2026 (four institutes), CC BY 4.0.
This script prepares source-pinned research data. It does NOT treat hormone
change as objective animal emotion or label a need for caretaker attention.

Run: python scripts/build_pigweb_historical_v02.py --manifest reports/pigweb-historical.json
Optional locally materialized train/eval rows: --local-dir /tmp/pigweb-v02
Never republish source data automatically. All raw data stays external unless
user explicitly materializes local files with source credit and license.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,io,json,math,urllib.request
from collections import Counter,defaultdict
from pathlib import Path

REPO="AnjaEggert/PigWeb_cross-institutes"
REVISION="e3dba18edba6e1b214b4bea8ae66109d0f0d1226"
LICENSE="CC BY 4.0"
FILES=("PigWeb-behavior.csv","PigWeb-physiology.csv")
MISSING={"","NA","NaN","nan","N/A"}
SITE_ORDER=("AU","INRAE","IRTA","WUR")

def fetch_file(name):
    url=f"https://raw.githubusercontent.com/{REPO}/{REVISION}/data/{name}"
    with urllib.request.urlopen(urllib.request.Request(url,headers={
        "User-Agent":"OpenFARM-PigWeb-historical-research/0.1"}),timeout=50) as r:
        buf=r.read(1_000_001)
    if len(buf)>1_000_000:raise RuntimeError("Source file unexpectedly exceeds cap")
    rows=list(csv.DictReader(io.StringIO(buf.decode("utf-8-sig"))))
    return rows,hashlib.sha256(buf).hexdigest()

def key(r):
    return (r["institute"].strip(),r["pig_id_unique"].strip(),r["context"].strip())
def clock(r):
    return datetime.datetime.strptime(r["date"]+" "+r["time"],"%d.%m.%Y %H:%M")
def num(value):
    if value in MISSING:return None
    try:
        a=float(value)
        return a if math.isfinite(a) else None
    except (ValueError,TypeError):return None
def ratio(a,b):
    x,y=num(a),num(b)
    return None if x is None or y is None or y<=0 else x/y

def build():
    b,bhash=fetch_file(FILES[0]);p,phash=fetch_file(FILES[1])
    if len(b)!=256 or len(p)!=496:raise RuntimeError("Unexpected source row count")
    if any(r["sample"]!="pre" for r in b):raise RuntimeError("Behavior no longer represents pre samples")
    behavior={}
    for r in b:
        k=key(r)
        if k in behavior:raise RuntimeError("Duplicate behavior session key")
        behavior[k]=r
    physical=defaultdict(dict)
    for r in p:
        k=key(r);sample=r["sample"].strip()
        if sample not in ("pre","post") or sample in physical[k]:
            raise RuntimeError("Invalid/duplicate physiological sample key")
        physical[k][sample]=r
    if len(physical)!=248 or set(behavior)-set(physical)!=set(behavior)-set(physical):
        raise RuntimeError("Unexpected physiology grouping")
    examples=[]
    for k,readings in sorted(physical.items()):
        if set(readings)!={"pre","post"}:raise RuntimeError("Incomplete paired session")
        if k not in behavior:raise RuntimeError("Paired physiological sample missing pre-behavior row")
        pre,post=readings["pre"],readings["post"]
        brow=behavior[k]
        if clock(pre)!=clock(brow):raise RuntimeError("Pre physiology and human behavior timestamps differ")
        diff=(clock(post)-clock(pre)).total_seconds()/60
        if not 0<diff<=180:raise RuntimeError("Noncausal or abnormal post interval")
        features={
          "pre_cortisol_ng_ml":num(pre["cort_ngmL"]),
          "pre_oxytocin_pg_ml":num(pre["oxyt_pgmL"]),
          "observed_locomotor_play":num(brow["locomotor_play"]),
          "observed_escape_events":num(brow["escape_events"]),
          "observed_inactive_fraction":ratio(brow["scans_inactive"],brow["scans_activity_coded"]),
          "observed_tailcurled_fraction":ratio(brow["scans_tailcurled"],brow["scans_tailposture_coded"]),
          "observed_tailtucked_fraction":ratio(brow["scans_tailtucked"],brow["scans_tailposture_coded"]),
          "observed_earforward_fraction":ratio(brow["ear_forward"],brow["scans_ear_coded"]),
        }
        targets={}
        for prefix,column in (("cortisol","cort_ngmL"),("oxytocin","oxyt_pgmL"),
                              ("eye_IRT","eye_IRT_max")):
            a,c=num(pre[column]),num(post[column])
            targets[prefix+"_delta"]=None if a is None or c is None else c-a
            targets[prefix+"_increase"]=None if a is None or c is None else int(c>a)
        examples.append({"features":features,"targets":targets,
          "provenance":{"institute":k[0],"pig_id":k[1],"context":k[2],
              "pre":clock(pre).isoformat(),"post":clock(post).isoformat(),
              "delta_minutes":diff}})
    if len(examples)!=248:raise RuntimeError("Lost source pairs")
    out={"name":"PigWeb source-pinned future biomarker response candidates v0.2",
         "status":"PROPOSED_TASK_AUDITED_SOURCE_NOT_YET_APPROVED_BENCHMARK",
         "source":f"https://github.com/{REPO}/tree/{REVISION}",
         "license":LICENSE,"source_sha256":dict(zip(FILES,(bhash,phash))),
         "source_behavior_rows":len(b),"source_physiology_rows":len(p),
         "matched_pre_to_post_sessions":len(examples),
         "unique_subjects":len(set((e["provenance"]["institute"],e["provenance"]["pig_id"]) for e in examples)),
         "distinct_sites":len(set(e["provenance"]["institute"] for e in examples)),
         "model_input_columns":list(examples[0]["features"]),
         "explicitly_excluded_model_inputs":["pig_id","institute","context",
             "post hormone/temperature readings","future timestamps",
             "treatment/code-derived valence","followup elapsed minutes"],
         "question_family":["Noul(post cortisol > pre cortisol)",
            "Noul(post oxytocin > pre oxytocin)",
            "Value(post - pre cortisol)","Value(post - pre oxytocin)"],
         "split_protocol":"4 leave-one-institute-out heldout folds. Never mix sessions of one pig across train/test. Training / tuning must use remaining institutes; no test-based model selection.",
         "warnings":["Biomarker change is independently measured, not scientifically validated emotion or care action.",
           "4 institutes and treatment/site confounding; leave-one-institute-out metrics may vary substantially.",
           "124 pigs have 2 conditions; 248 pairs are not 248 independent pigs.",
           "Date/time schedule varies 21–127 min. Never claim fixed 30-min forecasting or reveal actual post-sample time to model.",
           "Derived continuous delta and binary increase are alternative scores on same underlying biomarker, not distinct ground truth sources.",
           "Need source protocol review, independent heldout model baselines and proper calibration before task admission."],
         "per_site":{}}
    for site in SITE_ORDER:
        rows=[e for e in examples if e["provenance"]["institute"]==site]
        groups=set(e["provenance"]["pig_id"] for e in rows)
        o={"sessions":len(rows),"pigs":len(groups),
           "post_delay_min":min(e["provenance"]["delta_minutes"] for e in rows),
           "post_delay_max":max(e["provenance"]["delta_minutes"] for e in rows)}
        for a in ("cortisol","oxytocin","eye_IRT"):
            values=[e["targets"][a+"_increase"] for e in rows]
            valid=[v for v in values if v is not None]
            o[a]={"paired":len(valid),"increase":sum(valid),
                  "not_increase":len(valid)-sum(valid)}
        out["per_site"][site]=o
    out["usable_outcomes"]={}
    for a in ("cortisol","oxytocin","eye_IRT"):
        c=[e["targets"][a+"_increase"] for e in examples]
        v=[i for i in c if i is not None]
        out["usable_outcomes"][a]={"n":len(v),"increase":sum(v),"not_increase":len(v)-sum(v)}
    out["post_delay_min"]=min(e["provenance"]["delta_minutes"] for e in examples)
    out["post_delay_max"]=max(e["provenance"]["delta_minutes"] for e in examples)
    return out,examples

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--local-dir",type=Path)
    a=p.parse_args()
    report,examples=build()
    a.manifest.parent.mkdir(parents=True,exist_ok=True)
    a.manifest.write_text(json.dumps(report,indent=2)+"\n")
    if a.local_dir:
        a.local_dir.mkdir(parents=True,exist_ok=True)
        for testsite in SITE_ORDER:
            d=a.local_dir/testsite
            d.mkdir(exist_ok=True)
            with (d/"train.jsonl").open("w") as tr,(d/"test.jsonl").open("w") as te:
                for e in examples:
                    rec={"features":e["features"],"targets":e["targets"]}
                    stream=te if e["provenance"]["institute"]==testsite else tr
                    stream.write(json.dumps(rec)+"\n")
    print("PIGWEB_HISTORICAL_SOURCE_AUDIT:"+json.dumps(report))

if __name__=="__main__":main()
