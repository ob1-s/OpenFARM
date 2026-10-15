"""EquiFACS clinical interval alignment: expert FACS -> separate vet ratings.

Read-only source: Rashid et al PLOS ONE 2020 s003 XLSX. Do not redistribute the
original workbook or raw patient media. Creates a manifest and optional local
JSONL feature/target/provenance rows for research inspection only.

NO "Non-codeable" -> negative coercion. NO mixed-media train/test rows.
NO claim of inter-rater independence when FACS coder is also pain coder.
"""
from __future__ import annotations
import argparse,collections,datetime,hashlib,io,json,math,re
from pathlib import Path
import pandas as pd
import requests

URL="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0231608.s003&type=supplementary"
EXPECTED_SHA="f33fe32b8bd1283072df222038a36e6d9c34308adba7a174c6d6d47f960d2214"
MEDIA_RE=re.compile(r"^Media\s*(\d+)\b",re.I)
SEVERITY={"no pain":0,"mild pain":1,"moderate pain":2,"severe pain":3}
EXCLUDED_RATER="Alina"

def sec(v):
    if v is None or pd.isna(v):return None
    if isinstance(v,(int,float)):
        if 0<=v<=1:return float(v)*86400
        return float(v)
    s=str(v).strip()
    if not s:return None
    try:
        td=pd.to_timedelta(s)
        return float(td.total_seconds())
    except Exception:
        return None

def media_id(value):
    m=MEDIA_RE.match(str(value or "").strip())
    return int(m.group(1)) if m else None

def load():
    r=requests.get(URL,timeout=(20,55),headers={"User-Agent":"OpenFARM-EQUIfacs-audit/0.1"})
    r.raise_for_status()
    b=r.content
    if hashlib.sha256(b).hexdigest()!=EXPECTED_SHA:
        raise RuntimeError("Pinned clinical annotation file content changed")
    return pd.read_excel(io.BytesIO(b),sheet_name="FACS"),pd.read_excel(io.BytesIO(b),sheet_name="PAIN")

def build():
    facs,pain=load()
    # The first Excel header is 'Media 1 ' and additional media sections
    # appear within the first column; preserve their original grouping.
    current=media_id(facs.columns[0])
    events=[]
    diagnostics=collections.Counter()
    for row in facs.itertuples(index=False,name=None):
        maybe=media_id(row[0])
        if maybe is not None:
            current=maybe
            diagnostics["media_section_rows"]+=1
            continue
        if str(row[0]).strip().lower()=="tier":continue
        code=str(row[5]).strip() if len(row)>5 and pd.notna(row[5]) else ""
        if not code or code.lower()=="code":continue
        begin=sec(row[2]);end=sec(row[3])
        if current is None or begin is None or end is None or end<=begin:
            diagnostics["skipped_invalid_events"]+=1;continue
        events.append({"media":current,"from":begin,"to":end,"au":code})
    annotated=[]
    for row in pain.itertuples(index=False,name=None):
        source={str(k).strip():v for k,v in zip(pain.columns,row)}
        media=media_id(source.get("Media"))
        start=sec(source.get("Begin Time"))
        end=sec(source.get("End time"))
        level=str(source.get("Pain") or "").strip().lower()
        observer=str(source.get("Annotator") or "").strip()
        if media is None or start is None or end is None or end<=start:
            diagnostics["invalid_pain_interval"]+=1;continue
        annotated.append({"media":media,"from":start,"to":end,
                          "raw_level":level,"label":SEVERITY.get(level),
                          "rater":observer,"noncodeable":level=="non-codeable"})
    media_facs={x["media"] for x in events}
    media_pain={x["media"] for x in annotated}
    available=[]
    for row in annotated:
        matches=[ev for ev in events if ev["media"]==row["media"] and
                 ev["from"]<row["to"] and ev["to"]>row["from"]]
        overlap=collections.defaultdict(float)
        count=collections.Counter()
        for ev in matches:
            covered=min(ev["to"],row["to"])-max(ev["from"],row["from"])
            overlap[ev["au"]]+=covered
            count[ev["au"]]+=1
        d=row.copy()
        d["au_count"]=dict(count)
        d["au_time_seconds"]=dict(overlap)
        d["has_au_events"]=bool(matches)
        available.append(d)
    groups=collections.defaultdict(list)
    for x in available:groups[x["media"]].append(x)
    independent=[r for r in available if r["rater"].lower()!=EXCLUDED_RATER.lower()]
    nonmissing=[r for r in independent if r["label"] is not None]
    labels=collections.Counter(str(r["label"]) if r["label"] is not None else r["raw_level"] for r in available)
    by_rater={}
    for rat in sorted(set(r["rater"] for r in available)):
        rr=[r for r in available if r["rater"]==rat]
        by_rater[rat]={
            "rows":len(rr),"codeable":sum(x["label"] is not None for x in rr),
            "noncodeable":sum(x["noncodeable"] for x in rr),
            "class_counts":dict(collections.Counter(str(x["label"]) for x in rr if x["label"] is not None)),
            "media_count":len(set(x["media"] for x in rr)),
            "with_facs_events":sum(x["has_au_events"] for x in rr)
        }
    report={
        "source":URL,"source_sha256":EXPECTED_SHA,
        "source_facs_rows":len(facs),"source_pain_rows":len(pain),
        "facs_valid_interval_events":len(events),"facs_media_count":len(media_facs),
        "pain_annotation_intervals":len(annotated),
        "pain_media_count":len(media_pain),
        "media_id_overlap":len(media_facs & media_pain),
        "no_matched_facs_media":sorted(media_pain-media_facs),
        "all_pain_labels":dict(labels),
        "rater_stats":by_rater,
        "rater_independent_of_facs_coder_codeable_intervals":len(nonmissing),
        "rater_independent_media":len(set(x["media"] for x in nonmissing)),
        "rater_independent_labels":dict(collections.Counter(str(x["label"]) for x in nonmissing)),
        "aligned_intervals_with_facs_events":sum(x["has_au_events"] for x in available),
        "au_vocabulary":dict(collections.Counter(x["au"] for x in events).most_common(70)),
        "diag":dict(diagnostics),
        "caveats":[
            "FACS expert Alina also contributes pain labels: keep pain ratings from independently distinct Johan/Camilla as primary reference.",
            "A video/media ID is not a verified individual horse ID. Animal-heldout claim needs mapping in source manuscript/metadata.",
            "Pain ratings time intervals may differ in length/coverage from FACS events; no-event does not automatically mean no observed facial action when coding incomplete.",
            "Non-codeable means insufficient rating evidence, never class zero. With few severe/moderate intervals, 4-class calibration may be impossible.",
            "This is an expert-processed structured decision, not a raw video or cheap automatic perception model."
        ]
    }
    return report,available

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--local-rows",type=Path,default=None)
    a=p.parse_args()
    report,rows=build()
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n")
    if a.local_rows:
        a.local_rows.parent.mkdir(parents=True,exist_ok=True)
        with a.local_rows.open("w") as f:
            for x in rows:f.write(json.dumps(x)+"\n")
    print("EQUIFACS_STRUCTURED_AUDIT:"+json.dumps(report))
if __name__=="__main__":main()
