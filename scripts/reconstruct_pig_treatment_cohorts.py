"""Source-grounded pig pen cohorts and genuine first-treatment outcome.

Zenodo 5028016 / Ollagnier et al. 2023.
PenID is REUSED between cohorts; group by farm + pen + source TBSTART_ +
status and VALIDATED original date range. For controls TBSTART_ is a PSEUDO
comparison anchor, not an observed injury or treatment.

Dates Swedish: source date=YYYYMMDD and TBSTART_=M[ M ]DDYY or M[ M ]DDYYYY.
Dates Swiss: Excel serial. These never enter the model state as raw IDs/dates.
"""
from __future__ import annotations
import argparse,collections,datetime,hashlib,json
from pathlib import Path
import pandas as pd
from audit_pig_tail_damage_outcomes import download,parse

EPOCH=datetime.date(1899,12,30)
def parse_swedish_treatment(v):
    raw=str(int(v))
    # Swedish cohort year is 2004-07. Original Excel MM/DD/YYYY values lost
    # leading zero when converted numeric; two-digit years also occur.
    if raw.endswith(("2004","2005","2006","2007")):
        year=int(raw[-4:]);head=raw[:-4]
    elif raw.endswith(("04","05","06","07")):
        year=2000+int(raw[-2:]);head=raw[:-2]
    else:raise ValueError(f"Unknown Swedish year encoding: {raw}")
    if len(head)==3:mo=int(head[:1]);day=int(head[1:])
    elif len(head)==4:mo=int(head[:2]);day=int(head[2:])
    else:raise ValueError(f"Unknown Swedish month/day width: {raw}")
    return datetime.date(year,mo,day)
def as_date(site,v,kind):
    val=int(v)
    if site=="Swiss":
        return EPOCH+datetime.timedelta(days=val)
    if site!="Swedish":raise ValueError("Unexpected study site")
    if kind=="date":return datetime.datetime.strptime(str(val),"%Y%m%d").date()
    return parse_swedish_treatment(val)

def build():
    b,errors=download()
    d=parse(b)
    required={"Farm","PenID","PenType","TBSTART_","date","ID","DFV","DFC","StdFC"}
    if not required.issubset(d.columns):raise RuntimeError("Missing source cohort keys")
    d["calendar_date"]=[as_date(site,val,"date") for site,val in zip(d.Farm,d.date)]
    d["comparison_or_treatment_date"]=[as_date(site,val,"outcome") for site,val in zip(d.Farm,d["TBSTART_"])]
    key=["Farm","PenID","PenType","TBSTART_"]
    groups=[]
    n_duplicate_pig_date=0
    n_duplicate_cohorts_ignoring_status=0
    unique_keys=set()
    for ks,frame in d.groupby(key,dropna=False,sort=False):
        site,pen,typ,onset_value=ks
        onset=as_date(site,onset_value,"outcome")
        if not all(x==onset for x in frame["comparison_or_treatment_date"]):
            raise RuntimeError("Date encoding mismatch inside cohort")
        start,end=min(frame.calendar_date),max(frame.calendar_date)
        # A few source cohorts may have additional future rows: this does
        # not authorize using post-treatment data in prospective states.
        keystr="|".join(map(str,ks))
        if keystr in unique_keys:raise AssertionError("Duplicate cohort group")
        unique_keys.add(keystr)
        duplicate=int(frame.duplicated(["ID","calendar_date"]).sum())
        n_duplicate_pig_date+=duplicate
        coverage=len(set(frame.calendar_date))
        days_pre=len(set(x for x in frame.calendar_date if x<onset))
        days_post=len(set(x for x in frame.calendar_date if x>=onset))
        groups.append({
            "source_cohort_hash":hashlib.sha256(keystr.encode()).hexdigest()[:16],
            "site":str(site),"source_pen_id":str(pen),
            "event_type":str(typ),
            "rows":len(frame),"source_pigs":int(frame.ID.nunique()),
            "observed_distinct_days":coverage,"record_min":start.isoformat(),
            "record_max":end.isoformat(),"anchor":onset.isoformat(),
            "pre_anchor_observed_days":days_pre,"at_or_post_anchor_days":days_post,
            "duplicate_source_pig_dates":duplicate,
            "real_first_treatment":str(typ).upper()=="TB",
            "pen_days_with_at_least_8_observed_pigs_before_anchor":int(
                frame[frame.calendar_date<onset].groupby("calendar_date")["ID"].nunique().ge(8).sum())
        })
    by_type=dict(collections.Counter((g["site"],g["event_type"]) for g in groups))
    report={
        "origin":"https://zenodo.org/records/5028016",
        "original_sha256":hashlib.sha256(b).hexdigest(),"license":"CC BY-NC 4.0",
        "key_fields":key,
        "cohorts":len(groups),
        "real_documented_first_treatment_events":sum(g["real_first_treatment"] for g in groups),
        "pseudo_comparison_date_cohorts":sum(not g["real_first_treatment"] for g in groups),
        "by_site_type":{f"{site}|{typ}":count for (site,typ),count in by_type.items()},
        "repeated_pig_date_within_same_group":n_duplicate_pig_date,
        "first_treatment_outside_observed_coverage":sum(g["real_first_treatment"] and not (
            g["record_min"]<=g["anchor"]<=g["record_max"]) for g in groups),
        "minimum_pre_observed_days":min(g["pre_anchor_observed_days"] for g in groups if g["real_first_treatment"]),
        "cohort_summaries":groups,
        "warning":"First treatment is an actual documented welfare-related human action, NOT adjudicated optimal caretaker-alert label; controls have source synthetic anchors and are curated case-control units; time-window event rates are not population prevalences."
    }
    report["source_claim_comparison"]={
       "publication_pen_count":65,
       "publication_outbreak_count":27,
       "publication_site_outbreaks":{"Swedish":21,"Swiss":6},
       "source_original_pen_count":report["cohorts"],
       "source_original_outbreak_count":report["real_documented_first_treatment_events"],
       "unreconciled_label_disagreement":report["real_documented_first_treatment_events"]!=27
    }
    if report["cohorts"]!=65 or report["real_documented_first_treatment_events"]!=27:
        report["NOT_APPROVED"]="Published study says 27 outbreak pens (21 Swedish, 6 Swiss), but source composite pen/date/type groups show 28 (21 Swedish, 7 Swiss); preserve disputed record and investigate original metadata; do not silently drop a positive."
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=build()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    display={k:v for k,v in report.items() if k!="cohort_summaries"}
    display["tail_biting_source_cohorts"]=[
        {k:g[k] for k in ("site","source_pen_id","source_cohort_hash",
            "record_min","record_max","anchor","pre_anchor_observed_days",
            "at_or_post_anchor_days","rows","source_pigs",
            "duplicate_source_pig_dates")}
        for g in report["cohort_summaries"] if g["real_first_treatment"]]
    print("TAIL_DAMAGE_COHORT_RECON:"+json.dumps(display))
