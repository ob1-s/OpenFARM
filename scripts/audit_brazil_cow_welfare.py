"""OpenFARM Welfare Evidence Sprint: actual public Brazilian cow clinical labels.

Source: Bran et al., Zenodo 3408818. READ-ONLY, public original files.
Goal: audit genuine clinical gait and body condition score labels, link to
independent non-target cow/herd measurements with farm/group holdout.

Output is aggregate schema, label counts, leakage flags and candidate joins,
not original identifying source rows or target labels for redistribution.
"""
from __future__ import annotations
import argparse, collections, hashlib, io, json, time
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests

RECORD=3408818
TABLES=["BCS and gait score all cows.csv","Cowlevel.csv","Herd.csv","Penlevel.csv"]
KEYWORD=("farm","herd","id","cow","gait","lame","locomot","score","bcs","body",
         "parity","milk","yield","dim","lact","breed","health","mast","hoof",
         "hock","lesion","clean","housing","type","density","floor","pen",
         "day","treatment","age","sole","injur","feet","weight")
S=requests.Session()
S.headers["User-Agent"]="OpenFARM-Welfare-Sprint/0.1 (read-only academic source audit)"
def fetch(url,cap=3_000_000):
    last=None
    variants=[url]
    for trial in range(5):
        try:
            with S.get(variants[0],timeout=(18,70),stream=True) as r:
                r.raise_for_status()
                b=bytearray()
                for chunk in r.iter_content(65536):
                    b.extend(chunk)
                    if len(b)>cap:raise RuntimeError(f"Source over download cap: {url}")
                return bytes(b)
        except requests.exceptions.RequestException as e:
            last=e
            if trial<4:time.sleep([1,3,6,10][trial])
    raise RuntimeError(f"Official source download failed after 5 attempts: {type(last).__name__}: {str(last)[:120]}")

def ingest(fname):
    variants=[
      f"https://zenodo.org/records/{RECORD}/files/{quote(fname)}?download=1",
      f"https://zenodo.org/api/records/{RECORD}/files/{quote(fname)}/content"
    ]
    errors=[]
    for url in variants:
        try:
            data=fetch(url)
            if data[:1] in (b"<",b"{"):raise RuntimeError("Expected CSV, got HTML/JSON")
            df=pd.read_csv(io.BytesIO(data),encoding="utf-8-sig",low_memory=False)
            return df,hashlib.sha256(data).hexdigest(),[]
        except Exception as e:errors.append(f"{type(e).__name__}: {str(e)[:180]}")
    return None,None,errors

def describe(df):
    out={"rows":len(df),"columns":[str(c) for c in df.columns],
         "exact_full_row_duplicates":int(df.duplicated().sum()),
         "missing_fraction_top":{},
         "candidate_group_keys":{},
         "candidate_target_and_input_fields":{}}
    for col in df:
        v=df[col]
        l=str(col).lower()
        if v.isna().mean()>.01:out["missing_fraction_top"][str(col)]=round(float(v.isna().mean()),5)
        if any(term in l for term in KEYWORD):
            distinct=int(v.nunique(dropna=True))
            item={"unique":distinct,"nulls":int(v.isna().sum()),
                  "type":str(v.dtype)}
            if distinct<=14:
                item["distribution"]={str(k):int(n) for k,n in v.value_counts(dropna=False).items()}
            elif pd.api.types.is_numeric_dtype(v):
                nn=pd.to_numeric(v,errors="coerce").dropna()
                if len(nn):item["range"]=[round(float(nn.min()),4),round(float(nn.max()),4)]
            out["candidate_target_and_input_fields"][str(col)]=item
            if any(x in l for x in ("farm","herd","id","cow")):
                out["candidate_group_keys"][str(col)]=distinct
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    report={"source":"https://zenodo.org/records/3408818","study":"Bran et al. Southern Brazil dairy lameness, 50 farms, 13716 cows",
            "table_audit":{},"possible_joins":[],
            "approval":"DATA FILES UNDER AUDIT; not automatically task validated, no train/eval approved"}
    frames={}
    for fname in TABLES:
        df,sha,errors=ingest(fname)
        if df is None:report["table_audit"][fname]={"error":errors};continue
        frames[fname]=df
        report["table_audit"][fname]={"sha256":sha,**describe(df)}
    for a,da in frames.items():
        for b,db in frames.items():
            if a>=b:continue
            pairs=[]
            for col in set(da.columns)&set(db.columns):
                if not any(kw in str(col).lower() for kw in ("farm","herd","cow","id","pen")):continue
                if da[col].isna().all() or db[col].isna().all():continue
                left=set(str(x) for x in da[col].dropna().unique())
                right=set(str(x) for x in db[col].dropna().unique())
                pairs.append({"column":str(col),"shared_unique":len(left&right),
                              "left_unique":len(left),"right_unique":len(right),
                              "left_unique_rows":bool(da[col].dropna().is_unique),
                              "right_unique_rows":bool(db[col].dropna().is_unique)})
            report["possible_joins"].append({"left":a,"right":b,"same_column_candidates":pairs})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("BRAZIL_COW_WELFARE_AUDIT:"+json.dumps(report,ensure_ascii=False))
    if not frames:raise SystemExit(2)
if __name__=="__main__":main()
