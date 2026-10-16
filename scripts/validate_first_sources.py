"""Evidence gating for two actual sources: HotPig minute sensor tables and EquiFACS.

Read-only. No model performance, no welfare-event invention, no republishing source media.
Usage: python scripts/validate_first_sources.py --output reports/deep-schema.json
"""
from __future__ import annotations
from collections import Counter
import argparse,io,json,hashlib,zipfile,time
from pathlib import Path
import requests
import pandas as pd

S=requests.Session()
S.headers["User-Agent"]="OpenFARM-research-audit/0.2 (source-data schema only)"

def fetch(url,cap=6_000_000):
    # Some Zenodo frontends terminate otherwise valid 2.6 MB downloads.
    # Bounded retries with an alternate API file route; never promote
    # success if source media were not actually downloaded.
    urls=[url]
    if url.startswith("https://zenodo.org/records/17090997/files/series.zip"):
        urls.append("https://zenodo.org/api/records/17090997/files/series.zip/content")
    last=None
    for attempt in range(4):
        target=urls[attempt%len(urls)]
        try:
            with S.get(target,timeout=(20,65),stream=True) as r:
                r.raise_for_status()
                buf=bytearray()
                for chunk in r.iter_content(65536):
                    buf+=chunk
                    if len(buf)>cap:raise RuntimeError("Source file exceeds cap")
                return bytes(buf)
        except (requests.exceptions.RequestException, ConnectionResetError) as e:
            last=e
            if attempt<3:time.sleep([2,5,8][attempt])
    raise RuntimeError(f"4 source download attempts exhausted: {type(last).__name__}")

def hotpig():
    z=fetch("https://zenodo.org/records/17090997/files/series.zip?download=1")
    arch=zipfile.ZipFile(io.BytesIO(z))
    paths=sorted(i.filename for i in arch.infolist() if i.filename.endswith(".csv"))
    result={
        "archive_size":len(z),
        "sha256":hashlib.sha256(z).hexdigest(),
        "pig_files":len(paths),
        "files":[],
        "condition_totals":Counter(),
        "state_totals":Counter(),
        "period_totals":Counter(),
        "all_time_series_rows":0
    }
    for path in paths:
        df=pd.read_csv(arch.open(path))
        required={"period","conditions","feed","state",
                  "unknown","standing","seating","lying"}
        if not required.issubset(df.columns):raise RuntimeError(f"Missing required columns: {path}")
        series={
            "file":path,
            "rows":len(df),
            "columns":list(df.columns),
            "periods":dict(df["period"].astype(str).value_counts()),
            "conditions":dict(df["conditions"].astype(str).value_counts()),
            "states":dict(df["state"].astype(str).value_counts().head(10)),
            "feed_non_null":int(df["feed"].notna().sum()),
            "feed_positive_count":int((pd.to_numeric(df["feed"],errors="coerce")>0).sum()),
            "feed_min":float(pd.to_numeric(df["feed"],errors="coerce").min()) if df["feed"].notna().any() else None,
            "feed_max":float(pd.to_numeric(df["feed"],errors="coerce").max()) if df["feed"].notna().any() else None,
            "missing_by_col":{c:round(float(df[c].isna().mean()),4) for c in df.columns}
        }
        result["files"].append(series)
        result["all_time_series_rows"]+=len(df)
        for col,key in (("period","period_totals"),("conditions","condition_totals"),("state","state_totals")):
            result[key].update(df[col].astype(str).values.tolist())
    for key in ("condition_totals","period_totals","state_totals"):
        result[key]=dict(result[key].most_common())
    result["warning"]="Posture and activity are model-generated 1-min detections; source images have separate manual YOLO annotation, not yet downloaded. Feeder series is an independent instrument only if metadata confirms its provenance."
    return result

def equifacs():
    root="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0231608."
    result={}
    for which in ("s002","s003"):
        b=fetch(root+which+"&type=supplementary",cap=500_000)
        x=pd.ExcelFile(io.BytesIO(b))
        info={"bytes":len(b),"sheets":{}}
        for sheet in x.sheet_names:
            df=pd.read_excel(io.BytesIO(b),sheet_name=sheet)
            data={"rows":int(len(df)),"columns":[str(c) for c in df.columns]}
            if sheet.lower()=="pain":
                for col in df.columns:
                    cl=str(col).lower()
                    if "pain" in cl or "condition" in cl:
                        data[str(col)+"_counts"]={str(k):int(v) for k,v in df[col].value_counts(dropna=False).head(12).items()}
                    if "horse id" in cl or cl=="media" or "video" in cl:
                        data[str(col)+"_unique"]=int(df[col].nunique(dropna=True))
                data["warning"]="Row count is time-segment annotations, not independent animals; count distinct horse/video."
            if sheet.lower()=="facs":
                data["visible_headings"]=[str(v)[:65] for v in df.head(3).values.flatten().tolist() if pd.notna(v)][:30]
                data["warning"]="FACS file is a wide/ELAN-like interval sheet; schema parsing required before counting events."
            info["sheets"][sheet]=data
        result[which]=info
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--only-hotpig",action="store_true",help="Avoid re-downloading EquiFACS during HotPig retry")
    a=p.parse_args()
    out={}
    tasks=(("HotPig",hotpig),) if a.only_hotpig else (("HotPig",hotpig),("EquiFACS",equifacs))
    for name,fn in tasks:
        try:out[name]=fn()
        except Exception as e:out[name]={"error":f"{type(e).__name__}: {e}"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,default=str)+"\n",encoding="utf-8")
    print("PROSPECT_DEEP_SCHEMA:"+json.dumps(out,default=str))
    if any("error" in v for v in out.values()):raise SystemExit(2)

if __name__=="__main__":main()
