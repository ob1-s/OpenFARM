"""Read-only real-file discovery for positive-welfare data.

Inspects:
- Figshare 27960810 "Dataset on pig social nosing" (CC0 on landing page)
- Dryad 10.5061/dryad.76hdr7t55 "Rearing pigs with play opportunities"

Never uploads / republishes source rows, participant IDs, source videos or raw files.
All samples below are coarse schema/dtype/cardinality and aggregate distributions.
Run: python scripts/audit_positive_welfare_sources.py --output out/positive-sources.json
Deps: pandas, requests, openpyxl
"""
from __future__ import annotations
import argparse,hashlib,io,json,re,time
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests

UA="OpenFARM-provenance-microaudit/0.2 (read-only academic source inspection)"
S=requests.Session()
S.headers.update({"User-Agent":UA})
FIGSHARE=27960810
DRYAD_DOI="doi:10.5061/dryad.76hdr7t55"
KEYS=("pig","sow","pen","litter","animal","id","behavior","behaviour","nosing",
      "interact","affil","groom","play","lesion","score","day","date","trial",
      "viral","clinical","treat","control","group","recip","initiat")

def get(url,max_bytes=4_000_000):
    last=None
    for attempt in range(3):
        try:
            with S.get(url,timeout=(17,40),stream=True) as r:
                r.raise_for_status()
                data=bytearray()
                for c in r.iter_content(65536):
                    data.extend(c)
                    if len(data)>max_bytes:raise RuntimeError("Exceeded conservative file cap")
                return bytes(data)
        except requests.exceptions.RequestException as e:
            last=e
            if attempt<2:time.sleep([2,5][attempt])
    raise RuntimeError(f"Source inaccessible after retries: {type(last).__name__}: {str(last)[:120]}")

def tabular(data,name):
    ext=name.lower().split(".")[-1]
    if ext in ("xls","xlsx"):
        x=pd.ExcelFile(io.BytesIO(data))
        return {sheet:frame_summary(pd.read_excel(io.BytesIO(data),sheet_name=sheet))
                for sheet in x.sheet_names}
    if ext in ("csv","txt","tsv"):
        for sep in (None,"\t",";",","):
            try:
                df=pd.read_csv(io.BytesIO(data),sep=sep,engine="python",
                               encoding="utf-8-sig")
                if df.shape[1]>1:return {"table":frame_summary(df)}
            except Exception:continue
        return {"error":"No parseable multicolumn delimited text"}
    return {"info":"Not an XLSX/CSV/TXT small table"}

def frame_summary(df):
    cols=[str(c) for c in df.columns]
    if len(df)>100000:
        return {"error":"Exceeded study schema row cap"}
    important={}
    for c in df.columns:
        name=str(c).lower()
        if not any(w in name for w in KEYS):continue
        series=df[c]
        distinct=int(series.nunique(dropna=True))
        stats={"distinct":distinct,"missing":int(series.isna().sum())}
        if distinct<=25:
            stats["counts"]={str(k):int(v) for k,v in series.value_counts(dropna=False).head(20).items()}
        important[str(c)]=stats
    return {"rows":int(len(df)),"columns":cols[:90],
            "important_cardinality":important,
            "all_empty_fields":[str(c) for c in df if df[c].isna().all()],
            "duplicate_full_rows":int(df.duplicated().sum())}

def figshare():
    meta=json.loads(get(f"https://api.figshare.com/v2/articles/{FIGSHARE}",max_bytes=1_000_000))
    record={"article_id":FIGSHARE,"doi":meta.get("doi"),
            "title":meta.get("title"),"license":meta.get("license"),
            "version":meta.get("version"),"published_date":meta.get("published_date"),
            "files":[]}
    for f in meta.get("files",[]):
        item={"name":f.get("name"),"size":f.get("size"),"is_link_only":f.get("is_link_only"),
              "supplier_checksum":f.get("supplied_md5")}
        if f.get("size",0)>3_000_000:
            item["status"]="size cap";record["files"].append(item);continue
        try:
            blob=get(f.get("download_url"),max_bytes=3_000_000)
            item["sha256"]=hashlib.sha256(blob).hexdigest()
            item["tables"]=tabular(blob,item["name"] or "")
            item["status"]="downloaded_and_parsed"
        except Exception as e:
            item["status"]="error";item["error"]=f"{type(e).__name__}: {str(e)[:220]}"
        record["files"].append(item)
    return record

def dryad():
    from urllib.parse import quote
    url="https://datadryad.org/api/v2/datasets/"+quote(DRYAD_DOI,safe=":")
    j=json.loads(get(url,max_bytes=1_000_000))
    r={"doi":DRYAD_DOI,"title":j.get("title"),"id":j.get("id"),
       "license":j.get("license"),"files":[]}
    # Dryad API returns files through versions/<id>/files pagination, not
    # dataset directly; inspect schema first and fail closed if unavailable.
    versions=j.get("_links",{}).get("stash:version",{})
    if isinstance(versions,dict):
        href=versions.get("href","")
        if href.startswith("/"):href="https://datadryad.org"+href
        if href:
            v=json.loads(get(href,max_bytes=1_000_000))
            r["version_keys"]=list(v.keys())
            link=v.get("_links",{}).get("stash:files",{}).get("href","")
            if link.startswith("/"):link="https://datadryad.org"+link
            if link:
                fs=json.loads(get(link,max_bytes=1_000_000))
                for f in fs.get("_embedded",{}).get("stash:files",[]):
                    r["files"].append({"name":f.get("path"),"size":f.get("size"),
                                       "download_link_found":bool(f.get("_links",{}))})
    return r

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    results={}
    for key,fn in (("figshare_pig_social_nosing",figshare),("dryad_pig_play",dryad)):
        try:results[key]=fn()
        except Exception as e:results[key]={"error":f"{type(e).__name__}: {str(e)[:250]}"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,default=str,ensure_ascii=False)+"\n")
    print("POSITIVE_SOURCE_AUDIT:"+json.dumps(results,default=str,ensure_ascii=False))
    if "error" in results["figshare_pig_social_nosing"]:raise SystemExit(2)

if __name__=="__main__":main()
