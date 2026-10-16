"""Audit genuinely public, author-labeled cattle movement dataset(s).

Aim: find valid full-data supervised decision tasks (accelerometer motion-index
-> independently annotated grazing/walking/rumination/rest). Source Figshare
28801181 DOI 10.6084/m9.figshare.28801181.v1, CC BY 4.0. About 160KB.

No model scores from source-exposed test labels. This is a per-source data
audit only, with fold-safe grouping and rights checks before task promotion.
Never store third-party source rows in repository/artifact.
"""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
from collections import Counter
import requests,pandas as pd
S=requests.Session()
S.headers["User-Agent"]="OpenFARM-public-animal-decision-audit/0.1"
ARTICLES=(28801181,25920463)
COLLECTIONS=(6465301,)
def get(url,limit=25_000_000):
    last=None
    for attempt in range(3):
        try:
            with S.get(url,timeout=(18,55),stream=True) as resp:
                resp.raise_for_status()
                v=bytearray()
                for b in resp.iter_content(65536):
                    v.extend(b)
                    if len(v)>limit:raise RuntimeError("Download exceeds bounded audit cap")
                return bytes(v)
        except requests.RequestException as e:
            last=e
            if attempt<2:time.sleep(2+attempt*3)
    raise RuntimeError(f"Failed GET ({url[:110]}): {last}")

def json_url(url):return json.loads(get(url,limit=2_000_000))
def summary(df,name):
    result={"rows":int(len(df)),"columns":[str(x) for x in df.columns],
            "exact_duplicate_rows":int(df.duplicated().sum()),
            "missing_by_column":{str(c):round(float(df[c].isna().mean()),4) for c in df.columns}}
    fields={}
    keyterms=("id","cow","anim","ranch","breed","site","loc","day","date","class","behavior","behaviour","activity","state","mi","motion","sec","time","score","label","graz","rum","walk","rest")
    for c in df:
        key=str(c);series=df[c]
        if any(w in key.lower() for w in keyterms):
            n=int(series.nunique(dropna=True))
            d={"distinct":n,"null":int(series.isna().sum()),"dtype":str(series.dtype)}
            if n<=25:d["counts"]={str(k):int(v) for k,v in series.value_counts(dropna=False).head(25).items()}
            elif pd.api.types.is_numeric_dtype(series):
                d["min"]=float(series.min(skipna=True));d["max"]=float(series.max(skipna=True))
            fields[key]=d
    result["relevant_fields"]=fields
    return result

def inspect_blob(blob,name):
    lower=name.lower()
    if lower.endswith((".csv",".tsv",".txt")):
        for sep in (None,",","\t",";"):
            try:
                df=pd.read_csv(io.BytesIO(blob),sep=sep,engine="python",encoding="utf-8-sig")
                if df.shape[1]>1:return {"table":summary(df,name)}
            except Exception:pass
        return {"error":"source CSV not parsed"}
    if lower.endswith((".xlsx",".xls")):
        xl=pd.ExcelFile(io.BytesIO(blob))
        return {"worksheets":{sheet:summary(pd.read_excel(io.BytesIO(blob),sheet_name=sheet),name)
                              for sheet in xl.sheet_names}}
    if lower.endswith(".zip"):
        z=zipfile.ZipFile(io.BytesIO(blob))
        files=[v for v in z.infolist() if not v.is_dir()]
        out={"archive_count":len(files),"sample_paths":[v.filename for v in files[:45]],"tables":{}}
        for v in files:
            if len(out["tables"])>=12:break
            if v.file_size>3_000_000:continue
            if v.filename.lower().endswith((".csv",".xlsx",".tsv",".txt")):
                try:out["tables"][v.filename]=inspect_blob(z.read(v),v.filename)
                except Exception as e:out["tables"][v.filename]={"error":str(e)[:160]}
        return out
    return {"file_format":"not table"}

def inspect_article(article_id):
    m=json_url(f"https://api.figshare.com/v2/articles/{article_id}")
    result={"id":article_id,"title":m.get("title"),"doi":m.get("doi"),
            "version":m.get("version"),"published":m.get("published_date"),
            "license":m.get("license"),"files":[]}
    for f in m.get("files",[]):
        item={"name":f.get("name"),"id":f.get("id"),"size":f.get("size"),
              "supplier_md5":f.get("supplied_md5")}
        if not f.get("download_url") or f.get("size",0)>20_000_000:
            item["status"]="large/unavailable";result["files"].append(item);continue
        try:
            blob=get(f["download_url"])
            item["status"]="actual_file_read"
            item["sha256"]=hashlib.sha256(blob).hexdigest()
            item["evidence"]=inspect_blob(blob,f["name"])
        except Exception as e:item["error"]=f"{type(e).__name__}: {e}"
        result["files"].append(item)
    return result

def inspect_collection(collection_id):
    m=json_url(f"https://api.figshare.com/v2/collections/{collection_id}")
    return {"id":collection_id,"title":m.get("title"),
            "doi":m.get("doi"),"license":m.get("license"),"article_count":m.get("articles_count"),
            "links":m.get("references"),"description_snippet":(m.get("description") or "")[:200]}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();out={"figshare_articles":{},"figshare_collections":{}}
    for k in ARTICLES:
        try:out["figshare_articles"][str(k)]=inspect_article(k)
        except Exception as e:out["figshare_articles"][str(k)]={"error":f"{type(e).__name__}: {e}"}
    for k in COLLECTIONS:
        try:out["figshare_collections"][str(k)]=inspect_collection(k)
        except Exception as e:out["figshare_collections"][str(k)]={"error":f"{type(e).__name__}: {e}"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,default=str)+"\n")
    print("OPEN_CATTLE_PUBLIC_AUDIT:"+json.dumps(out,default=str))
    if "error" in out["figshare_articles"]["28801181"]:raise SystemExit(2)

if __name__=="__main__":main()
