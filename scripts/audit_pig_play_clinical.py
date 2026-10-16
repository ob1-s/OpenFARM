"""Audit real behavioral and independent clinical outcomes from Dryad 76hdr7t55.

Files are under CC0; read-only. Never train on experimental group as if it were
a measurement of positive emotional valence.
"""
from __future__ import annotations
import argparse,hashlib,io,json,time
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests
D="doi:10.5061/dryad.76hdr7t55"
S=requests.Session()
S.headers.update({"User-Agent":"OpenFARM-pig-play-clinical-source-audit/0.2"})
NEEDS=("Play__exploratory__active__inactive__feeding_behaviours",
       "Clinical_signs__rectal_temperature_and_medical_treatments",
       "Skin_lesion_scores","Lung_lesions")
def get(url,cap=1_000_000):
    last=None
    for i in range(3):
        try:
            with S.get(url,stream=True,timeout=(15,42)) as r:
                r.raise_for_status()
                buf=bytearray()
                for x in r.iter_content(65536):
                    buf+=x
                    if len(buf)>cap:raise RuntimeError("File size cap exceeded")
                return bytes(buf)
        except requests.exceptions.RequestException as e:
            last=e
            if i<2:time.sleep([2,5][i])
    raise RuntimeError(f"Source unavailable: {type(last).__name__}")

def read_json(url):
    return json.loads(get(url,cap=2_000_000))

def read_files():
    ds=read_json("https://datadryad.org/api/v2/datasets/"+quote(D,safe=":"))
    href=ds.get("_links",{}).get("stash:version",{}).get("href")
    if not href:raise RuntimeError("No Dryad version link")
    if href.startswith("/"):href="https://datadryad.org"+href
    ver=read_json(href)
    fhref=ver.get("_links",{}).get("stash:files",{}).get("href")
    if not fhref:raise RuntimeError("No Dryad file collection link")
    if fhref.startswith("/"):fhref="https://datadryad.org"+fhref
    j=read_json(fhref)
    return ds, j.get("_embedded",{}).get("stash:files",[])

def analysis():
    ds,fs=read_files()
    result={"doi":D,"license":ds.get("license"),"source_total_files":len(fs),
            "files":{},"caveat":"nine source litters, 28 challenged pigs; repeated observations not independent units."}
    for f in fs:
        name=f.get("path") or ""
        if not any(s in name for s in NEEDS):continue
        links=f.get("_links",{})
        raw=links.get("stash:download",{})
        official=raw.get("href") if isinstance(raw,dict) else raw
        if not official:
            raise RuntimeError("Dryad API file metadata missing official stash:download link")
        if official.startswith("/"):
            official="https://datadryad.org"+official
        if not official.startswith("https://datadryad.org/"):
            raise RuntimeError("Dryad file download URL is not an expected official HTTPS host")
        rec={"name":name,"size":f.get("size"),
             "official_download_url":official,
             "links":list(links)}
        # IMPORTANT: this endpoint is furnished by the dataset's public REST
        # API. Older audit guessed a file ID f.get('id') but the API's file
        # objects omit 'id', producing /None/download => false 403.
        candidates=[official]
        err=[]
        for url in candidates:
            try:
                data=get(url,cap=900_000)
                rec["downloaded_bytes"]=len(data)
                rec["sha256"]=hashlib.sha256(data).hexdigest()
                xl=pd.ExcelFile(io.BytesIO(data))
                rec["sheets"]={}
                for sn in xl.sheet_names:
                    dataf=pd.read_excel(io.BytesIO(data),sheet_name=sn)
                    cols=[str(c) for c in dataf.columns]
                    stat={"rows":len(dataf),"columns":cols[:70]}
                    for col in cols:
                        if any(k in col.lower() for k in ("pig","sow","lit","pen","play","clinical","injur","lesion","treat","score","resp","day","dpi")):
                            s=dataf[col]
                            stat.setdefault("important_fields",{})[col]={
                                "distinct":int(s.nunique(dropna=True)),
                                "missing":int(s.isna().sum()),
                                "values":{str(k):int(v) for k,v in s.value_counts().head(7).items()} if s.nunique(dropna=True)<=8 else None
                            }
                    rec["sheets"][sn]=stat
                break
            except Exception as e:err.append(f"{type(e).__name__}: {str(e)[:120]}")
        if "downloaded_bytes" not in rec:rec["download_errors"]=err
        result["files"][name]=rec
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:r=analysis()
    except Exception as e:r={"error":f"{type(e).__name__}: {str(e)[:200]}"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,default=str)+"\n")
    print("DRYAD_PIG_PLAY_CLINICAL:"+json.dumps(r,default=str))
    if "error" in r:raise SystemExit(2)
