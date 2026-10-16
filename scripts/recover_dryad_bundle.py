"""Try Dryad's DOCUMENTED dataset-wide download (not guessed file IDs).

Source: https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.76hdr7t55/download
Documentation: datadryad/dryad-app/documentation/apis/README.md

No auth/cookie bypass, no mirror from unauthorized sources. Inspect intended
behavior and clinical XLSX with coarse schema, hashes and origin only. Never
publish original data rows or raw ZIP.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests

DOI="10.5061/dryad.76hdr7t55"
PREFIX="https://datadryad.org/api/v2/datasets/"
S=requests.Session()
S.headers.update({"User-Agent":"OpenFARM-independent-welfare-data-audit/0.3"})

def get(url,limit=30_000_000):
    with S.get(url,timeout=(20,85),stream=True) as r:
        if r.status_code>=400:
            return None,{"http_status":r.status_code,"response_content_type":r.headers.get("Content-Type"),
                         "error":"Official public API declined this request; no authentication bypass attempted"}
        data=bytearray()
        for chunk in r.iter_content(65536):
            data.extend(chunk)
            if len(data)>limit:raise RuntimeError("Download exceeds archive cap")
        return bytes(data),{"http_status":r.status_code,"url":r.url,"mime":r.headers.get("Content-Type"),
                             "sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}

def summary_excel(b):
    x=pd.ExcelFile(io.BytesIO(b))
    sheets={}
    for sheet in x.sheet_names:
        df=pd.read_excel(io.BytesIO(b),sheet_name=sheet,header=0)
        cols=[str(c) for c in df.columns]
        keys=["pig","litter","sow","pen","play","behav","day","dpi","clin",
              "resp","lesion","score","treat","temp"]
        d={"rows":len(df),"columns":cols[:70],"key_summary":{}}
        for col in df.columns:
            if not any(k in str(col).lower() for k in keys):continue
            vals=df[col]
            d["key_summary"][str(col)]={
                "distinct":int(vals.nunique(dropna=True)),
                "missing":int(vals.isna().sum()),
                "small_value_counts":{str(k):int(v) for k,v in vals.value_counts().head(9).items()}
                   if vals.nunique(dropna=True)<=12 else None}
        sheets[sheet]=d
    return sheets

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    url=PREFIX+quote("doi:"+DOI,safe="")+"/download"
    report={"dataset_doi":DOI,"documented_download_url":url}
    raw,meta=get(url)
    report["bundle"]=meta
    if raw is not None:
        try:
            ar=zipfile.ZipFile(io.BytesIO(raw))
            report["entries_count"]=len(ar.infolist())
            report["entries"]={}
            for m in ar.infolist():
                if m.is_dir():continue
                name=m.filename
                item={"size":m.file_size}
                if name.lower().endswith(".xlsx") and m.file_size<=950_000:
                    try:
                        content=ar.read(name)
                        item["sha256"]=hashlib.sha256(content).hexdigest()
                        if any(word in name.lower() for word in ("play__","clinical_signs","lung_lesion","skin_lesion")):
                            item["sheets"]=summary_excel(content)
                    except Exception as e:item["inspect_error"]=f"{type(e).__name__}: {str(e)[:140]}"
                report["entries"][name]=item
        except Exception as e:
            report["bundle_parse_error"]=f"{type(e).__name__}: {str(e)[:230]}"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print("DRYAD_BUNDLE_RECON:"+json.dumps(report,default=str))

if __name__=="__main__":main()
