"""Inspect whether EquiFACS clinical FACS events align with separate vet pain ratings.

Source: Rashid et al PLOS ONE 2020 S2 Data xlsx. Strictly source-inspection;
no model scores or representation of non-codeable labels as "no pain".
"""
from __future__ import annotations
import argparse,hashlib,io,json
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

BASE="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0231608."
S=requests.Session()
S.headers.update({"User-Agent":"OpenFARM-equiFACS-expert-alignment/0.1"})
def download(code):
    r=S.get(BASE+code+"&type=supplementary",timeout=(20,50))
    r.raise_for_status()
    b=r.content
    if len(b)>250_000:raise RuntimeError("unexpected huge Excel")
    return b

def clean(v):
    if pd.isna(v):return ""
    return str(v).strip()

def preview(df,n=14):
    return [[clean(x)[:90] for x in row] for row in df.iloc[:n,:min(8,len(df.columns))].itertuples(index=False,name=None)]

def clinical(data):
    facs=pd.read_excel(io.BytesIO(data),sheet_name="FACS")
    pain=pd.read_excel(io.BytesIO(data),sheet_name="PAIN")
    if not {"Media","Pain"}.issubset(pain.columns):raise RuntimeError("Unexpected vet annotation schema")
    labels=pain["Pain"].map(clean)
    media=pain["Media"].map(clean)
    out={
        "facs_rows":len(facs),"pain_rows":len(pain),
        "facs_column_names":[str(x) for x in facs.columns],
        "facs_first_rows":preview(facs,n=16),
        "pain_first_rows":preview(pain,n=12),
        "facs_media_column_names":[str(v) for v in facs.iloc[:,0].dropna().unique()][:40],
        "facs_first_column_frequency":dict(facs.iloc[:,0].map(clean).value_counts().head(24)),
        "pain_media_distribution":dict(media.value_counts()),
        "pain_labels_original":dict(pain["Pain"].map(str).value_counts()),
        "pain_labels_trimmed":dict(labels.value_counts()),
        "pain_annotator_counts":dict(pain.iloc[:,-1].map(clean).value_counts()),
        "missing_media_ids":int((media=="").sum()),
        "pain_annotator_median_by_media":float(pain.groupby("Media")["Annotator "].nunique().median()) if "Annotator " in pain else None,
    }
    return out

def experimental(data):
    xl=pd.ExcelFile(io.BytesIO(data))
    pain=pd.read_excel(io.BytesIO(data),sheet_name="Pain")
    facs=pd.read_excel(io.BytesIO(data),sheet_name="FACS")
    return {"video_rows":len(pain),"horses":int(pain["Horse ID"].nunique()),
            "pain_labels":dict(pain["Pain/no pain"].astype(str).value_counts()),
            "facs_rows":len(facs),"facs_columns":len(facs.columns),
            "facs_early_rows":preview(facs,n=11)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out={}
    for name,code,fn in (("clinical","s003",clinical),("experimental","s002",experimental)):
        try:
            data=download(code)
            out[name]={"source_sha256":hashlib.sha256(data).hexdigest(),**fn(data)}
        except Exception as e:out[name]={"error":f"{type(e).__name__}: {str(e)[:240]}"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,default=str,ensure_ascii=False)+"\n")
    print("EQUIFACS_ALIGNMENT:"+json.dumps(out,default=str,ensure_ascii=False))
    if "error" in out["clinical"]:raise SystemExit(2)

if __name__=="__main__":main()
