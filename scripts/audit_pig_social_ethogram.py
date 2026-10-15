"""Inspect real expert ethogram observations + lesion sheets from Zenodo 5920843.

Read-only, no original data/animal-level rows redistributed. Outcomes and
follow-up temporality must not be inferred from filename alone.
"""
import argparse,hashlib,io,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests

RECORD=5920843
FILES=("animal.txt","activity.txt","confrontations.txt","conspecifics.txt",
       "lesions.txt","objects.txt","rooting.txt")
NEEDLES=("pig","animal","id","day","date","time","lesion","injur","skin",
         "tail","behav","object","interact","group","context","score",
         "root","activity","conspecific","receiver","recipient","actor","initiator")

def get(name):
    urls=[
        f"https://zenodo.org/records/{RECORD}/files/{quote(name)}?download=1",
        f"https://zenodo.org/api/records/{RECORD}/files/{quote(name)}/content"
    ]
    last=None
    for i in range(4):
        try:
            r=requests.get(urls[i%2],timeout=(15,45),
                           headers={"User-Agent":"OpenFARM-academic-provenance-review/0.2"})
            r.raise_for_status()
            if len(r.content)>2_000_000:raise ValueError("File over cap")
            return r.content
        except requests.RequestException as e:
            last=e
            if i<3:time.sleep([1,2,4][i])
    raise RuntimeError(f"Failed after retries: {type(last).__name__}: {str(last)[:130]}")

def parse(data):
    for encoding in ("utf-8-sig","latin1"):
        for sep in ("\t",";",",",None):
            try:
                df=pd.read_csv(io.BytesIO(data),sep=sep,engine="python",encoding=encoding)
                if df.shape[1]>=3:return df
            except Exception:pass
    raise RuntimeError("No usable tabular parser")

def inspect(name):
    raw=get(name)
    df=parse(raw)
    out={"filename":name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
         "rows":len(df),"columns":[str(c) for c in df.columns],
         "full_row_duplicates":int(df.duplicated().sum()),
         "column_diagnostics":{}}
    for col in df.columns:
        key=str(col)
        if any(k in key.lower() for k in NEEDLES):
            series=df[col]
            distinct=int(series.nunique(dropna=True))
            d={"distinct":distinct,"missing":int(series.isna().sum())}
            if distinct<=24:
                d["distribution"]={str(k):int(v) for k,v in series.value_counts(dropna=False).head(24).items()}
            out["column_diagnostics"][key]=d
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    results={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={k:pool.submit(inspect,k) for k in FILES}
        for k,f in futures.items():
            try:results[k]=f.result()
            except Exception as e:results[k]={"error":f"{type(e).__name__}: {str(e)[:210]}"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report={"record":RECORD,"source":"10.5281/zenodo.5920843",
            "files":results,
            "caveat":"Recorded after-observation lesion score is not a proven prospective later outcome; determine sequence/time first."}
    a.output.write_text(json.dumps(report,indent=2,default=str)+"\n")
    print("PIG_SOCIAL_ETHOGRAM_AUDIT:"+json.dumps(report,default=str))
    if not any("error" not in r for r in results.values()):raise SystemExit(2)
if __name__=="__main__":main()
