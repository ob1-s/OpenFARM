"""Micro-audit actual published source tables + source rights before task approval.

Fetches only small public assets. Does NOT publish the original data or approve
a benchmark on the basis of a schema-only review.

Usage: python scripts/prospecting_microaudit.py --output reports/sources.json
Deps: requests pandas openpyxl
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,zipfile
from pathlib import Path
from urllib.parse import quote
import requests
import pandas as pd

SOURCES={
    "pig_feeder_tail_biting":{"record":5028016,"keep":["Swiss_Swedish_25.06.21.txt"]},
    "pig_social_interaction":{"record":5920843,"keep":["animal.txt","activity.txt","confrontations.txt","conspecifics.txt","lesions.txt","objects.txt","rooting.txt"]},
    "hotpig":{"record":17090997,"keep":["metadata.txt","series.zip"]},
}
USER_AGENT="OpenFARM-scientific-audit/0.1 (read-only metadata; public research data)"
S=requests.Session()
S.headers.update({"User-Agent":USER_AGENT})

def download(url,limit=8_000_000):
    with S.get(url,timeout=40,stream=True,allow_redirects=True) as response:
        response.raise_for_status()
        payload=bytearray()
        for chunk in response.iter_content(65536):
            payload.extend(chunk)
            if len(payload)>limit:raise ValueError("Download larger than source-audit cap")
        return bytes(payload)

def meta(record):
    r=S.get(f"https://zenodo.org/api/records/{record}",timeout=35)
    r.raise_for_status()
    return r.json()

def preview_table(raw,filename):
    """Read only tabular data and report schema/distribution, not personal identities."""
    for sep in (None, "\t", ";", ","):
        try:
            data=pd.read_csv(io.BytesIO(raw),sep=sep,engine="python",on_bad_lines="warn",
                             encoding="utf-8-sig",low_memory=False if sep is not None else None)
        except (Exception,):
            try:
                data=pd.read_csv(io.BytesIO(raw),sep=sep,engine="python",
                                 encoding="latin1")
            except Exception:
                continue
        if data.shape[1]>=3:
            break
    else: return {"error":"Could not parse 3+ tabular columns"}
    columns=[str(x) for x in data.columns]
    result={"rows":int(len(data)),"columns":columns[:80],
            "unique_primary_columns":{},
            "nulls":{c:round(float(data[c].isna().mean()),4) for c in columns[:35]}}
    for col in columns:
        cl=col.lower()
        if any(x in cl for x in ("farm","pen","pig","animal","group","label","tail","start","date","behav","type","activity","lesion","event","posture")):
            if data[col].dtype=="object" or data[col].nunique(dropna=True)<100:
                result["unique_primary_columns"][col] = {
                    "distinct":int(data[col].nunique(dropna=True)),
                    "counts":{str(k):int(v) for k,v in
                              list(data[col].value_counts(dropna=False).head(8).items())}
                }
            else:
                result["unique_primary_columns"][col]={"distinct":int(data[col].nunique(dropna=True))}
    if filename.startswith("Swiss_Swedish"):
        required={"Farm","PenID","PenType","TBSTART","date"}
        if required.issubset(set(columns)):
            pens=data.groupby(["Farm","PenID"],dropna=False,sort=False)
            df=data.groupby(["Farm","PenID"],dropna=False).agg(
                type=("PenType","first"),onset=("TBSTART","first"),
                records=("PenType","size"),start=("date","min"),end=("date","max"))
            result["tail_biting"]={
                "pens":int(len(df)),
                "farms":int(data["Farm"].nunique()),
                "pen_type_counts":{str(k):int(v) for k,v in df["type"].value_counts().items()},
                "pen_has_conflicting_outcome":int((pens["PenType"].nunique()>1).sum()),
                "pen_has_multiple_onsets":int((pens["TBSTART"].nunique()>1).sum()),
                "date_min":str(data["date"].min()),"date_max":str(data["date"].max()),
                "warn":"Control pens have arbitrarily assigned TBSTART. NEVER treat their pseudo-date as an observed event."
            }
    return result

def audit_zenodo(name,cfg):
    j=meta(cfg["record"])
    metadata=j.get("metadata",{})
    access=metadata.get("access_right",{})
    lic=metadata.get("license",{})
    if isinstance(lic,dict):lic={"id":lic.get("id"),"title":lic.get("title"),"url":lic.get("url")}
    entry={
        "zenodo_record":cfg["record"],
        "doi":metadata.get("doi",j.get("doi")),
        "license":lic,
        "access_right":access,
        "published":metadata.get("publication_date"),
        "all_file_names":[f.get("key") for f in j.get("files",[])],
        "files":{},
    }
    files={f.get("key"):f for f in j.get("files",[])}
    for fname in cfg["keep"]:
        if fname not in files:
            entry["files"][fname]={"error":"missing source file"};continue
        f=files[fname]; link=f.get("links",{}).get("self") or (
                f"https://zenodo.org/records/{cfg['record']}/files/{quote(fname)}?download=1")
        try:
            blob=download(link)
            sha=hashlib.sha256(blob).hexdigest()
            cell={"length":len(blob),"sha256":sha,"record_checksum":f.get("checksum")}
            if fname.endswith(".zip"):
                z=zipfile.ZipFile(io.BytesIO(blob))
                names=[i.filename for i in z.infolist() if not i.is_dir()]
                cell["archive_entries"]=len(names)
                cell["entry_names"]=names[:45]
                cell["small_tabular_members"]={}
                for member in names:
                    if len(cell["small_tabular_members"])>=3:break
                    if not member.lower().endswith((".txt",".csv",".tsv")):continue
                    if z.getinfo(member).file_size>3_000_000:continue
                    cell["small_tabular_members"][member]=preview_table(z.read(member),member)
            elif fname.lower().endswith((".csv",".txt",".tsv")):
                cell["table"]=preview_table(blob,fname)
            else:cell["text_preview"]=blob.decode("utf-8","replace")[:1600]
            entry["files"][fname]=cell
        except Exception as e:
            entry["files"][fname]={"error":f"{type(e).__name__}: {e}"}
    return entry

def audit_equifacs():
    root="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0231608."
    r={"article":"10.1371/journal.pone.0231608","studies":"6 experimental (12 videos) + 21 clinical"}
    for idx in ("s002","s003"):
        name=f"equifacs_{idx}.xlsx"
        try:
            blob=download(root+idx+"&type=supplementary",limit=1_000_000)
            b=io.BytesIO(blob)
            x=pd.ExcelFile(b)
            sheets={}
            for sheet in x.sheet_names:
                df=pd.read_excel(io.BytesIO(blob),sheet_name=sheet)
                sheets[sheet]={"rows":int(len(df)),"columns":[str(c) for c in df.columns[:35]]}
            r[name]={"length":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),"sheets":sheets}
        except Exception as e:r[name]={"error":f"{type(e).__name__}: {e}"}
    r["rights"]="PLOS article CC BY, but verify supplementary video/data rights independently; don't redistribute original videos in audit reports."
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();results={}
    for name,cfg in SOURCES.items():
        try:results[name]=audit_zenodo(name,cfg)
        except Exception as e:results[name]={"error":f"{type(e).__name__}: {e}"}
    results["equifacs_supplements"]=audit_equifacs()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("PROSPECT_MICROAUDIT:"+json.dumps(results,ensure_ascii=False))
    # Ingest may fail intermittently; report transparently rather than claim pass.
    print("SOURCE_AUDIT_WRITTEN",str(args.output))
if __name__=="__main__":main()
