"""Welfare Evidence Sprint: injury/treatment starts vs actual feeder records.

Source Ollagnier et al. 2023 PLOS ONE, Zenodo 5028016, CC BY-NC 4.0.
Dataset includes date of *FIRST ACTUAL TREATMENT* on tail-damage pens.
Controls have ARBITRARY pseudo-start dates. This source CAN'T be treated as
naturally sampled prospective pen-day surveillance until selection is audited.
Script audits bytes, group consistency, calendar and event coverage only.
"""
from __future__ import annotations
import argparse,hashlib,io,json,time,collections
from pathlib import Path
import requests
import pandas as pd

RECORD=5028016
NAME="Swiss_Swedish_25.06.21.txt"
URLS=[
f"https://zenodo.org/records/{RECORD}/files/{NAME}?download=1",
f"https://zenodo.org/api/records/{RECORD}/files/{NAME}/content",
f"https://zenodo.org/records/{RECORD}/files/{NAME}"
]
MD5="c6106233c17c0672ad87b4cbb92dd899"
def download():
    errors=[]
    for i in range(9):
        try:
            with requests.get(URLS[i%len(URLS)],timeout=(22,85),stream=True,
                headers={"User-Agent":"OpenFARM-Welfare-Data-Scientific-Audit/0.1"}) as r:
                r.raise_for_status()
                data=bytearray()
                for ch in r.iter_content(65536):
                    data.extend(ch)
                    if len(data)>5_000_000:raise RuntimeError("Source exceeded known 4MB cap")
                b=bytes(data)
            if hashlib.md5(b).hexdigest()!=MD5:
                raise RuntimeError("File digest fails published source MD5")
            return b,errors
        except (requests.exceptions.RequestException,ConnectionError,TimeoutError) as e:
            errors.append(f"{type(e).__name__}: {str(e)[:110]}")
            if i<8:time.sleep([2,2,4,5,8,8,10,10][i])
    raise RuntimeError("Source unavailable after 9 bounded official-source attempts: "+" / ".join(errors[-3:]))

def parse(raw):
    for sep in ("\t",";",",",None):
        try:
            d=pd.read_csv(io.BytesIO(raw),sep=sep,engine="python",encoding="utf-8-sig")
            if d.shape[1]>=8:return d
        except Exception:pass
    raise RuntimeError("No tabular parse, cannot validate source")

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    b,errs=download()
    d=parse(b)
    info={
       "source":"https://zenodo.org/records/5028016","license_from_original_metadata":"CC BY-NC 4.0 (research-only until rights cleared)",
       "actual_md5":hashlib.md5(b).hexdigest(),"sha256":hashlib.sha256(b).hexdigest(),
       "file_bytes":len(b),"rows":len(d),"columns":list(d.columns),"download_retry_failures":len(errs),
       "null_fraction":{str(k):round(float(d[k].isna().mean()),5) for k in d if d[k].isna().any()},
       "warning":"Controls have synthetic TBSTART; NEVER train with TBSTART, PenType, source event-window labels or future feeder data exposed."
    }
    for key in d:
        low=str(key).lower()
        if any(s in low for s in ("farm","pen","tail","tbstart","date","id","dfv","dfc","stdfc","type")):
            v=d[key]
            col={"distinct":int(v.nunique(dropna=True)), "nulls":int(v.isna().sum())}
            if v.nunique(dropna=True)<=40:
                col["distribution"]={str(k):int(x) for k,x in v.value_counts(dropna=False).head(35).items()}
            info.setdefault("key_fields",{})[key]=col
    needed={"Farm","PenID","PenType","TBSTART_","date"}
    if needed.issubset(d.columns):
        by_pen=d.groupby(["Farm","PenID"],dropna=False)
        out=[]
        for (site,pen),frame in by_pen:
            out.append({"farm":str(site),"type":str(frame.PenType.iloc[0]),
                        "rows":len(frame),"unique_animals":int(frame["ID"].nunique()) if "ID" in frame else None,
                        "start":str(frame.date.min()),"end":str(frame.date.max()),
                        "onset":str(frame["TBSTART_"].iloc[0]),
                        "type_conflict":bool(frame.PenType.nunique()>1),
                        "date_conflict":bool(frame["TBSTART_"].nunique()>1)})
        info["pen_groups"]=len(out)
        info["pens_by_farm_type"]={f"{site}|{pen_type}":int(n) for (site,pen_type),n in
            collections.Counter((str(x["farm"]),x["type"]) for x in out).items()}
        info["pen_type_or_onset_conflicts"]=[x for x in out if x["type_conflict"] or x["date_conflict"]][:15]
        info["example_group_summaries"]=out[:12]
        info["first_treatment_events"]=sum(str(x["type"]).upper()=="TB" for x in out)
        info["source_date_encoding_by_farm"]={
          str(site):{
            "date_length_distribution":dict(collections.Counter(
                map(lambda s:len(str(s)),d.loc[d.Farm==site,"date"]))),
            "onset_length_distribution":dict(collections.Counter(
                map(lambda s:len(str(s)),d.loc[d.Farm==site,"TBSTART_"]))),
            "first_five_source_date_literals":list(map(str,d.loc[d.Farm==site,"date"].head(5))),
            "first_five_onset_literals":list(map(str,d.loc[d.Farm==site,"TBSTART_"].head(5)))
          } for site in d.Farm.unique()
        }
        info["farm_pen_groups"]=out
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")
    print("TAIL_DAMAGE_OUTCOME_AUDIT:"+json.dumps(info,ensure_ascii=False))
if __name__=="__main__":main()
