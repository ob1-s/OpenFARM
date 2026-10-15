"""Independent audit of SoundWel's published train/test recording identities.

Checks exact audio bytes, canonical filename-derived pig identities, and coarse
recording-session overlap. No data changes. Does NOT certify subjects because
the archive does not publish original PigID/Leakage_Group fields.
"""
import argparse, hashlib, json, re
from collections import Counter,defaultdict
from pathlib import Path

SOURCE="oliveirabruno01/soundwel-pig-vocalizations"
PATTERNS=[
 r"Sow([0-9]+)Pig([0-9]+)",r"Pr([0-9]+)Sele([0-9]+)",
 r"s([0-9]+)p([0-9]+)",r"Pair([0-9]+)",r"VT([0-9]+)",
 r"pig([A-Za-z0-9]+)"
]

def key(row):
    filename=str(row.get("Audio Filename") or "")
    team=str(row.get("Recording Team") or "")
    context=str(row.get("Context") or "")
    sex=str(row.get("Sex") or "")
    age=str(row.get("Age Category") or "")
    subject=None
    for pattern in PATTERNS:
        match=re.search(pattern,filename,re.I)
        if match:
            subject="_".join(match.groups()).lower()
            break
    if subject:
        return {"base":f"{team}|{subject}",
                "composite":f"{team}|{subject}|{sex}|{age}",
                "fallback":False}
    return {"base":f"{team}|session|{context}",
            "composite":f"{team}|session|{context}","fallback":True}

def dataset_rows():
    from datasets import load_dataset,Audio
    from huggingface_hub import HfApi
    rev=HfApi().dataset_info(SOURCE).sha
    out={}
    for split in ("train","test"):
        ds=load_dataset(SOURCE,split=split,revision=rev,streaming=True)
        ds=ds.cast_column("audio",Audio(decode=False))
        records=[]
        for r in ds:
            media=r.get("audio")
            if not isinstance(media,dict) or not isinstance(media.get("bytes"),bytes):
                raise ValueError(f"{split} audio bytes unavailable")
            grouping=key(r)
            records.append({
                "audio_sha256":hashlib.sha256(media["bytes"]).hexdigest(),
                "base":grouping["base"],"composite":grouping["composite"],
                "fallback":grouping["fallback"],
                "team":str(r.get("Recording Team") or ""),
                "session":str(r.get("Recording Team") or "")+"|"+
                          str(r.get("Recording Date") or "")+"|"+str(r.get("Context") or ""),
                "type":str(r.get("Type") or ""),
                "valence":str(r.get("Valence") or ""),
                "context_category":str(r.get("Context Category") or "")
            })
        out[split]=records
    return rev,out

def audit(revision,rows):
    a,b=rows["train"],rows["test"]
    shared={}
    for col in ("audio_sha256","base","composite","session"):
        sa={r[col] for r in a if r[col]};sb={r[col] for r in b if r[col]}
        shared[col]={"train_unique":len(sa),"test_unique":len(sb),
                     "shared":len(sa&sb)}
    contradictions={}
    for name,v in rows.items():
        labels=defaultdict(set)
        for r in v: labels[r["audio_sha256"]].add(r["type"])
        contradictions[name]=sum(len(val)>1 for val in labels.values())
    return {
       "source":SOURCE,"revision":revision,
       "row_counts":{name:len(v) for name,v in rows.items()},
       "type_counts":{name:dict(Counter(r["type"] for r in v)) for name,v in rows.items()},
       "fallback_session_fraction":{name:round(sum(r["fallback"] for r in v)/len(v),3)
                                   for name,v in rows.items()},
       "cross_split_overlap":shared,"conflicting_HF_LF_exact_media":contradictions,
       "notes":"Base group is team+filename-regex subject, otherwise team+context. Conservative approximation; original source pig identities and recording session annotation are NOT verified."
    }

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    revision,rows=dataset_rows()
    result=audit(revision,rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("SOUNDWEL GROUP AUDIT:"+json.dumps(result))
if __name__=="__main__":main()
