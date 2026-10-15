"""Low-cost data-quality + simple-rule screening of a *candidate* HotPig decision.

Candidate: Is independently recorded electronic-feeder intake >0 in this
minute, using only camera-derived posture/event counts? This is NOT a pain,
heat stress, or welfare alert label. It does NOT use target feed, treatment
condition, timestamps, or pig identity in model inputs.

Only report aggregate metrics and group manifests; no raw author data committed.
"""
import argparse,hashlib,io,json,time,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SRC="https://zenodo.org/records/17090997/files/series.zip?download=1"
FROZEN_HASH="1ba5125167224b96"
FEATURES=["unknown","standing","seating","lying","eating","drinking","cuddling",
          "curious","idle","drink","eat","mate"]

def source_bytes():
    session=requests.Session()
    urls=[SRC,"https://zenodo.org/api/records/17090997/files/series.zip/content"]
    last=None
    for i in range(4):
        try:
            with session.get(urls[i%2],timeout=(20,65),stream=True) as response:
                response.raise_for_status()
                blob=bytearray()
                for chunk in response.iter_content(65536):
                    blob.extend(chunk)
                    if len(blob)>6_000_000:raise ValueError("Source too large")
                b=bytes(blob)
                if not hashlib.sha256(b).hexdigest().startswith(FROZEN_HASH):
                    raise RuntimeError("Unexpected updated HotPig version/bytes; do not silently drift")
                return b
        except requests.exceptions.RequestException as e:
            last=e
            if i<3:time.sleep([1,3,7][i])
    raise RuntimeError(f"Cannot access published source after retries: {last}")

def metrics(y,pred):
    y=np.asarray(y,dtype=bool)
    pred=np.asarray(pred,dtype=bool)
    tp=int(np.logical_and(y,pred).sum())
    fp=int(np.logical_and(~y,pred).sum())
    fn=int(np.logical_and(y,~pred).sum())
    tn=int(np.logical_and(~y,~pred).sum())
    return {"rows":len(y),"positives":int(y.sum()),"rate":round(float(y.mean()),5),
            "tp":tp,"fp":fp,"tn":tn,"fn":fn,
            "precision":round(tp/max(1,tp+fp),4),
            "recall":round(tp/max(1,tp+fn),4),
            "false_detector_events_per_pig_day":round(fp/(len(y)/1440),2)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    zipblob=source_bytes()
    z=zipfile.ZipFile(io.BytesIO(zipblob))
    pigids=sorted([i for i in range(1,25)],key=lambda i:i)
    splits={"train":pigids[:16],"calibration":pigids[16:20],"test":pigids[20:]}
    output={"source_doi":"10.5281/zenodo.17090997",
            "source_sha256":hashlib.sha256(zipblob).hexdigest(),
            "dataset_license":"CC BY 4.0 (Zenodo API)",
            "task":"Noul(actual electronic-feeder intake >0 in current minute)",
            "input_features":FEATURES,
            "not_used_as_input":["feed","period","conditions","state","time_index","pig_id"],
            "label_origin":"electronic feeder (not human emotion/welfare and not YOLO posture label)",
            "splits":splits,
            "grouping":"entire pig individual; 16 train / 4 calibration / 4 test, single trial/site",
            "pig_rows":{},
            "rule":{},
            "safety_note":"False events here are discrepancies with a feeder sensor, NOT false welfare alerts. No clinical distress inference permitted."}
    agg={}
    for split,ids in splits.items():
        ground=[]
        posture_rule=[]
        for pig in ids:
            data=pd.read_csv(z.open(f"series/P{pig}.csv"))
            if len(data)!=23040:raise RuntimeError("Each pig must have 16 complete 1440-minute days")
            if not set(FEATURES+["feed"]).issubset(data.columns):
                raise RuntimeError("Required model feature or independent sensor reference absent")
            valid=data["feed"].notna()
            for feat in FEATURES:
                valid &= data[feat].notna()
            sub=data.loc[valid]
            feed=pd.to_numeric(sub["feed"],errors="coerce")
            if feed.isna().any():raise RuntimeError("Nonnumeric measured feeder data")
            observed=feed.to_numpy()>0
            predict=(pd.to_numeric(sub["eating"],errors="coerce")>0).to_numpy()
            ground.extend(observed.tolist())
            posture_rule.extend(predict.tolist())
            output["pig_rows"][str(pig)]={
                "source_rows":len(data),"valid_rows":int(valid.sum()),
                "feeder_positive":int(observed.sum()),
                "eating_detector_positive":int(predict.sum()),
                "feed_nonnegative":bool((feed>=0).all())
            }
        result=metrics(ground,posture_rule)
        result["naive_no-feeder_recall"]=0.0
        result["naive_no-feeder_accuracy"]=round(1-result["rate"],4)
        output["rule"][split]=result
    output["approval_gate"]="Task-level research training/evaluation decision awaits reviewer verdict, checks of data provenance and triviality; no field welfare benchmark."
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    print("HOTPIG_RULE_CANDIDATE:"+json.dumps(output))

if __name__=="__main__":main()
