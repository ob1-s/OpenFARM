"""Build strictly scoped HotPig instrumental feeding datasets for System-1 research.

Verified source: Zenodo 17090997 series.zip, CC BY 4.0.
Source data belongs to its authors. Outputs (train/cal/test) are created locally,
never committed here. Use exact author attribution and original rights.

Training/eval question:
  Noul("Did the independent electronic feeder report intake >0 in this minute?")
Inputs ONLY 12 published machine-generated camera posture/event counts.
Important: This is *not* pain, affect, or welfare alert supervision.

Reproduce:
  python scripts/build_hotpig_feeder_v0.py --output-dir /tmp/hotpig-feeder-v0
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_hotpig_feeder_candidate import source_bytes,FEATURES

SPLITS={"train":list(range(1,17)),"calibration":list(range(17,21)),
        "test":list(range(21,25))}
SOURCE_DOI="10.5281/zenodo.17090997"

def build(output_dir):
    archive=source_bytes()
    zipfile_obj=zipfile.ZipFile(io.BytesIO(archive))
    if len(set(sum(SPLITS.values(),[])))!=24:
        raise RuntimeError("Pig source grouping must be disjoint")
    report={
      "question":"Noul(actual electronic feeder intake >0 in the same minute)",
      "scope":"single-study, 24 subject-heldout pigs; not generalizable across sites by design",
      "source_doi":SOURCE_DOI,
      "source_sha256":hashlib.sha256(archive).hexdigest(),
      "rights":"CC BY 4.0 source (record metadata); attribute original authors",
      "label_origin":"electronic feeders, not inferred welfare/stress",
      "input_features":FEATURES,
      "excluded":["feed","state","period","conditions","pig_id","minute_id","timestamp"],
      "splits":SPLITS,
      "materialized":{}
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    for split,pigs in SPLITS.items():
        frames=[]
        labels=[]
        provenance=[]
        for pig in pigs:
            src=pd.read_csv(zipfile_obj.open(f"series/P{pig}.csv"))
            if len(src)!=23040:raise RuntimeError(f"Wrong original minute count for pig {pig}")
            required=set(FEATURES+["feed"])
            if not required.issubset(src.columns):raise RuntimeError(f"Missing original feature: {required-set(src.columns)}")
            numeric=src[FEATURES+["feed"]].apply(pd.to_numeric,errors="coerce")
            finite=np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
            positive=(numeric.loc[finite,"feed"]>0).astype(int)
            input_df=numeric.loc[finite,FEATURES].reset_index(drop=True)
            if (numeric.loc[finite,FEATURES]<0).any().any():
                raise RuntimeError("Negative video event counts: wrong preprocessing schema")
            frames.append(input_df)
            labels.append(pd.DataFrame({"feeder_event":positive.to_numpy(dtype=int)}))
            provenance.append(pd.DataFrame({
                "pig_id":[pig]*int(finite.sum()),
                "source_minute_index":np.flatnonzero(finite)
            }))
        x=pd.concat(frames,ignore_index=True)
        y=pd.concat(labels,ignore_index=True)
        ids=pd.concat(provenance,ignore_index=True)
        assert len(x)==len(y)==len(ids)
        assert set(ids["pig_id"])==set(pigs)
        # Group/temporal columns are deliberately isolated: never model inputs.
        x.to_csv(output_dir/f"{split}.features.csv.gz",index=False,compression="gzip")
        y.to_csv(output_dir/f"{split}.labels.csv.gz",index=False,compression="gzip")
        ids.to_csv(output_dir/f"{split}.provenance.csv.gz",index=False,compression="gzip")
        digest=hashlib.sha256(
            ("\n".join(map(str,ids["pig_id"]))+"\n"+
             "\n".join(map(str,ids["source_minute_index"]))).encode()).hexdigest()
        report["materialized"][split]={
            "samples":len(x),"positive_samples":int(y["feeder_event"].sum()),
            "positive_prevalence":float(y["feeder_event"].mean()),
            "animal_groups":pigs,"membership_digest":digest
        }
    assert report["materialized"]["train"]["samples"]==368508
    assert report["materialized"]["calibration"]["samples"]==90150
    assert report["materialized"]["test"]["samples"]==92160
    report["warning"]="Heldout pig identifiers alone do not imply heldout farms, cameras, experimental conditions, or independent time domains. Prepared data are diagnostic instrumental behavior labels, not affective welfare labels."
    (output_dir/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    result=build(args.output_dir)
    print("HOTPIG_EXPORT_OK:"+json.dumps(result))

if __name__=="__main__":main()
