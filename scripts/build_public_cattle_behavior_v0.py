"""Admit and benchmark CC-BY human-labeled wearable cattle behavior; fail closed.

Cattle Figshare 28801181 v1, "cattle_behavior_data_time_V2.csv":
  observed input: proprietary 1-minute motion index (MI) from accelerometer
  independently reviewed video label: GR, WA, RE, RU
  groups: 24 cows (6/breed) across two sites, site confounded with breed
  NOT pain, emotion, clinical illness, or animal-care alert

Comparators: train prior, CPU-only standardized multinomial logistic regression,
minimal histogram gradient boosted tree. No evaluation data in preprocessing,
model fitting or hyperparameter/threshold decisions.
"""
from __future__ import annotations
import argparse,hashlib,io,json,collections
from pathlib import Path
import requests,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,f1_score,balanced_accuracy_score,log_loss,
                             confusion_matrix)
FIG=28801181
NAME="cattle_behavior_data_time_V2.csv"
BEHAVIORS=["GR","WA","RE","RU"]
SITES={"RC":"Jornada","AH":"Jornada","BR":"Chihuahuan_Desert","BH":"Chihuahuan_Desert"}
def source():
    """Official Figshare file retrieval with hash-pinned cache and fallback.

    Figshare's ndownloader intermittently returned HTTP 500 on Actions
    (2026-10-19). Try documented public API download and browser endpoint;
    never accept alternate bytes without BOTH source MD5 and frozen SHA256.
    """
    import time
    meta=requests.get(f"https://api.figshare.com/v2/articles/{FIG}",timeout=25)
    meta.raise_for_status()
    meta=meta.json()
    if meta.get("license",{}).get("name")!="CC BY 4.0":
        raise RuntimeError("Figshare cattle source's license changed")
    matches=[x for x in meta["files"] if x.get("name")==NAME]
    if len(matches)!=1:raise RuntimeError("Timestamped human annotations not found")
    f=matches[0]
    source_sha="b4d6e14d6ff029dbf497287a774aaa27ec471e7ed1ef5884e9cf0207c784fe37"
    cache=Path(".cache/cattle/cattle_behavior_data_time_V2.csv")
    def good(blob):
        return hashlib.md5(blob).hexdigest()==f.get("supplied_md5") and hashlib.sha256(blob).hexdigest()==source_sha
    blob=None
    if cache.exists():
        candidate=cache.read_bytes()
        if good(candidate):blob=candidate
    if blob is None:
        fid=f["id"]
        routes=[f["download_url"],
                f"https://api.figshare.com/v2/file/download/{fid}",
                f"https://figshare.com/ndownloader/files/{fid}"]
        failures=[]
        for attempt in range(3):
            for url in dict.fromkeys(routes):
                try:
                    resp=requests.get(url,timeout=55,headers={"User-Agent":"Mozilla/5.0 OpenFARM-source-audit/0.2"})
                    resp.raise_for_status()
                    if not good(resp.content):
                        raise RuntimeError("Figshare source checksum/content mismatch")
                    blob=resp.content
                    break
                except (requests.RequestException,RuntimeError) as exc:
                    failures.append(f"{url.split('/')[2]}:{type(exc).__name__}:{str(exc)[:90]}")
            if blob is not None:break
            time.sleep(2*(attempt+1))
        if blob is None:
            raise RuntimeError("Original pinned cattle CSV inaccessible through official URLs: "+
                               "; ".join(failures[-6:]))
        cache.parent.mkdir(parents=True,exist_ok=True)
        cache.write_bytes(blob)
    df=pd.read_csv(io.BytesIO(blob))
    return df,{"article_id":FIG,"doi":meta.get("doi"),"version":meta.get("version"),
               "file_md5":hashlib.md5(blob).hexdigest(),"file_sha256":source_sha,
               "license":"CC BY 4.0"}
def prepare(df):
    expected={"Breed","ID","Time.MTD","MI","State","Behavior"}
    if set(df.columns)!=expected:raise RuntimeError("Changed cattle schema")
    if df.isna().any().any():raise RuntimeError("Missing observation fields")
    if set(df["Behavior"])!=set(BEHAVIORS):raise RuntimeError("Unknown behavior class")
    if df["Breed"].nunique()!=4 or df["ID"].nunique()!=24:raise RuntimeError("Animal group mismatch")
    if len(df)!=9222:raise RuntimeError("Original row count changed")
    if df.groupby("ID")["Breed"].nunique().max()!=1:raise RuntimeError("Cow breed ID drift")
    # Source 'Time.MTD' is *not* proven a unique date-time and may repeat
    # within one cow across days or trials. Do NOT force a 1-row-per-ID-time
    # index, collapse disagreeing annotations, or pretend unique episodes.
    # Preserve complete original source rows; cow-level grouping is sound.
    keys=["Breed","ID","Time.MTD"]
    dups=df.duplicated(keys,keep=False)
    collisions=df[dups].groupby(keys)[["Behavior","MI","State"]].nunique()
    conflicts=(collisions>1).any(axis=1)
    clean=df.copy()
    clean["site"]=clean["Breed"].map(SITES)
    if clean["site"].isna().any():raise RuntimeError("Unmapped site")
    if any(clean.groupby("Breed")["ID"].nunique()!=6):
        raise RuntimeError("Unequal source cow counts per breed")
    # The paper says State=AC for GR/WA; ST for RE/RU. Verify source integrity.
    exp=clean["Behavior"].map({"GR":"AC","WA":"AC","RE":"ST","RU":"ST"})
    if (exp!=clean["State"]).any():raise RuntimeError("Behavior/State contradictory")
    return clean,{"raw_rows":len(df),"full_row_exact_repeats":int(df.duplicated().sum()),
                  "source_cow_time_repeated_rows":int(dups.sum()),
                  "source_cow_time_conflicting_groups":int(conflicts.sum()),
                  "time_key_validity":"NOT VERIFIED. Do not use to construct event durations or temporal splits.",
                  "rows_retained":len(clean),
                  "site_counts":clean.groupby("site")["ID"].nunique().to_dict(),
                  "class_counts":clean["Behavior"].value_counts().to_dict(),
                  "animal_counts":{str(k):int(v) for k,v in clean["ID"].value_counts().items()}}
def groups(clean):
    bybreed={k:sorted(set(clean.loc[clean.Breed==k,"ID"])) for k in sorted(SITES)}
    split={"train":[],"calibration":[],"test":[]}
    for breed,ids in bybreed.items():
        split["train"]+=ids[:4];split["calibration"].append(ids[4]);split["test"].append(ids[5])
    j=sorted(set(clean.loc[clean.site=="Jornada","ID"]))
    c=sorted(set(clean.loc[clean.site=="Chihuahuan_Desert","ID"]))
    # Source-context transfer, only 2 sites; train and calibrate within one
    # site, test completely disjoint site and breeds. In both directions.
    transfer={
      "jornada_to_chihuahuan":{"train":j[:9],"calibration":j[9:],"test":c},
      "chihuahuan_to_jornada":{"train":c[:9],"calibration":c[9:],"test":j},
    }
    return {"breed_balanced_group_holdout":split,**transfer}
def score(y,p,labels=BEHAVIORS):
    p=np.asarray(p)
    y=np.asarray(y)
    yp=np.array(labels)[np.argmax(p,axis=1)]
    return {
       "n":len(y),"accuracy":round(float(accuracy_score(y,yp)),5),
       "macro_f1":round(float(f1_score(y,yp,labels=labels,average="macro",zero_division=0)),5),
       "balanced_accuracy":round(float(balanced_accuracy_score(y,yp)),5),
       "log_loss":round(float(log_loss(y,p,labels=labels)),5),
       "brier_multiclass":round(float(np.mean(np.sum((p-np.eye(len(labels))[[labels.index(v) for v in y]])**2,axis=1))),5),
       "confusion_gt_rows_pred_cols":confusion_matrix(y,yp,labels=labels).tolist(),
       "prevalence":{k:int((y==k).sum()) for k in labels}
    }
def experiment(df,parts):
    pools={k:df[df.ID.isin(ids)] for k,ids in parts.items()}
    for a,b in (("train","calibration"),("train","test"),("test","calibration")):
        if set(parts[a])&set(parts[b]):raise AssertionError("Cow overlap "+a+"/"+b)
    X=pools["train"][["MI"]].to_numpy(float)
    y=pools["train"]["Behavior"].to_numpy()
    classes=BEHAVIORS
    results={"groups":parts,"rows":{k:len(p) for k,p in pools.items()},
             "sites":{k:dict(p.site.value_counts()) for k,p in pools.items()},
             "by_breed":{k:dict(p.Breed.value_counts()) for k,p in pools.items()},
             "per_animal_test":{},"model_scores":{}}
    if len(set(y))!=4:raise RuntimeError("Train group misses class")
    prior=np.array([(y==c).mean() for c in classes],float)
    models={
      "train_prior":None,
      "L2_multinomial_logistic":make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=17)),
      "tiny_gradient_boosted_trees":HistGradientBoostingClassifier(max_iter=75,
         max_depth=3,max_leaf_nodes=7,min_samples_leaf=30,learning_rate=.08,
         l2_regularization=2.,random_state=17,early_stopping=False)
    }
    for name,model in models.items():
        if model is not None:model.fit(X,y)
        test=pools["test"]
        if model is None:prob=np.tile(prior,(len(test),1))
        else:
            raw=model.predict_proba(test[["MI"]].to_numpy(float))
            prob=raw[:,[list(model.classes_).index(c) for c in classes]]
        r=score(test["Behavior"].to_numpy(),prob)
        r["test_by_cow"]={str(id):score(test.loc[test.ID==id,"Behavior"].to_numpy(),prob[test.ID.to_numpy()==id])
                           for id in parts["test"]}
        results["model_scores"][name]=r
    return results
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--local-dir",type=Path,default=None,
                    help="Materialize local-only split inputs, targets and provenance for research")
    a=ap.parse_args()
    raw,manifest=source();df,integrity=prepare(raw)
    result={"source":manifest,"integrity":integrity,
            "task":"Choice(observed grazing GR, walking WA, resting RE, rumination RU) from prior wearable MI only",
            "scope":"Human video-coded behavior target. One scalar motion-index (proprietary upstream); no raw video inference, pain, distress, welfare judgment or alert.",
            "fit_inputs":["MI"],"forbidden_as_model_inputs":["ID","Breed","site","Time.MTD","State","Behavior"],
            "site_groups":"Paper-defined: AH/RC at Jornada and BR/BH at Chihuahuan Desert center; site and breed confounded.",
            "results":{},
            "rights":"Figshare CC BY 4.0; original video content not part of this release.",
            "narrow_claim":"Within-source sensor-to-independent human-observed behavior generalization under cow and 2-site holdouts; exact group numbers tiny."}
    for name,part in groups(df).items():
        result["results"][name]=experiment(df,part)
        if a.local_dir is not None:
            out=a.local_dir/name
            out.mkdir(parents=True,exist_ok=True)
            for split,ids in part.items():
                rows=df[df.ID.isin(ids)].copy()
                # Model-visible sensor-only state, independent human annotation
                # and source/group keys are ALWAYS separated on disk.
                rows[["MI"]].to_csv(out/f"{split}.features.csv.gz",
                                     index=False,compression="gzip")
                rows[["Behavior"]].to_csv(out/f"{split}.labels.csv.gz",
                                           index=False,compression="gzip")
                rows[["Breed","ID","site","Time.MTD","State"]].to_csv(
                    out/f"{split}.provenance.csv.gz",index=False,compression="gzip")
                if rows["ID"].nunique()!=len(ids):
                    raise RuntimeError("Source grouping invalid during export")
    def json_primitive(obj):
        if isinstance(obj,np.generic):return obj.item()
        if isinstance(obj,np.ndarray):return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj).__name__}")
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,default=json_primitive)+"\n")
    print("CATTLE_SYSTEM1_REAL_LABEL_BENCH:"+json.dumps({
        "integrity":integrity,"source":manifest,"outcomes":{
            key:{m:{x:v for x,v in metrics.items() if x not in ("test_by_cow","confusion_gt_rows_pred_cols")}
                 for m,metrics in run["model_scores"].items()}
            for key,run in result["results"].items()}},default=json_primitive))
if __name__=="__main__":main()
