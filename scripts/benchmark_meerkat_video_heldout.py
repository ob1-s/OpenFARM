"""OpenFARM: reproducible *whole-video-heldout* human meerkat ethogram benchmark.

Source:
 Rogers et al. (CV4Animals 2023), Wellington Zoo, 20 annotated videos.
 https://meerkat-dataset.github.io/Annotations.zip (SHA pinned)
 https://auckland.figshare.com/articles/dataset/23538261 (CC BY-NC-SA 4.0)

Tasks:
 Choice(15 human ethogram actions) from geometry + **causal** bounding-box
 displacement (past only, manually annotated boxes), no video frames/pixels.
 This is a *pose/geometry input* lightweight System-1 diagnostic, not actual
 raw-video behavior recognition nor independently measured emotional valence.

Independent test is four WHOLE unseen videos, not random adjacent frames/tracks.
Test videos may contain the SAME animals as train videos (one zoo enclosure).
Never use label, frame number, track, video ID, date, camera as model features.
No labels and media are redistributed in reports.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile,collections
from pathlib import Path
import numpy as np,pandas as pd,requests
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix

SOURCE="https://meerkat-dataset.github.io/Annotations.zip"
SHA256="7ad7b369d5a9fc90a29c28dabfc0503c7c67af10d304d97ba1b5413260890463"
FILE_COLS=["frame","object_type","track_id","behavior","occluded","x1","x2","y1","y2"]
STEP=50
FEAT_STATIC=["cx","cy","w","h","aspect","log_area"]
FEAT_MOTION=["dx_per_frame","dy_per_frame","norm_speed","sample_gap","prev_aspect","relative_scale_change"]
FEATS=FEAT_STATIC+FEAT_MOTION
ALL_LABELS=list(range(1,16))
def archive():
    with requests.get(SOURCE,timeout=(25,90),stream=True) as r:
        r.raise_for_status()
        out=bytearray()
        for chunk in r.iter_content(262144):
            out.extend(chunk)
            if len(out)>30_000_000:raise RuntimeError("Meerkat ZIP exceeded source cap")
        b=bytes(out)
    if hashlib.sha256(b).hexdigest()!=SHA256:
        raise RuntimeError("Exact research source archive changed")
    return zipfile.ZipFile(io.BytesIO(b)),hashlib.sha256(b).hexdigest()

def groups(names):
    # Fixed deterministic video splits, stratified only by camera name,
    # NEVER by the test labels or downstream model scores.
    cameras=collections.defaultdict(list)
    for name in names:
        parts=Path(name).stem.split("_")
        camera=parts[1]
        if camera not in ("C2","C3"):raise RuntimeError("unexpected setup camera")
        cameras[camera].append(name)
    result={"train":[],"calibration":[],"test":[]}
    for cam,nms in cameras.items():
        ordered=sorted(nms,key=lambda n:hashlib.sha256(("OpenFARM-videoholdout-2026-v1:"+n).encode()).hexdigest())
        if len(ordered)<6:raise RuntimeError("Too few videos for camera-balanced split")
        nval=max(1,int(round(len(ordered)*.2)))
        ntest=nval
        result["test"]+=ordered[:ntest]
        result["calibration"]+=ordered[ntest:ntest+nval]
        result["train"]+=ordered[ntest+nval:]
    for k in result:result[k]=sorted(result[k])
    allids=sum(result.values(),[])
    if len(allids)!=len(names) or len(set(allids))!=len(names):
        raise RuntimeError("A source VIDEO occurred in multiple partitions")
    return result

def extract_one(z,name):
    with z.open(name) as f:
        df=pd.read_csv(f,header=None,names=FILE_COLS,dtype={
            "frame":"int32","object_type":"int16","track_id":"int32",
            "behavior":"int16","occluded":"int16","x1":"float32",
            "x2":"float32","y1":"float32","y2":"float32"})
    if df.empty:raise RuntimeError(f"No rows in {name}")
    if len(df.columns)!=9:raise RuntimeError(f"Wrong column count {name}")
    if not set(df.behavior.unique()).issubset(set(range(1,17))):
        raise RuntimeError(f"Unrecognized human behavior code in {name}")
    if not set(df.object_type.unique()).issubset({0,1}):
        raise RuntimeError("Source animal type code changed")
    if not set(df.occluded.unique()).issubset({0,1}):
        raise RuntimeError("Source occlusion status changed")
    if df.duplicated(["frame","track_id","object_type"]).any():
        raise RuntimeError(f"Duplicate time/track/object rows: {name}")
    total=len(df)
    quality={"all_human_rows":total,"max_frame":int(df.frame.max()),
             "pup_rows":int((df.object_type==1).sum()),
             "occluded_meerkat_rows":int(((df.object_type==0)&(df.occluded==1)).sum()),
             "unclassified_none_rows":int((df.behavior==16).sum()),
             "all_label_counts":{str(k):int(v) for k,v in df.behavior.value_counts().items()}}
    df=df[(df.object_type==0)&(df.occluded==0)&(df.behavior.between(1,15))]
    df=df[df.frame.mod(STEP)==0].copy()
    df.sort_values(["track_id","frame"],inplace=True)
    if (df.x2<=df.x1).any() or (df.y2<=df.y1).any():
        raise RuntimeError("Invalid positive-area human bounding boxes")
    df["cx"]=(df.x1+df.x2)*.0005
    df["cy"]=(df.y1+df.y2)*.0005
    df["w"]=(df.x2-df.x1)*.001
    df["h"]=(df.y2-df.y1)*.001
    df["aspect"]=df.w/(df.h+1e-6)
    df["log_area"]=np.log1p(df.w*df.h)
    gp=df.groupby("track_id",sort=False)
    prior=gp[["cx","cy","w","h","aspect","frame"]].shift(1)
    dt=(df.frame-prior["frame"]).to_numpy(float)
    valid=(dt>0)&(dt<=3*STEP)
    dt=np.where(valid,dt,STEP).astype(float)
    df["dx_per_frame"]=np.where(valid,(df.cx-prior.cx)/dt,0.)
    df["dy_per_frame"]=np.where(valid,(df.cy-prior.cy)/dt,0.)
    df["norm_speed"]=np.hypot(df.dx_per_frame,df.dy_per_frame)/(np.maximum(df.w,df.h)+1e-3)
    df["sample_gap"]=np.where(valid,dt/STEP,0.)
    df["prev_aspect"]=np.where(valid,prior.aspect,df.aspect)
    df["relative_scale_change"]=np.where(valid,np.log((df.w*df.h+1e-4)/(prior.w*prior.h+1e-4)),0.)
    X=df[FEATS].to_numpy(dtype=np.float32)
    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    if not np.isfinite(X).all():raise RuntimeError("Nonfinite geometry features")
    y=df.behavior.to_numpy(dtype=np.int16)
    quality.update({"sampled_visible_labeled_rows":len(df),
                    "sampled_behavior_counts":{str(k):int(v) for k,v in df.behavior.value_counts().items()},
                    "sampled_tracks":int(df.track_id.nunique()),
                    "noncausal_motion_used":False})
    return X,y,quality

def probabilities(model,X,classes):
    if model is None:
        raise RuntimeError("prior passed incorrectly")
    raw=model.predict_proba(X)
    p=np.zeros((len(X),len(ALL_LABELS)),float)+1e-9
    for col,label in enumerate(model.classes_):
        if int(label) in ALL_LABELS:p[:,ALL_LABELS.index(int(label))]=raw[:,col]
    p/=p.sum(axis=1,keepdims=True)
    return p

def metrics(y,probs):
    yp=np.asarray(ALL_LABELS)[np.argmax(probs,axis=1)]
    prec,rec,f1,support=precision_recall_fscore_support(y,yp,labels=ALL_LABELS,zero_division=0)
    return {"n":len(y),"accuracy":float(accuracy_score(y,yp)),
            "balanced_accuracy":float(balanced_accuracy_score(y,yp)),
            "macro_f1_all_15":float(f1_score(y,yp,labels=ALL_LABELS,average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(y,yp,labels=ALL_LABELS,average="weighted",zero_division=0)),
            "log_loss":float(log_loss(y,np.clip(probs,1e-9,1.),labels=ALL_LABELS)),
            "confusion_gt_rows_pred_cols":confusion_matrix(y,yp,labels=ALL_LABELS).tolist(),
            "behavior_by_id":{str(c):{"precision":float(prec[i]),"recall":float(rec[i]),
                                     "f1":float(f1[i]),"support":int(support[i]),
                                     "predicted":int((yp==c).sum())}
                                 for i,c in enumerate(ALL_LABELS)}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--cache",type=Path,default=None)
    a=ap.parse_args()
    # cache content in memory: the original zip is just 25MB
    z,sha=archive()
    names=sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
    if len(names)!=20:raise RuntimeError(f"Expected 20 distinct source videos, got {len(names)}")
    b=json.loads(z.read("behaviours.json"))
    label_map={int(idx):label for label,idx in b.items()}
    if set(label_map)!=set(range(1,17)):raise RuntimeError("Changed annotated ethogram codebook")
    split=groups(names)
    obs={};audit={}
    for n in names:
        obs[n]=extract_one(z,n)[:2]
        audit[n]=extract_one(z,n)[2] if False else None
    # Process each video exactly once below; avoid rereading 20 large CSVs.
    # First-pass source is already computed in obs; audit summary from sampled rows.
    for n in names:
        X,y=obs[n]
        audit[n]={"sampled_visible_labeled_rows":len(y),
                  "sampled_behavior_counts":dict(collections.Counter(str(int(c)) for c in y))}
    # Collate independent whole-video partitions; no within-video shuffling.
    data={}
    for fold,vids in split.items():
        data[fold]=(np.concatenate([obs[v][0] for v in vids]),
                    np.concatenate([obs[v][1] for v in vids]))
    trainX,trainy=data["train"]
    output={"source":SOURCE,"sha256":sha,
      "source_rights":"CC BY-NC-SA 4.0; source hosted by University of Auckland, intended noncommercial research only.",
      "task":"Human ethogram Choice(15 classes) from current+pasts of manually annotated bounding boxes",
      "model_visible_features":FEATS,
      "excluded_inputs":["frame","track_id","video","date","camera","human_behavior_code","object_type","occluded"],
      "partition_unit":"whole independent recordings/videos; same zoo/animals may recur across recordings",
      "sample_every_nth_frame":STEP,"screening":"meerkat only; visible only; excludes None/unknown behavior",
      "fold_video_ids":split,"fold_n":{key:len(v) for key,v in split.items()},
      "fold_samples":{fold:len(y) for fold,(X,y) in data.items()},
      "source_video_audit":audit,
      "labels":{str(k):v for k,v in label_map.items()},
      "models":{},"selection":"Best calibration macroF1; all model comparisons reported on test, fixed folds.",
      "human_coder_reliability":"UNKNOWN: one released consensus code per frame; this dataset doesn't supply triple-annotator comparison.",
      "deployment":"No raw video inference, no proof of clinical welfare states, no cross-enclosure generalization.",
      "candidate_promotions":[]}
    prior=np.array([(trainy==c).mean() for c in ALL_LABELS])
    model_specs={
      "train_prior":None,
      "logistic_static_bbox":make_pipeline(StandardScaler(),LogisticRegression(C=.35,max_iter=180,random_state=17)),
      "logistic_bbox_motion":make_pipeline(StandardScaler(),LogisticRegression(C=.35,max_iter=180,random_state=17)),
      "tiny_extra_trees_motion":ExtraTreesClassifier(n_estimators=55,max_depth=11,min_samples_leaf=35,max_features=.85,n_jobs=2,random_state=17)
    }
    def column_slice(name):
        return slice(0,len(FEAT_STATIC)) if name=="logistic_static_bbox" else slice(None)
    for name,model in model_specs.items():
        xsel=column_slice(name)
        if model is not None:model.fit(trainX[:,xsel],trainy)
        output["models"][name]={}
        for fold in ("calibration","test"):
            x,y=data[fold]
            pred=(np.tile(prior,(len(y),1))
                  if model is None else probabilities(model,x[:,xsel],ALL_LABELS))
            output["models"][name][fold]=metrics(y,pred)
            # Need independent-video results, not only aggregate correlated frames.
            output["models"][name][fold]["individual_video_accuracy"]={}
            start=0
            for v in split[fold]:
                n=len(obs[v][1])
                output["models"][name][fold]["individual_video_accuracy"][v]=float(accuracy_score(obs[v][1],np.asarray(ALL_LABELS)[pred[start:start+n].argmax(axis=1)]))
                start+=n
    chosen=max(model_specs,key=lambda name:output["models"][name]["calibration"]["macro_f1_all_15"])
    output["selected_model"]=chosen
    test=output["models"][chosen]["test"]["behavior_by_id"]
    # Predeclared promotion rule about *recoverability from bounding boxes*,
    # not human agreement or valence: at least 2 separate test videos with
    # >=3 train videos; minimum 50 test positives, F1>=.55, P>=.6, R>=.5.
    for c in ALL_LABELS:
        testvideos=sum(np.any(obs[v][1]==c) for v in split["test"])
        trainvideos=sum(np.any(obs[v][1]==c) for v in split["train"])
        v=test[str(c)]
        yes=(testvideos>=2 and trainvideos>=3 and v["support"]>=50 and
             v["f1"]>=.55 and v["precision"]>=.6 and v["recall"]>=.5)
        output["candidate_promotions"].append({
            "behavior":label_map[c],"id":c,"test_positive_videos":testvideos,
            "train_positive_videos":trainvideos,**v,
            "observability_status":"RECOVERABLE_PROBE" if yes else "NOT_YET_RELIABLY_RECOVERABLE",
            "welfare_affect_ground_truth":"NOT AVAILABLE",
            "annotation_interrater_reliability":"NOT MEASURED"})
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(output,indent=2)+"\n")
    print("MEERKAT_WHOLE_VIDEO_BENCH:"+json.dumps({
        "source_sha256":sha,"videos":output["fold_n"],"samples":output["fold_samples"],
        "selected":chosen,"models":{k:{"cal_mf1":v["calibration"]["macro_f1_all_15"],
                                      "test_mf1":v["test"]["macro_f1_all_15"],
                                      "test_acc":v["test"]["accuracy"]}
                                 for k,v in output["models"].items()},
        "behavior_decisions":output["candidate_promotions"]}))
if __name__=="__main__":main()
