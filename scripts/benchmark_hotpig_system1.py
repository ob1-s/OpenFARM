"""Track A — honest, very-cheap System-1 baselines on validated HotPig source.

Compare a ZERO-training detector rule vs regularized logistic regression and
small gradient-boosted trees. Task: independent electronic feeder detected
intake in CURRENT minute; inputs: 12 author-precomputed machine-camera counts.

Whole pigs held out 16/4/4 train/validation/test. Never fit preprocessing,
thresholds, calibrator or model parameters on test pigs. Reports by-pig
uncertainty and episode/lag diagnostics. False-positive *minutes* are not
welfare alerts. Source rights: Zenodo 17090997 CC BY 4.0 (attribute authors).
"""
from __future__ import annotations
import argparse,collections,hashlib,io,json,math,random,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score,roc_auc_score,
                             brier_score_loss,log_loss)
from evaluate_hotpig_feeder_candidate import source_bytes,FEATURES

SPLITS={"train":range(1,17),"calibration":range(17,21),"test":range(21,25)}
SCORE_VERSION="0.1"
def source():
    b=source_bytes()      # pinned SHA via reused audited downloader
    return zipfile.ZipFile(io.BytesIO(b)),hashlib.sha256(b).hexdigest()

def load_pig(z,pig):
    f=pd.read_csv(z.open(f"series/P{pig}.csv"))
    if len(f)!=23040:raise RuntimeError(f"Unexpected pig {pig} row count")
    numeric=f[FEATURES+["feed"]].apply(pd.to_numeric,errors="coerce")
    finite=np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    good=numeric.loc[finite]
    X=good[FEATURES].to_numpy(dtype=np.float32)
    y=(good["feed"].to_numpy(dtype=np.float32)>0).astype(np.int8)
    source_minute=np.flatnonzero(finite)
    return {"id":pig,"X":X,"y":y,"minute":source_minute,
            "rule":(good["eating"].to_numpy(dtype=np.float32)>0).astype(np.int8)}

def combine(pigs):
    return np.concatenate([p["X"] for p in pigs]),np.concatenate([p["y"] for p in pigs])

def scores(y,p,threshold):
    y=np.asarray(y,dtype=bool)
    predict=np.asarray(p)>=threshold
    tp=int(np.logical_and(predict,y).sum());fp=int(np.logical_and(predict,~y).sum())
    fn=int(np.logical_and(~predict,y).sum());tn=int(np.logical_and(~predict,~y).sum())
    return {"rows":len(y),"base_rate":round(float(y.mean()),6),
            "precision":round(tp/max(1,tp+fp),5),"recall":round(tp/max(1,tp+fn),5),
            "f1":round(2*tp/max(1,2*tp+fp+fn),5),
            "specificity":round(tn/max(1,tn+fp),5),
            "true_positive_minutes":tp,"false_positive_minutes":fp,
            "false_negative_minutes":fn,
            "false_positive_minutes_per_pig_day":None}

def runs(values,minute):
    """Runs of positive predicted minutes, split at *actual* source minute gaps."""
    values=np.asarray(values,dtype=bool)
    minute=np.asarray(minute,dtype=int)
    if not len(minute):return []
    starts=np.flatnonzero(values&np.r_[True,~values[:-1]|(np.diff(minute)!=1)])
    ends=np.flatnonzero(values&np.r_[~values[1:]|(np.diff(minute)!=1),True])
    if len(starts)!=len(ends):raise AssertionError("Run pairing")
    return [(int(minute[s]),int(minute[e])) for s,e in zip(starts,ends)]

def episode_metrics(pig,prob,threshold):
    y=pig["y"];minute=pig["minute"];pred=np.asarray(prob)>=threshold
    observed=runs(y,minute);guessed=runs(pred,minute)
    def close(a,b,tol):
        return a[0]<=b[1]+tol and b[0]<=a[1]+tol
    result={"real_feeder_episodes":len(observed),"predicted_episodes":len(guessed)}
    for tol in (0,2,5):
        fp=sum(not any(close(g,t,tol) for t in observed) for g in guessed)
        fn=sum(not any(close(t,g,tol) for g in guessed) for t in observed)
        result[f"unmatched_pred_episodes_{tol}min"]=fp
        result[f"unmatched_true_episodes_{tol}min"]=fn
    return result

def evaluate_pigs(pigs,probs,thr,probabilistic):
    individual=[]
    for pig,p in zip(pigs,probs):
        y=pig["y"]
        stats=scores(y,p,thr)
        stats["false_positive_minutes_per_pig_day"]=round(
            stats["false_positive_minutes"]/(len(y)/1440),4)
        eps=episode_metrics(pig,p,thr)
        stats.update(eps)
        if probabilistic:
            p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
            stats["brier"]=round(float(brier_score_loss(y,p)),7)
            stats["log_loss"]=round(float(log_loss(y,p,labels=[0,1])),7)
            stats["avg_precision"]=round(float(average_precision_score(y,p)),7)
            if len(set(y))==2:stats["auroc"]=round(float(roc_auc_score(y,p)),7)
        individual.append({"pig":pig["id"],**stats})
    total_y=np.concatenate([p["y"] for p in pigs])
    total_p=np.concatenate([np.asarray(p) for p in probs])
    pooled=scores(total_y,total_p,thr)
    pooled["false_positive_minutes_per_pig_day"]=round(
        pooled["false_positive_minutes"]/(len(total_y)/1440),4)
    for key in ("unmatched_pred_episodes_0min","unmatched_pred_episodes_2min",
                "unmatched_pred_episodes_5min","real_feeder_episodes",
                "predicted_episodes"):
        pooled[key]=int(sum(x[key] for x in individual))
    for key in ("precision","recall","f1","false_positive_minutes_per_pig_day"):
        pooled[f"mean_per_pig_{key}"]=round(float(np.mean([r[key] for r in individual])),5)
    if probabilistic:
        p=np.clip(total_p.astype(float),1e-7,1-1e-7)
        pooled.update({
          "brier":round(float(brier_score_loss(total_y,p)),7),
          "log_loss":round(float(log_loss(total_y,p,labels=[0,1])),7),
          "avg_precision":round(float(average_precision_score(total_y,p)),7),
          "auroc":round(float(roc_auc_score(total_y,p)),7)
        })
        for bins in (10,):
            edges=np.linspace(0,1,bins+1)
            calibration=[]
            for a,b in zip(edges[:-1],edges[1:]):
                mask=(total_p>=a)&((total_p<b) if b<1 else (total_p<=b))
                if mask.sum():
                    calibration.append({"count":int(mask.sum()),
                      "mean_prediction":round(float(total_p[mask].mean()),5),
                      "observed_frequency":round(float(total_y[mask].mean()),5)})
            pooled["reliability_bins"]=calibration
    return {"pooled":pooled,"by_pig":individual}

def choose_threshold(pigs,probs):
    y=np.concatenate([p["y"] for p in pigs])
    p=np.concatenate(probs)
    # Cal-only tuning. Explicitly report that four calibration pigs are tiny.
    candidates=[round(x,3) for x in np.arange(.05,.951,.025)]
    ranked=[]
    for t in candidates:
        m=scores(y,p,t)
        ranked.append((m["f1"],-abs(t-.5),t))
    return max(ranked)[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    z,sha=source()
    split={name:[load_pig(z,i) for i in ids] for name,ids in SPLITS.items()}
    x,y=combine(split["train"])
    models={
      "logistic_standardized_L2":make_pipeline(
         StandardScaler(),LogisticRegression(max_iter=300,C=1.0,random_state=17)),
      "tiny_gradient_boosted_tree":HistGradientBoostingClassifier(
          max_iter=90,max_leaf_nodes=10,max_depth=4,min_samples_leaf=350,
          learning_rate=.1,l2_regularization=1.,early_stopping=False,
          random_state=17)
    }
    result={
      "source":"https://zenodo.org/records/17090997",
      "source_zip_sha256":sha,"scope":"Instrumental feeder event from author camera-derived features; not welfare nor stress.",
      "features":FEATURES,
      "time_unit":"individual 60-second readings; samples strongly correlated",
      "source_group_sizes":{k:[p["id"] for p in v] for k,v in split.items()},
      "valid_counts":{k:sum(len(p["y"]) for p in v) for k,v in split.items()},
      "calibration_policy":"No probability calibration applied; calibration/reliability measured descriptively, four calibration pigs insufficient for robust sigmoid/isotonic tuning.",
      "rule":{"test":evaluate_pigs(split["test"],[p["rule"] for p in split["test"]],.5,False)},
      "models":{},
      "limitations":["Only one trial/site, source author computer-vision detectors are a pretrained perception input.",
       "No independent emotion or welfare-related outcome; feeder-positive is sensor reference.",
       "False positive minutes and episode counts are not welfare alert rates.",
       "Calibration/test confidence must cluster by pig (4 subjects each), not minutes.",
       "Lag tolerance is only a diagnostic; 5 minutes is not necessarily an allowed delay for monitoring."]
    }
    for name,model in models.items():
        model.fit(x,y)
        cal=[model.predict_proba(p["X"])[:,1] for p in split["calibration"]]
        test=[model.predict_proba(p["X"])[:,1] for p in split["test"]]
        threshold=choose_threshold(split["calibration"],cal)
        result["models"][name]={
          "chosen_threshold_from_calibration":threshold,
          "calibration_at_0_5":evaluate_pigs(split["calibration"],cal,.5,True),
          "test_at_0_5":evaluate_pigs(split["test"],test,.5,True),
          "test_at_calibration_threshold":evaluate_pigs(split["test"],test,threshold,True),
          "parametric_model_size_bytes_approx":None
        }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("HOTPIG_SYSTEM1_BASELINES:"+json.dumps({
      "source_sha":sha,
      "rule_test":result["rule"]["test"]["pooled"],
      "learned":{k:{"threshold":v["chosen_threshold_from_calibration"],
                    "test_0_5":v["test_at_0_5"]["pooled"],
                    "test_selected":v["test_at_calibration_threshold"]["pooled"]}
                 for k,v in result["models"].items()}
    },default=str))
if __name__=="__main__":main()
