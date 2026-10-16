"""Very-cheap baselines for source-real pen-level first-treatment research pilot.

IMPORTANT: Research-only, CC BY-NC original, Sweden selected case-control
42 cohorts. No population true alert rate, no clinical recommendation,
no cross-site generalization. Positive examples are correlated with original
17 separately documented first-treatment event cohorts, not 17 independent
daily welfare-event datasets.
"""
from __future__ import annotations
import argparse,collections,json,math,tempfile
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score,average_precision_score,
    brier_score_loss,log_loss,precision_recall_fscore_support)
from build_swedish_tail_damage_historical import build

def rows(folder,split):
    d=folder/"swedish_tail_treatment_case_control"/split
    def parse(fname):
        with (d/fname).open() as f:return [json.loads(l) for l in f]
    states=parse("states.jsonl")
    answers={x["state_id"]:x for x in parse("references.jsonl")}
    provenance={x["state_id"]:x for x in parse("provenance.jsonl")}
    result=[]
    for s in states:
        q=answers[s["state_id"]]["answers"]
        if len(q)!=1 or not q[0]["valid"]:raise RuntimeError("Missing measured future treatment target")
        p=provenance[s["state_id"]]
        result.append((s["features"],int(q[0]["value"]),p["group_id"]))
    return result

def measures(rows,prob,threshold=.5):
    y=np.array([r[1] for r in rows],dtype=int)
    p=np.clip(np.asarray(prob,dtype=float),1e-8,1-1e-8)
    decisions=p>=threshold
    res={"windows":len(y),"true_future_treatment_windows":int(y.sum()),
         "observed_positive_pen_cohorts":len({r[2] for r in rows if r[1]==1}),
         "independent_source_cohorts":len({r[2] for r in rows}),
         "positive_rate_in_selected_windows":float(y.mean()),
         "brier":float(brier_score_loss(y,p)),
         "log_loss":float(log_loss(y,p,labels=[0,1])),
         "precision":float(y[decisions].mean()) if decisions.sum() else None,
         "recall":float(decisions[y==1].mean()) if y.sum() else None,
         "predicted_positive_windows":int(decisions.sum())}
    if y.min()!=y.max():
        res["auroc"]=float(roc_auc_score(y,p))
        res["average_precision"]=float(average_precision_score(y,p))
    return res

def score():
    with tempfile.TemporaryDirectory() as temp:
        loc=Path(temp)
        compiled,_=build(loc)
        tr=rows(loc,"train");cal=rows(loc,"calibration");te=rows(loc,"test")
    keys=sorted(tr[0][0])
    def X(records):return np.array([[float(r[0][k]) for k in keys] for r in records],dtype=float)
    x,y=X(tr),np.array([r[1] for r in tr])
    if len(set(y))!=2:raise RuntimeError("Train source missing positive treatment outcomes")
    result={
        "source":"https://zenodo.org/records/5028016",
        "scope":"Sweden-only selected noncommercial case-control, group-heldout, NOT population alert eval",
        "input_features":keys,
        "compiler":{k:compiled["study_audit"][k] for k in ("known_cohorts",
                  "actual_source_treatment_events","cohorts_with_positive_examples",
                  "states_with_true_next7_treatment","no_pre_event_cohorts")},
        "partitions":{
            name:{"windows":len(rr),"positive_windows":sum(r[1] for r in rr),
                "pen_cohorts":len({r[2] for r in rr}),
                "positive_pen_cohorts":len({r[2] for r in rr if r[1]==1})}
            for name,rr in (("train",tr),("calibration",cal),("test",te))},
        "metrics":{},
        "limitations":[
            "A Swedish-only study with 42 selected case/control pen cohorts is not representative ongoing surveillance.",
            "The positive event reflects observed TAIL-DAMAGE TREATMENT, not independent evidence when to alert.",
            "Unobserved right-censoring after case/control selection and source pseudo-control dates make prevalence artificial.",
            "No test-farm transfer, no animal-outcome rater disagreements; counts are original cohorts, not daily events.",
            "The wider pooled data show a mismatched 28 source TB pens vs 27 in paper; Swiss subset deliberately not used.",
            "Threshold and model evaluation exploratory, NOT sealed; no genuine false alarms per pen-day claim."
        ]
    }
    prior=float(y.mean())
    result["metrics"]["train_prior"]={"test":measures(te,[prior]*len(te)),
                                       "calibration":measures(cal,[prior]*len(cal))}
    models={
        "regularized_logistic":make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=300,random_state=20261016)),
        "small_random_forest":RandomForestClassifier(n_estimators=60,max_depth=3,min_samples_leaf=5,random_state=20261016)
    }
    for name,model in models.items():
        model.fit(x,y)
        pc=model.predict_proba(X(cal))[:,1]
        pt=model.predict_proba(X(te))[:,1]
        result["metrics"][name]={"calibration":measures(cal,pc),"test":measures(te,pt)}
    return result

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    r=score()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2)+"\n")
    print("TAIL_TREATMENT_RESEARCH_BASELINES:"+json.dumps({
        "partitions":r["partitions"],"metrics":r["metrics"],
        "limitations":r["limitations"]}))
