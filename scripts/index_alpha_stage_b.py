#!/usr/bin/env python3
"""OpenFARM Stage-B: matched two-species, two-task shared-decision falsification.

Inputs: exact original-author camera-derived count channels (HotPig) and MI
(wearable cattle). Heads predict binary actual feeder reading, or four-way
human-observed behavior. NO welfare emotion, clinical inference or raw pixels.

Both tasks reuse historically examined developmental animals. No heldout data
is used for model selection. Trial does NOT claim cross-species knowledge
transfer: source-specific input adapters and output heads are inevitable.
"""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from benchmark_hotpig_system1 import source as hp_source,load_pig,combine,choose_threshold
from build_public_cattle_behavior_v0 import source as cow_source,prepare as cow_prepare,groups as cow_groups,BEHAVIORS
from evaluate_hotpig_feeder_candidate import FEATURES

SEED=20261019
EPOCHS=6
STEPS_PER_EPOCH=180
BATCH_HP=256
BATCH_COW=128

def fixed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(2)
    try:torch.use_deterministic_algorithms(True)
    except RuntimeError:pass

class Decision(nn.Module):
    def __init__(self,shared):
        super().__init__()
        self.adapters=nn.ModuleDict({
          "hotpig":nn.Sequential(nn.Linear(12,24),nn.ReLU()),
          "cattle":nn.Sequential(nn.Linear(1,24),nn.ReLU())})
        def trunk():return nn.Sequential(nn.Linear(24,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.trunks=nn.ModuleDict({"common":trunk()} if shared
                                  else {"hotpig":trunk(),"cattle":trunk()})
        self.heads=nn.ModuleDict({"hotpig":nn.Linear(16,1),"cattle":nn.Linear(16,4)})
        self.shared=shared
    def forward(self,domain,x):
        v=self.adapters[domain](x)
        return self.heads[domain](self.trunks["common" if self.shared else domain](v))

def softmax(z):
    z=np.asarray(z,dtype=float)
    z-=z.max(axis=1,keepdims=True)
    e=np.exp(z);return e/e.sum(axis=1,keepdims=True)

def brier(y,p):
    if p.ndim==1:return float(np.mean((np.asarray(y,float)-p)**2))
    return float(np.mean(np.sum((np.eye(p.shape[1])[np.asarray(y,int)]-p)**2,axis=1)))

def preprocess(xtr,xc,xt,logarithm=False):
    if logarithm:
        xtr=np.log1p(np.maximum(0,xtr))
        xc=np.log1p(np.maximum(0,xc))
        xt=np.log1p(np.maximum(0,xt))
    s=StandardScaler().fit(xtr)
    return [np.clip(s.transform(q),-8,8).astype(np.float32) for q in (xtr,xc,xt)]

def prepare():
    z,sha=hp_source()
    expected="1ba5125167224b969281509b88ef7b3cba121114aeb0532f3903894c796df7e8"
    assert sha==expected,"HOTPIG source hash shift"
    hp={key:[load_pig(z,i) for i in ids] for key,ids in
        {"train":range(1,17),"cal":range(17,21),"test":range(21,25)}.items()}
    # For cattle use precisely the existing 16/4/4 cow-disjoint protocol.
    orig,meta=cow_source()
    assert meta["file_sha256"]=="b4d6e14d6ff029dbf497287a774aaa27ec471e7ed1ef5884e9cf0207c784fe37"
    cows,integrity=cow_prepare(orig)
    cs=cow_groups(cows)["breed_balanced_group_holdout"]
    assert len(set(sum(cs.values(),[])))==24
    ca={}
    for k,k2 in (("train","train"),("cal","calibration"),("test","test")):
        d=cows[cows.ID.isin(cs[k2])]
        ca[k]={"X":d[["MI"]].to_numpy(float),
               "y":np.asarray([BEHAVIORS.index(v) for v in d["Behavior"]],dtype=np.int64),
               "id":d["ID"].to_numpy()}
    hp_xy={k:{"X":combine(v)[0],"y":combine(v)[1]} for k,v in hp.items()}
    for k in hp_xy:
        assert np.isfinite(hp_xy[k]["X"]).all()
    if [len(hp_xy[k]["X"]) for k in ("train","cal","test")]!=[368508,90150,92160]:
        raise RuntimeError("HotPig per-source row count shift")
    assert len(cows)==9222 and len(ca["test"]["X"])>0
    for domain,parts in (("hotpig",hp_xy),("cattle",ca)):
        xtr,xcal,xte=preprocess(parts["train"]["X"],parts["cal"]["X"],parts["test"]["X"],domain=="hotpig")
        for k,xx in zip(("train","cal","test"),(xtr,xcal,xte)):parts[k]["scaled_X"]=xx
    assert list(hp["test"][0]["X"].shape)[1]==12
    return hp, hp_xy, ca,cs

def predict(net,task,x):
    net.eval()
    chunks=[]
    with torch.no_grad():
        for part in np.array_split(x,max(1,int(np.ceil(len(x)/4096)))):
            z=net(task,torch.from_numpy(np.ascontiguousarray(part))).numpy()
            if task=="hotpig":chunks.append(torch.sigmoid(torch.from_numpy(z[:,0])).numpy())
            else:chunks.append(softmax(z))
    return np.concatenate(chunks)

def fit_net(shared,hp,ca):
    fixed()
    net=Decision(shared)
    opt=torch.optim.AdamW(net.parameters(),lr=0.002,weight_decay=0.001)
    rng=np.random.default_rng(SEED)
    xh=torch.from_numpy(hp["train"]["scaled_X"]);yh=torch.from_numpy(hp["train"]["y"].astype(np.float32))
    xc=torch.from_numpy(ca["train"]["scaled_X"]);yc=torch.from_numpy(ca["train"]["y"].astype(np.int64))
    # Equal-weight source losses, not 55:1 weighting by minute frequency.
    # Stop at epoch minimizing equal PRIOR-NORMALIZED calibration Brier.
    hp_prior=float(hp["train"]["y"].mean())
    cow_freq=np.bincount(ca["train"]["y"],minlength=4)/len(ca["train"]["y"])
    den_h=brier(hp["cal"]["y"],np.full(len(hp["cal"]["y"]),hp_prior))
    den_c=brier(ca["cal"]["y"],np.tile(cow_freq,(len(ca["cal"]["y"]),1)))
    selected=None;trajectory=[]
    begin=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        net.train()
        for step in range(STEPS_PER_EPOCH):
            ih=torch.from_numpy(rng.integers(0,len(xh),size=BATCH_HP))
            ic=torch.from_numpy(rng.integers(0,len(xc),size=BATCH_COW))
            zh=net("hotpig",xh[ih]).flatten();zc=net("cattle",xc[ic])
            loss=nn.functional.binary_cross_entropy_with_logits(zh,yh[ih])+nn.functional.cross_entropy(zc,yc[ic])
            opt.zero_grad();loss.backward();opt.step()
        ph=predict(net,"hotpig",hp["cal"]["scaled_X"])
        pc=predict(net,"cattle",ca["cal"]["scaled_X"])
        bh=brier(hp["cal"]["y"],ph)
        bc=brier(ca["cal"]["y"],pc)
        norm=(bh/den_h+bc/den_c)/2
        trajectory.append({"epoch":epoch,"cal_hotpig_brier":bh,"cal_cattle_brier":bc,
                           "equal_normalized_brier":norm})
        if selected is None or norm<selected["norm"]:
            selected={"epoch":epoch,"norm":norm,
                      "state":{k:v.detach().clone() for k,v in net.state_dict().items()}}
    net.load_state_dict(selected["state"])
    return net,{"parameters":sum(p.numel() for p in net.parameters()),
                "training_seconds":round(time.perf_counter()-begin,2),
                "cal_selected_epoch":selected["epoch"],"calibration_history":trajectory}

def score_hp(y,p,pig_info):
    ph=np.concatenate([p for p in p]) if isinstance(p,list) else np.asarray(p)
    scores={"n":int(len(y)),"brier":brier(y,ph),"positive_rate":float(np.mean(y))}
    # Predictions index preserves exact pig source order.
    ix=0;ind=[]
    for pig in pig_info:
        n=len(pig["y"]);pr=ph[ix:ix+n];ix+=n
        ind.append({"pig":pig["id"],"n":n,"brier":brier(pig["y"],pr)})
    assert ix==len(y)
    scores["per_group"]=ind
    return scores
def score_cow(y,p,ids):
    out={"n":len(y),"brier":brier(y,p),
         "argmax_accuracy":float((p.argmax(axis=1)==y).mean()),
         "macro_f1":float(f1_score(y,p.argmax(axis=1),labels=[0,1,2,3],average="macro",zero_division=0))}
    out["per_group"]=[{"cow":str(g),"n":int((ids==g).sum()),
                        "brier":brier(y[ids==g],p[ids==g])} for g in sorted(set(ids))]
    return out

def group_diff(a,b,group_key):
    first={r[group_key]:r["brier"] for r in a["per_group"]}
    second={r[group_key]:r["brier"] for r in b["per_group"]}
    if first.keys()!=second.keys():raise ValueError("missing independent groups")
    return {"groups_better":sum(first[g]<second[g] for g in first),
            "groups_total":len(first),
            "per_group_relative_improvement":[{"group":str(g),"improvement":second[g]-first[g]}
                                              for g in sorted(first)]}

def strong_baselines(hp,hpraw,ca,cs):
    train_hp=hpraw["train"]["X"];y_hp=hpraw["train"]["y"]
    train_c=ca["train"]["X"];y_c=ca["train"]["y"]
    models={
      "hotpig":{
        "train_prior":None,
        "logistic":make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,C=1,random_state=17)),
        "tiny_tree":HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=10,max_depth=4,
           min_samples_leaf=350,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=17)},
      "cattle":{
        "train_prior":None,
        "logistic":make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=1,random_state=17)),
        "tiny_tree":HistGradientBoostingClassifier(max_iter=75,max_depth=3,max_leaf_nodes=7,
            min_samples_leaf=30,learning_rate=.08,l2_regularization=2.,
            random_state=17,early_stopping=False)}
    }
    out={"hotpig":{},"cattle":{}}
    for task,train,y in (("hotpig",train_hp,y_hp),("cattle",train_c,y_c)):
        for name,model in models[task].items():
            if model is not None:model.fit(train,y)
            if task=="hotpig":
                probs=(np.full(len(hpraw["test"]["X"]),float(y.mean()))
                       if model is None else model.predict_proba(hpraw["test"]["X"])[:,1])
                metrics=score_hp(hpraw["test"]["y"],probs,hp["test"])
            else:
                prior=np.bincount(y,minlength=4)/len(y)
                probs=(np.tile(prior,(len(ca["test"]["X"]),1)) if model is None
                       else model.predict_proba(ca["test"]["X"])[:,[list(model.classes_).index(i) for i in range(4)]])
                metrics=score_cow(ca["test"]["y"],probs,ca["test"]["id"])
            out[task][name]=metrics
    out["hotpig"]["camera_eating_rule"]=score_hp(hpraw["test"]["y"],
        np.concatenate([p["rule"] for p in hp["test"]]).astype(float),hp["test"])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("reports/index-alpha/stage-b-source-matched.json"))
    a=ap.parse_args()
    fixed()
    hp,hp_xy,ca,cs=prepare()
    baselines=strong_baselines(hp,hp_xy,ca,cs)
    models={}
    for label,shared in (("separate_neural_specialists",False),("shared_trunk_two_task",True)):
        net,train_info=fit_net(shared,hp_xy,ca)
        metrics={}
        for t,ds in (("hotpig",hp_xy),("cattle",ca)):
            x=ds["test"]["scaled_X"]
            ts=time.perf_counter();p=predict(net,t,x)
            elapsed=time.perf_counter()-ts
            metrics[t]=(score_hp(ds["test"]["y"],p,hp["test"]) if t=="hotpig"
                        else score_cow(ds["test"]["y"],p,ca["test"]["id"]))
            metrics[t]["inference_ms_per_1000_rows"]=round(1000*elapsed/len(x)*1000,5)
        models[label]={"training":train_info,"tasks":metrics}
    # PREDECLARED blocking success gate:
    # shared beats best standalone trained comparator (including own separate neural)
    # by >=2% relative pooled Brier in BOTH domains; improved on >=3/4 test
    # individuals in each domain. No claim that 4 animals provide external power.
    comparison={};success=True
    for task,key in (("hotpig","pig"),("cattle","cow")):
        specialists={name:r for name,r in baselines[task].items() if name in ("logistic","tiny_tree")}
        specialists["separate_neural"]=models["separate_neural_specialists"]["tasks"][task]
        best_name=min(specialists,key=lambda n:specialists[n]["brier"])
        best=specialists[best_name]
        sh=models["shared_trunk_two_task"]["tasks"][task]
        gain=(best["brier"]-sh["brier"])/best["brier"]
        grp=group_diff(sh,best,key)
        ok=gain>=.02 and grp["groups_better"]>=3
        comparison[task]={"best_specialist_name":best_name,
                          "best_specialist_brier":best["brier"],
                          "shared_brier":sh["brier"],
                          "relative_brier_gain":gain,
                          **grp,"passes_2pct_and_3_of_4":ok}
        success &=ok
    output={
      "version":"stage-b-2026-10-19-0.1",
      "scientific_status":"SHARED_MODEL_PASS_DEVELOPMENTAL" if success else "BLOCKED_NO_EVIDENCE_SHARED_MODEL_BEATS_SPECIALISTS",
      "predeclared_success":"Shared trunk reduces pooled heldout Brier >=2% versus best fitted specialist in BOTH tasks, and improves >=3 of 4 animals per task; these are DEVELOPMENTAL, not fresh blind scores.",
      "sources":{"hotpig_sha":"1ba5125167224b969281509b88ef7b3cba121114aeb0532f3903894c796df7e8",
                 "cattle_sha":"b4d6e14d6ff029dbf497287a774aaa27ec471e7ed1ef5884e9cf0207c784fe37"},
      "task_semantics":{"hotpig":"binary feeder-positive from 12 machine-derived counts",
                        "cattle":"4-class human-behavior from one author wearable scalar"},
      "excluded":"patient/source IDs, actual feeder input, source behavior label, clock, source camera/condition context; test groups never train or calibrate",
      "original_splits":{"hotpig":{"train":list(range(1,17)),"calibration":list(range(17,21)),
                                  "test":list(range(21,25))},
                         "cattle":cs},
      "comparison_notes":[
        "All models consume identical original task-visible channels, but neural models transform count scales using train-fitted log1p/scaler.",
        "Shared model has domain-specific input adapters and output heads, only trunk weights shared. This does NOT prove zero-shot transfer or a domain-universal representation.",
        "Cattle behavior is human observed; feeding target is feeder measured; neither is a validated pain/distress/clinical care task.",
        "Only 4 heldout animals per task. Time autocorrelation extremely high in HotPig, trials already repeatedly examined; no sealed blind data.",
        "No matched open generative foundation model arm: existing Liquid video/clinical runs use different modality/state and cannot be compared fairly here.",
        "No time-series future information provided; no raw video learned. Tiny specialists remain valid default until independent transfer evidence."
      ],
      "baselines":baselines,"trained":models,"stage_b_gate":comparison}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2,default=str)+"\n")
    print("OPENFARM_STAGE_B_GATE",json.dumps({"status":output["scientific_status"],
      "baselines":{t:{n:round(v["brier"],5) for n,v in x.items()} for t,x in baselines.items()},
      "trained":{k:{t:round(v["brier"],5) for t,v in x["tasks"].items()} for k,x in models.items()},
      "comparison":comparison},sort_keys=True),flush=True)
if __name__=="__main__":main()
