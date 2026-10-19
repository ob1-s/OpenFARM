#!/usr/bin/env python3
"""Two-authenticated-source Stage B decision gate. Developmental, NOT blind.

HotPig binary feeder intake versus 12 camera-derived counts; Brazilian cow
binary locomotion >=3 versus breed/parity/days in milk. Shared versus separate
small nonautoregressive neural trunks and matched cheap classical specialists.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from benchmark_hotpig_system1 import source as hot_source,load_pig,combine
from evaluate_hotpig_feeder_candidate import FEATURES
from audit_brazil_cow_welfare import ingest

SEED=20261019
H_SHA="1ba5125167224b969281509b88ef7b3cba121114aeb0532f3903894c796df7e8"
C_SHA="3b345fa0a0c2495aab40e313b5703c7e7c5d0688393a1759684b801179e6b529"
def reset():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
def brier(y,p):return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def data():
    z,sha=hot_source()
    assert sha==H_SHA
    pigs={k:[load_pig(z,int(i)) for i in ids] for k,ids in
          {"train":range(1,17),"cal":range(17,21),"test":range(21,25)}.items()}
    hot={}
    for k,ps in pigs.items():
        x,y=combine(ps)
        hot[k]={"X":x.astype(np.float32),"y":y.astype(np.float32),
                "group":np.concatenate([np.full(len(p["y"]),p["id"]) for p in ps])}
    assert [len(hot[k]["X"]) for k in ("train","cal","test")]==[368508,90150,92160]
    df,sha,errors=ingest("Cowlevel.csv")
    if df is None or sha!=C_SHA:raise RuntimeError("Original clinical cow CSV unavailable: "+str(errors))
    if len(df)!=5529 or df.farm_ID.nunique()!=16:raise ValueError("Source cohort drift")
    df=df.copy()
    gait=pd.to_numeric(df["Gait score"],errors="coerce")
    df=df.loc[gait.notna()].copy()
    df["target"]=(gait[gait.notna()]>=3).astype(int)
    farms=np.asarray(sorted(int(x) for x in df.farm_ID.unique()))
    perm=np.random.default_rng(SEED).permutation(farms)
    groups={"train":sorted(map(int,perm[:10])),"cal":sorted(map(int,perm[10:13])),
            "test":sorted(map(int,perm[13:]))}
    assert len(set(sum(groups.values(),[])))==16
    tr=df[df.farm_ID.isin(groups["train"])]
    breeds=sorted(tr.Breed.astype(str).unique())
    means={col:float(pd.to_numeric(tr[col],errors="coerce").median()) for col in ("Lactation","DIM")}
    def features(rows):
        numeric=[pd.to_numeric(rows[col],errors="coerce").fillna(means[col]).to_numpy()[:,None]
                 for col in ("Lactation","DIM")]
        cats=[(rows.Breed.astype(str).to_numpy()==b).astype(float)[:,None] for b in breeds]
        return np.column_stack(numeric+cats).astype(np.float32)
    cow={}
    for k,ids in groups.items():
        sub=df[df.farm_ID.isin(ids)]
        cow[k]={"X":features(sub),"y":sub.target.to_numpy(np.float32),
                "group":sub.farm_ID.to_numpy(int)}
    for task,parts in (("pig",hot),("cow",cow)):
        raw=[parts[k]["X"] for k in ("train","cal","test")]
        xx=[np.log1p(np.maximum(0,a)) if task=="pig" else a for a in raw]
        scale=StandardScaler().fit(xx[0])
        for k,x in zip(("train","cal","test"),xx):
            parts[k]["scaled"]=np.clip(scale.transform(x),-8,8).astype(np.float32)
    return hot,cow,groups

class Tiny(nn.Module):
    def __init__(self,cow_dim,shared):
        super().__init__()
        self.shared=shared
        self.adapters=nn.ModuleDict({"pig":nn.Linear(12,24),"cow":nn.Linear(cow_dim,24)})
        def block():return nn.Sequential(nn.ReLU(),nn.Linear(24,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.blocks=nn.ModuleDict({"shared":block()} if shared else {"pig":block(),"cow":block()})
        self.heads=nn.ModuleDict({"pig":nn.Linear(16,1),"cow":nn.Linear(16,1)})
    def forward(self,t,x):
        return self.heads[t](self.blocks["shared" if self.shared else t](self.adapters[t](x))).flatten()
def predict(net,t,X):
    net.eval();chunks=[]
    with torch.no_grad():
        for a in range(0,len(X),4096):
            xx=torch.from_numpy(np.ascontiguousarray(X[a:a+4096]))
            chunks.append(torch.sigmoid(net(t,xx)).numpy())
    return np.concatenate(chunks)
def evaluate(parts,p):
    d=parts["test"];ids=d["group"];y=d["y"]
    return {"brier":brier(y,p),"samples":int(len(y)),"positive":int(y.sum()),
            "by_group":[{"id":int(g),"samples":int((ids==g).sum()),
                         "brier":brier(y[ids==g],p[ids==g])}
                         for g in sorted(set(ids))]}
def neural(shared,hot,cow):
    reset()
    m=Tiny(cow["train"]["scaled"].shape[1],shared)
    opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.001)
    rng=np.random.default_rng(SEED)
    tensors={t:(torch.from_numpy(d["train"]["scaled"]),torch.from_numpy(d["train"]["y"]))
             for t,d in (("pig",hot),("cow",cow))}
    denom={t:brier(d["cal"]["y"],np.full(len(d["cal"]["y"]),float(d["train"]["y"].mean())))
           for t,d in (("pig",hot),("cow",cow))}
    best=None;history=[];started=time.perf_counter()
    for ep in range(1,7):
        m.train()
        for _ in range(180):
            opt.zero_grad();ls=[]
            for task,bs in (("pig",256),("cow",128)):
                x,y=tensors[task]
                pos=torch.from_numpy(rng.integers(0,len(x),bs))
                ls.append(nn.functional.binary_cross_entropy_with_logits(m(task,x[pos]),y[pos]))
            (ls[0]+ls[1]).backward();opt.step()
        calib={t:brier(d["cal"]["y"],predict(m,t,d["cal"]["scaled"]))
               for t,d in (("pig",hot),("cow",cow))}
        criterion=np.mean([calib[t]/denom[t] for t in ("pig","cow")])
        history.append({"epoch":ep,"relative_cal_brier":float(criterion),"cal_brier":calib})
        if best is None or criterion<best["criterion"]:
            best={"criterion":criterion,"epoch":ep,
                  "weights":{k:v.detach().clone() for k,v in m.state_dict().items()}}
    m.load_state_dict(best["weights"])
    out={}
    for task,d in (("pig",hot),("cow",cow)):
        start=time.perf_counter()
        p=predict(m,task,d["test"]["scaled"])
        out[task]=evaluate(d,p)
        out[task]["ms_per_1000_rows"]=round((time.perf_counter()-start)*1000000/len(p),5)
    return {"shared":shared,"parameters":sum(p.numel() for p in m.parameters()),
            "fit_seconds":round(time.perf_counter()-started,2),
            "calibration_selected_epoch":best["epoch"],"calibration_history":history,
            "test":out}
def classic(hot,cow):
    results={}
    for task,d in (("pig",hot),("cow",cow)):
        y=d["train"]["y"].astype(int);x=d["train"]["X"]
        models={
          "train_prior":None,
          "regularized_logistic":make_pipeline(StandardScaler(),
             LogisticRegression(max_iter=400,C=1,random_state=SEED)),
          "tiny_tree":HistGradientBoostingClassifier(max_iter=90,max_depth=4,
             max_leaf_nodes=10,min_samples_leaf=350 if task=="pig" else 25,
             learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=SEED)}
        out={}
        for name,mod in models.items():
            if mod is None:p=np.full(len(d["test"]["y"]),float(y.mean()))
            else:
                mod.fit(x,y);p=mod.predict_proba(d["test"]["X"])[:,1]
            out[name]=evaluate(d,p)
        if task=="pig":
            idx=FEATURES.index("eating")
            out["author_camera_eating_rule"]=evaluate(d,(d["test"]["X"][:,idx]>0).astype(float))
        results[task]=out
    return results
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();reset()
    hot,cow,groups=data()
    baselines=classic(hot,cow)
    separate=neural(False,hot,cow)
    shared=neural(True,hot,cow)
    gates={};both=True
    for task,minimum in (("pig",3),("cow",2)):
        candidates={k:v for k,v in baselines[task].items() if k in ("regularized_logistic","tiny_tree")}
        candidates["separate_neural"]=separate["test"][task]
        top=min(candidates,key=lambda name:candidates[name]["brier"])
        a=candidates[top];b=shared["test"][task]
        gain=(a["brier"]-b["brier"])/a["brier"]
        ar={r["id"]:r["brier"] for r in a["by_group"]}
        br={r["id"]:r["brier"] for r in b["by_group"]}
        if set(ar)!=set(br):raise AssertionError("Group identity mismatch")
        wins=sum(br[g]<ar[g] for g in ar)
        good=bool(gain>=.02 and wins>=minimum)
        gates[task]={"best_individual_model":top,
                     "best_brier":a["brier"],"shared_brier":b["brier"],
                     "relative_improvement":gain,
                     "heldout_groups_better":wins,"groups_tested":len(ar),"pass":good}
        both=both and good
    out={
      "scientific_status":"PASS_DEVELOPMENTAL_SHARED_MODEL" if both else "BLOCKED_SHARED_MODEL_NOT_JUSTIFIED",
      "predeclared_gate":"Shared model >=2% lower heldout Brier than best trained specialist on BOTH source tasks AND wins on >=3/4 pigs and >=2/3 farm groups.",
      "source_hashes":{"hotpig":H_SHA,"cow":C_SHA},"cow_farm_splits":groups,
      "pig_splits":{"train":list(range(1,17)),"cal":list(range(17,21)),"test":list(range(21,25))},
      "claims":"Instrument feeding from author camera counts; contemporaneous expert locomotion risk from cow demographics, NOT direct pain/welfare or prospective action.",
      "limits":[
        "One HotPig site/four test pigs; one Brazilian study/three new developmental heldout farms.",
        "Clinical cow split derived deterministically AFTER previous all-16-farm exploratory work; NEVER blind independent test.",
        "Only neural internal trunk shares parameters, source adapters and task-specific outputs remain separate.",
        "No raw video/audio/clinical real-time welfare capability or pretrained open decision model same-input arm.",
        "No per-minute HotPig independence; all inferential uncertainty must cluster by subject/farm.",
        "No frozen multimodal foundation baseline can be equated to a model using processed camera counts and demographic features."
      ],
      "classical":baselines,"separate_neural":separate,"shared_neural":shared,"gates":gates}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("OPENFARM_STAGE_B_TWO_SOURCE_GATE",json.dumps({"status":out["scientific_status"],
      "baselines":{t:{n:round(v["brier"],6) for n,v in vals.items()} for t,vals in baselines.items()},
      "separate":{t:round(x["brier"],6) for t,x in separate["test"].items()},
      "shared":{t:round(x["brier"],6) for t,x in shared["test"].items()},
      "gates":gates},sort_keys=True),flush=True)
if __name__=="__main__":main()
