#!/usr/bin/env python3
"""Original 605 equine expert-pain-face frames -> small probabilistic image heads.

Original source MD5 verified. ResNet18 frozen ImageNet features + regularized
binary probabilistic head vs cheap 64-bin RGB color-histogram shortcut probe.
11-horse identity map inferred from companion 2025 author filenames, NOT yet
pixel-linked across source releases. No raw image or feature data exported.
"""
import argparse,collections,hashlib,io,json,math,re,statistics,tempfile,urllib.request,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18,ResNet18_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score,average_precision_score,balanced_accuracy_score

REPO=Path("benchmarks/decision-index/v0.2")
MAP=REPO/"haf2024-video-horse-id-map.json"
URL="https://zenodo.org/records/14541977/files/Pain%20and%20NoPain.zip?download=1"
MD5="be09a4ac6c1d3db3c4908cad1aeca075"
PAT=re.compile(r"cropped_(pain|nopain)_video(\d+)_1_(\d+)\.png",re.I)
def original():
    req=urllib.request.Request(URL,headers={"User-Agent":"OpenFARM-train-original-equine-ordinal/0.2"})
    return urllib.request.urlopen(req,timeout=100)
def source_zip(path):
    md5=hashlib.md5();n=0
    with original() as s,open(path,"wb") as d:
        while b:=s.read(4*1024*1024):
            md5.update(b);d.write(b);n+=len(b)
    if md5.hexdigest()!=MD5:raise ValueError("Original equine media source mismatch")
    print("HORSE_SPECIALIST_SOURCE_BYTES",n,flush=True)
def read_labels(z,ident):
    original=[x for x in z.namelist() if x.lower().endswith(".png")]
    if len(original)!=605:raise ValueError("605 original source frames expected")
    out=[]
    for path in sorted(original):
        m=PAT.fullmatch(Path(path).name)
        if not m:raise ValueError("Unrecognized original media annotation")
        kind,v,frame=m.groups();horse=ident.get(str(v))
        if horse is None:raise ValueError("Horse ID not in original author map")
        out.append({"source_member":path,"horse":horse,"video":int(v),
             "frame":int(frame),"gold":int(kind.lower()=="pain")})
    return out
def split(rows,source_manifest):
    group={name:set(v) for name,v in source_manifest["proposed_split_inferred_horse_ids"].items()}
    if any(group[a]&group[b] for a in group for b in group if a<b):raise ValueError("Horse split collision")
    out={}
    for i,r in enumerate(rows):
        matches=[name for name,s in group.items() if r["horse"] in s]
        if len(matches)!=1:raise ValueError("Unassigned horse")
        out.setdefault(matches[0],[]).append(i)
    if set(out)!={"train","calibration","test"}:raise ValueError("3 splits required")
    for name in out:
        if set(rows[i]["gold"] for i in out[name])!={0,1}:
            raise ValueError("Split has no both pain-face classes: "+name)
    return out
def model_features(z,rows):
    torch.set_num_threads(4)
    torch.manual_seed(20261018)
    model=resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc=torch.nn.Identity()
    model.eval()
    transform=ResNet18_Weights.DEFAULT.transforms()
    deep=[];weak=[]
    for i,row in enumerate(rows,1):
        with Image.open(io.BytesIO(z.read(row["source_member"]))) as im:
            rgb=im.convert("RGB")
            tensor=transform(rgb)
            small=np.asarray(rgb.resize((64,64)),dtype=np.float32)
        with torch.inference_mode():
            emb=model(tensor.unsqueeze(0)).cpu().numpy()[0]
        deep.append(emb)
        # 64 RGB bins, histogram and no facial shape/structure information.
        hist=[]
        for ch in range(3):
            h,_=np.histogram(small[:,:,ch],bins=16,range=(0,256),density=False)
            hist.extend((h/h.sum()).tolist())
        weak.append(hist)
        if i%100==0:print("HORSE_SPECIALIST_FEATURES",i,"/605",flush=True)
    return {"resnet18_frozen":np.asarray(deep,dtype=np.float64),
            "rgb_histogram_control":np.asarray(weak,dtype=np.float64)}
def evaluate(y,probs):
    hard=(probs>=.5).astype(int)
    n=len(y)
    p=float(np.mean(y))
    ans={"n":n,"reference_pain_fraction":p,
        "brier":float(brier_score_loss(y,probs)),
        "accuracy":float(np.mean(hard==y)),
        "balanced_accuracy":float(balanced_accuracy_score(y,hard)),
        "pain_recall":float(np.mean(hard[y==1]==1)),
        "no_pain_specificity":float(np.mean(hard[y==0]==0)),
        "positive_predictions":int(hard.sum()),
        "mean_predicted_pain":float(np.mean(probs))}
    if len(set(y))==2:
        ans.update(auroc=float(roc_auc_score(y,probs)),auprc=float(average_precision_score(y,probs)))
    return ans
def main():
    args=argparse.ArgumentParser();args.add_argument("--output",type=Path,default=Path("reports/decision-index/v02_haf605_cpu_specialist.json"))
    opt=args.parse_args()
    m=json.loads(MAP.read_text())
    if "605-FRAME_CROSS_RELEASE_IDENTITIES_INFERRED" not in m["status"]:raise ValueError("Wrong source identity evidence status")
    with tempfile.TemporaryDirectory(prefix="openfarm_haf_train_") as tmp:
        path=Path(tmp)/"author-original-605.zip";source_zip(path)
        with zipfile.ZipFile(path) as z:
            rows=read_labels(z,m["video_to_horse"])
            splits=split(rows,m)
            features=model_features(z,rows)
    y=np.asarray([r["gold"] for r in rows],dtype=int)
    train,cal,test=[splits[x] for x in ("train","calibration","test")]
    # Avoid treating the most photographed horse as dozens of independent animals.
    num_by_horse=collections.Counter(rows[i]["horse"] for i in train)
    weights=np.asarray([1/num_by_horse[rows[i]["horse"]] for i in train],dtype=np.float64)
    weights=weights*len(weights)/weights.sum()
    out={"status":"EXPLORATORY_RESEARCH_NOT_CLINICAL","original_source_md5":MD5,
      "animal_mapping_status":m["status"],"source_evidence":m["source_evidence_run"],
      "modeling":"frozen torchvision ResNet18 ImageNet embedding + logistic vs RGB histogram shortcut",
      "selected_by":"mean per-horse calibration Brier, 2 calibration horses; test not used for selection",
      "split_horses":m["proposed_split_inferred_horse_ids"],
      "split_counts":{name:{"frames":len(ix),"videos":len(set(rows[i]["video"] for i in ix)),
                            "pain":int(y[ix].sum()),"nopain":int(len(ix)-y[ix].sum())} for name,ix in splits.items()},
      "models":{},"note":"Small 11-horse single-source correlated frames; expert-reported pain face != physiological nociception. Same 22 source video ids inferred across author releases. Training/evaluation research only."}
    Cs=[.001,.03,.3,3,30]
    for name,X in features.items():
        candidates=[]
        for C in Cs:
            clf=LogisticRegression(C=C,max_iter=2500,random_state=20261018)
            clf.fit(X[train],y[train],sample_weight=weights)
            pp=clf.predict_proba(X[cal])[:,1]
            h=collections.defaultdict(list)
            for i,val in zip(cal,pp):
                h[rows[i]["horse"]].append((y[i]-val)**2)
            b=float(statistics.mean(statistics.mean(xs) for xs in h.values()))
            candidates.append((b,C,clf))
        best=sorted(candidates,key=lambda t:(t[0],t[1]))[0]
        _,C,clf=best
        p=clf.predict_proba(X[test])[:,1]
        vals=evaluate(y[test],p)
        by_horse={}
        for horse in sorted({rows[i]["horse"] for i in test}):
            ids=np.asarray([j for j,i in enumerate(test) if rows[i]["horse"]==horse])
            by_horse[horse]=evaluate(y[test][ids],p[ids])
        out["models"][name]={
           "selected_C":C,"mean_horse_calibration_brier":best[0],
           "test":vals,"test_per_horse":by_horse,
           "source_feature_dim":int(X.shape[1]),
           "calibration_search":[{"C":cc,"cal_brier":bb} for bb,cc,_ in candidates]}
    prior=float(np.average(y[train],weights=weights))
    out["models"]["constant_train_pain_prior"]={"test":evaluate(y[test],np.full(len(test),prior))}
    opt.output.parent.mkdir(parents=True,exist_ok=True)
    opt.output.write_text(json.dumps(out,indent=2)+"\n")
    print("HORSE_SPECIALIST_REAL_SOURCE_RESULT",json.dumps({"status":out["status"],"split_counts":out["split_counts"],
          "models":{k:v["test"] for k,v in out["models"].items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
