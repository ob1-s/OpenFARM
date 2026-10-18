#!/usr/bin/env python3
"""Ask et al 2024 source-verified EquiFACS→induced orthopedic pain benchmark.

Reads 348 original video-level facial-action summaries and evaluates structured
small probabilistic heads. 8 horses, source IDs stay entirely out of features.
Reference is gait/induction-derived pain condition, not human pain-face labels or
an observed clinical treatment effect. No raw video is supplied.
"""
import collections,hashlib,io,json,math,statistics,urllib.request,zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,roc_auc_score,balanced_accuracy_score,average_precision_score

SRC="https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-023-50383-y/MediaObjects/41598_2023_50383_MOESM1_ESM.xlsx"
SHA="7df3174b0190dc6b8f51a58648957500f221bf6fc16aa97252f67211f61367ad"
NS="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
def addr(code):
    n=0
    for c in code:
        if not c.isalpha():break
        n=26*n+ord(c.upper())-64
    return n-1
def get_sheet():
    req=urllib.request.Request(SRC,headers={"User-Agent":"OpenFARM-Ask2024-pinned-EquiFACS-research/0.2"})
    with urllib.request.urlopen(req,timeout=90) as r:buf=r.read(20000000)
    if hashlib.sha256(buf).hexdigest()!=SHA:raise ValueError("Source supplementary workbook fingerprint changed")
    with zipfile.ZipFile(io.BytesIO(buf)) as z:
        if "xl/sharedStrings.xml" in z.namelist():
            shared=["".join(t.text or "" for t in si.iter(NS+"t"))
                for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS+"si")]
        else:shared=[]
        w=ET.fromstring(z.read("xl/workbook.xml"))
        sheets=w.find(NS+"sheets").findall(NS+"sheet")
        sheet_ids={s.attrib["name"]:i for i,s in enumerate(sheets,1)}
        if "Frequency and duration" not in sheet_ids:raise ValueError("Missing author 348-sequence data sheet")
        path=f"xl/worksheets/sheet{sheet_ids['Frequency and duration']}.xml"
        root=ET.fromstring(z.read(path))
        xmlrows=root.find(NS+"sheetData").findall(NS+"row")
    rows=[]
    for row in xmlrows:
        data={}
        for cell in row.findall(NS+"c"):
            p=addr(cell.attrib["r"])
            value=cell.find(NS+"v")
            if value is None:
                if cell.attrib.get("t")=="inlineStr":
                    s="".join(x.text or "" for x in cell.iter(NS+"t"))
                else:continue
            else:
                s=value.text or ""
                if cell.attrib.get("t")=="s":s=shared[int(s)]
            data[p]=s
        rows.append(data)
    return rows
def extract(raw):
    header={i:str(name).strip() for i,name in raw[0].items()}
    cols=sorted(header)
    require={"subj","pain","video"}
    if not require.issubset(set(header.values())):raise ValueError("Original required columns missing")
    keys={v:k for k,v in header.items()}
    feature_cols={idx:name for idx,name in header.items() if name.lower().startswith(("dur_","count_"))}
    if len(feature_cols)<70:
        print("ASK2024_ORIGINAL_COLUMN_NAMES",json.dumps(list(header.values())),flush=True)
        raise ValueError("Insufficient original 78 AU features; inspect source names before accepting activity input")
    videos=[];ids=set();counts=collections.Counter()
    for raw_row in raw[1:]:
        animal=str(raw_row.get(keys["subj"],"")).strip()
        if not animal:continue
        pain=str(raw_row.get(keys["pain"],"")).strip().lower()
        if pain not in ("yes","no"):raise ValueError("Unknown original source binary pain coding "+repr(pain))
        video=str(raw_row.get(keys["video"],"")).strip()
        if video in ids:raise ValueError("Duplicate original source video")
        ids.add(video);counts[(animal,pain)]+=1
        values=[]
        for col in sorted(feature_cols):
            original=raw_row.get(col)
            if original is None or str(original).strip()=="":
                values.append(0.)
            else:
                try:values.append(float(original))
                except ValueError:raise ValueError("Non-numeric author AU feature")
        if any(not math.isfinite(v) or v<0 for v in values):raise ValueError("Invalid annotation features")
        videos.append((animal,video,int(pain=="yes"),values))
    if len(videos)!=348 or len(set(x[0] for x in videos))!=8:
        raise ValueError("Original 348 source videos and 8 horses required")
    return videos,[feature_cols[c] for c in sorted(feature_cols)],dict((str(k),v) for k,v in counts.items())
def metrics(y,p):
    pred=(p>=.5).astype(int)
    return {"n":len(y),"pain":int(y.sum()),"nopain":int(len(y)-y.sum()),
        "brier":float(brier_score_loss(y,p)),
        "accuracy":float(np.mean(y==pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "auroc":float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        "auprc":float(average_precision_score(y,p)) if len(set(y))==2 else None,
        "pain_recall":float(np.mean(pred[y==1]==1)) if any(y==1) else None,
        "nopain_specificity":float(np.mean(pred[y==0]==0)) if any(y==0) else None,
        "n_positive_predictions":int(pred.sum())}
def main():
    raw=get_sheet()
    rows,names,counts=extract(raw)
    animals=[x[0] for x in rows];y=np.array([x[2] for x in rows])
    X=np.array([x[3] for x in rows],dtype=np.float64)
    # Precommitted animal IDs; deterministic, no label-based split search.
    splits={"train":["h1","h2","h3","h4","h5"],"calibration":["h6"],"test":["h7","h8"]}
    ixs={key:np.array([i for i,a in enumerate(animals) if a in herd],dtype=int)
        for key,herd in splits.items()}
    for key,idx in ixs.items():
        if not len(idx) or set(y[idx])!={0,1}:raise ValueError("Animal split lacks both source classes "+key)
    train,cal,test=[ixs[x] for x in ("train","calibration","test")]
    byhorse=collections.Counter(animals[i] for i in train)
    wt=np.array([1/byhorse[animals[i]] for i in train])
    wt=wt*len(wt)/wt.sum()
    # Nuisance probes: raw AU counts vs proportions (video-length/activity).
    raw_matrix=X
    totals=np.maximum(X.sum(axis=1,keepdims=True),1.)
    representations={
      "all_event_count_or_duration_features":raw_matrix,
      "within_video_relative_action_activity":X/totals,
      "total_action_activity_only":totals,
    }
    result={"source_doi":"10.1038/s41598-023-50383-y","source_sha256":SHA,
      "reference":"Published binary induced orthopedic pain condition; informed by objective gait asymmetry, not a direct independent treatment outcome",
      "modality":"source EquiFACS per-video human-coded activity durations/frequencies; NOT raw video",
      "n_individual_AU_events_in_original_workbook":37072,
      "n_source_videos":len(rows),"n_independent_horses":8,
      "feature_names":names,"n_numeric_AU_features":len(names),
      "splits":{kind:{"horses":herd,"videos":len(ixs[kind]),"pain":int(y[ixs[kind]].sum())} for kind,herd in splits.items()},
      "source_label_counts":{kind:int(sum(y==kind)) for kind in (0,1)},
      "models":{},
      "warning":"Induced pain proxy/gait condition with human-coded AU features; video windows within 8 horses correlated. Animal-disjoint evaluation still tiny. Not a model of raw image/video comprehension."}
    for key,mat in representations.items():
        trials=[]
        for C in (.001,.03,.3,3,30):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000,random_state=20261018))
            model.fit(mat[train],y[train],logisticregression__sample_weight=wt)
            probs=model.predict_proba(mat[cal])[:,1]
            brier=float(brier_score_loss(y[cal],probs))
            trials.append((brier,C,model))
        brier,C,model=min(trials,key=lambda p:(p[0],p[1]))
        probs=model.predict_proba(mat[test])[:,1]
        result["models"][key]={"calibration_brier":brier,"selected_regularization_C":C,
          "test":metrics(y[test],probs),
          "test_per_animal":{horse:metrics(y[np.array([i for i in test if animals[i]==horse])],
                        model.predict_proba(mat[np.array([i for i in test if animals[i]==horse])])[:,1])
                     for horse in splits["test"]}}
        print("ASK2024_MODEL_DONE",key,"test_brier",result["models"][key]["test"]["brier"],flush=True)
    prior=float(np.average(y[train],weights=wt))
    result["models"]["constant_5horse_pain_prior"]={"test":metrics(y[test],np.full(len(test),prior))}
    f=Path("reports/decision-index/v02_ask2024_348_equifacs_specialists.json")
    f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(result,indent=2)+"\n")
    print("ASK2024_TRAINED_SOURCE_RESULT",json.dumps({"n_videos":len(rows),"n_AU_features":len(names),
        "splits":result["splits"],"summary":{k:v["test"] for k,v in result["models"].items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
