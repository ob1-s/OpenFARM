#!/usr/bin/env python3
"""True LiquidAI d1-omni-600M Q8 GGUF inference via llama.cpp CPU SystemOne.

Original expert references and cases; source data stay private except Farrow's
upstream public records. Never emits clinical text or respondent IDs.
"""
import argparse,collections,json,math,statistics,time,urllib.request,urllib.error
from pathlib import Path
from index_l3_cpu_julia_evaluate import load_cases,rubric,WONG_LABELS,WONG_KEYS,FARROW_KEYS

def go(task,limit):
    cases=load_cases(task)
    keys=FARROW_KEYS if task=="farrow" else WONG_KEYS
    vals=[];lat=[];errors=collections.Counter();n=min(limit,len(cases));critical=hits=0
    for i,case in enumerate(cases[:n],1):
        body={"state":case["state"],"questions":{"triage":rubric(task)}}
        req=urllib.request.Request("http://127.0.0.1:8080/v1/systemone",
             method="POST",data=json.dumps(body,ensure_ascii=False).encode(),
             headers={"Content-Type":"application/json"})
        try:
            t=time.perf_counter()
            with urllib.request.urlopen(req,timeout=120) as r:message=json.loads(r.read(300000))
            elapsed=(time.perf_counter()-t)*1000
            a=message["answers"]["triage"]
            p=a["probabilities"]
            if a.get("type")!="choice" or not isinstance(p,dict) or set(p)!=set(keys):
                raise ValueError("Model answer incompatible with frozen rubric")
            q=[float(p[k]) for k in keys]
            if any(not math.isfinite(x) or x<0 for x in q) or abs(sum(q)-1)>.025:
                raise ValueError("Model answer invalid probability vector")
            q=[x/sum(q) for x in q]
            pred=max(range(len(keys)),key=lambda j:q[j])
            if task=="farrow":
                gold=keys.index(case["reference"]);ref=[gold];distribution=[int(i==gold) for i in range(len(keys))]
            else:
                ref=sorted(WONG_LABELS.index(x) for x in case["vet_votes"])
                gold=ref[1];distribution=[ref.count(i)/3 for i in range(5)]
            if gold==0:critical+=1;hits+=int(pred==0)
            vals.append({
                "exact":int(pred==gold),
                "brier":sum((q[j]-distribution[j])**2 for j in range(len(keys))),
                "undertriage":sum(pred>x for x in ref)/len(ref),
                "overtriage":sum(pred<x for x in ref)/len(ref),
                "ordinal_mae":sum(abs(pred-x) for x in ref)/len(ref)})
            lat.append(elapsed)
        except Exception as e:
            errors[type(e).__name__]+=1
            print("LIQUID_DECISION_ERROR",task,i,type(e).__name__,flush=True)
        if i==1 or i%10==0 or i==n:print("LIQUID_PROGRESS",task,i,"/",n,flush=True)
    outcome={"model":"LiquidAI/d1-omni-600M-GGUF:Q8_0","runtime":"llama.cpp CPU /v1/systemone",
       "task":task,"attempted":n,"scored":len(vals),"coverage":len(vals)/n,
       "errors":dict(errors),"primitive":"choice","source_examples":len(cases),
       "metrics":{k:round(statistics.mean(x[k] for x in vals),6) for k in vals[0]} if vals else {},
       "median_end_to_end_cpu_ms":round(statistics.median(lat),2) if lat else None,
       "majority_immediate_n":critical,"majority_immediate_recall":hits/critical if critical else None,
       "note":"Zero-shot original public or privately supplied human-labelled clinical examples; no clinical safety evidence"}
    f=Path("reports/index_l3_vet")/("cpu_liquid_d1_omni_"+task+"_choice.json")
    f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(outcome,indent=2)+"\n")
    print("LIQUID_REAL_EVAL_RESULT",json.dumps(outcome,sort_keys=True),flush=True)
    if len(vals)<n:raise RuntimeError("Liquid true-weights benchmark incomplete")
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--task",choices=["farrow","wong"],required=True);a.add_argument("--limit",type=int,default=999)
    args=a.parse_args();go(args.task,args.limit)
