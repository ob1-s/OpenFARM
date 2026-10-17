#!/usr/bin/env python3
"""Research-only ordinal Jev Score ablation on 102 private Wong triage test cases.

Only aggregated metrics appear in artifacts/logs. Source supplied via repo secret.
"""
import collections,json,math,os,statistics,time,urllib.error,urllib.request
from pathlib import Path
from index_l3_wong_jev_private_runner import load,metrics,boot,LABELS

RUBRIC=[
    "Lowest urgency: veterinarian can assess within 240 minutes.",
    "Stable noncritical presentation: veterinarian should assess within 120 minutes.",
    "Urgent presentation: veterinarian should assess within 30 to 60 minutes.",
    "Very urgent presentation: veterinarian should assess within 15 minutes.",
    "Immediate highest-priority emergency: veterinarian must assess now."
]
QUESTION={"type":"score",
    "instructions":"Assign urgency for a dog already at an emergency veterinary hospital using only the presenting signs and intake clinical information. The five score levels are ordered from least urgent to most urgent. Balance delayed dangerous conditions with unnecessary emergency escalation.",
    "criteria":RUBRIC}

def ask(case,secret,model):
    req=urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",method="POST",
        data=json.dumps({"model":model,"state":case["state"],"questions":{"vet_triage":QUESTION}},ensure_ascii=False).encode(),
        headers={"Authorization":"Bearer "+secret,"Content-Type":"application/json"})
    for j in range(4):
        started=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=50) as response:
                message=json.loads(response.read(128000))
            answer=message.get("answers",{}).get("vet_triage",{})
            values=answer.get("probabilities")
            if isinstance(values,dict):
                v=[values.get(str(i)) for i in range(5)]
            elif isinstance(values,list):
                v=values
            else:
                v=[]
            if answer.get("type")=="score" and len(v)==5 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in v):
                total=sum(v)
                if .975<=total<=1.025:
                    return [x/total for x in reversed(v)],(time.perf_counter()-started)*1000,message.get("usage",{}),str(message.get("model"))
        except urllib.error.HTTPError as e:
            if e.code not in (429,529,502,503):raise RuntimeError("Jev API returned HTTP "+str(e.code)) from None
        except urllib.error.URLError:
            pass
        time.sleep(.2*(j+1))
    return None,None,{},None

def main():
    d=load()
    if d is None:return
    key=os.getenv("TYPESAFE_API_KEY")
    if not key:raise RuntimeError("API credential not available")
    cases=sorted(d["cases"],key=lambda c:c["id"])
    output={};summaries={}
    for model in ("jev-1.13.0","jev-1.14.0"):
        pred={};lat=[];tokens=collections.Counter();invalid=0;served=set()
        for i,c in enumerate(cases,1):
            p,ms,usage,name=ask(c,key,model)
            if p is None:invalid+=1;continue
            pred[c["id"]]=p;lat.append(ms);served.add(name)
            tokens.update(input_tokens=int(usage.get("input_tokens",0)),output_tokens=int(usage.get("output_tokens",0)))
            if i%25==0 or i==102:print("WONG_SCORE_PROGRESS",model,i,"/102",flush=True)
        result,_=metrics(cases,pred)
        majority_critical=[c for c in cases if sorted(LABELS.index(v) for v in c["vet_votes"])[1]==0]
        result.update(model_requested=model,models_served=sorted(served),invalid_responses=invalid,
            median_latency_ms=statistics.median(lat) if lat else None,usage=dict(tokens),
            unanimous_references=sum(len(set(c["vet_votes"]))==1 for c in cases),
            critical_cases=len(majority_critical),
            critical_cases_immediate_predictions=sum(c["id"] in pred and pred[c["id"]].index(max(pred[c["id"]]))==0 for c in majority_critical))
        summaries[model]=result;output[model]=pred
    common=[c for c in cases if all(c["id"] in output[m] for m in output)]
    _,a=metrics(common,output["jev-1.13.0"]);_,b=metrics(common,output["jev-1.14.0"])
    result={"task":"Wong original hospital triage, 102 case-exclusive heldout",
       "primitive":"score","scores":summaries,"paired_n":len(common),
       "paired_brier_1_14_minus_1_13":boot(b,a) if common else None,
       "warning":"Retrospective expert opinion with few majority-immediate cases; Score and Choice prompts differ in both primitive and rubric. Not a clinical safety test."}
    file=Path("reports/index_l3_vet/wong_jev_score_secret_aggregate.json")
    file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(result,indent=2)+"\n")
    print("WONG_JEV_SCORE_SUMMARY",json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":main()
