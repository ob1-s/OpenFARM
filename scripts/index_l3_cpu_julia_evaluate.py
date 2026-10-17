#!/usr/bin/env python3
"""Real Julia-1 weights on pinned Wong/Farrow original L3 human-label tasks.

CPU-only, zero-shot. Original clinical texts and single-case predictions NEVER
logged or committed. Output is aggregate case-level research metrics.
"""
import argparse,base64,collections,gzip,json,math,os,statistics,time
from pathlib import Path

WONG_EXPECTED_SHA="4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027"
WONG_LABELS=["Immediate","15 mins","30-60 mins","120 mins","240 mins"]
WONG_KEYS=["immediate","within_15min","within_30to60min","within_120min","within_240min"]
WONG_RULES={
'immediate':'Immediate veterinarian assessment, no acceptable delay; severe unstable emergency.',
'within_15min':'Veterinarian assessment within 15 minutes; very urgent.',
'within_30to60min':'Veterinarian assessment within 30 to 60 minutes; urgent.',
'within_120min':'Veterinarian assessment within 120 minutes; stable non-critical presentation.',
'within_240min':'Veterinarian assessment within 240 minutes; lowest urgency.'}
FARROW_KEYS=["within_1h","within_12h","within_24h","within_7d","within_14d","self_care"]
FARROW_RULES={
'within_1h':'Seek veterinary care immediately, within one hour.',
'within_12h':'Seek veterinary care within 12 hours.',
'within_24h':'Seek veterinary care within 24 hours.',
'within_7d':'Seek veterinary care within seven days.',
'within_14d':'Seek veterinary care within fourteen days.',
'self_care':'Manage at home for now, without immediately seeking veterinary care.'
}
def load_cases(task):
    if task=="farrow":
        data=json.loads(Path("reports/index_l3_farrow/cases_local_reference.json").read_text())
        if len(data)!=30 or len({v["id"] for v in data})!=30:
            raise ValueError("Farrow original 30 source-only case count mismatch")
        return sorted(data,key=lambda d:d["id"])
    secret=os.environ.get("WONG_TEST_CASES_B64")
    if not secret:raise RuntimeError("Required original Wong test cases not mounted via Actions secret")
    data=json.loads(gzip.decompress(base64.b64decode(secret,validate=True)))
    if data.get("source_sha256")!=WONG_EXPECTED_SHA or data.get("split")!="test":
        raise ValueError("Unexpected frozen Wong source or split")
    cases=data.get("cases",[])
    if len(cases)!=102 or len({v["id"] for v in cases})!=102:
        raise ValueError("Wong original 102 unique test cases required")
    if any(len(x["vet_votes"])!=3 or not all(v in WONG_LABELS for v in x["vet_votes"]) for x in cases):
        raise ValueError("Missing independent veterinarian votes")
    return sorted(cases,key=lambda d:d["id"])

def rubric(task):
    if task=="farrow":
        return {"type":"choice","instructions":
            "For this owner-reported dog case, how soon should the owner seek veterinary advice and care? Use only the vignette, and choose the safest proportionate option.",
            "criteria":FARROW_RULES}
    return {"type":"choice","instructions":
        "For this dog already presenting to an emergency veterinary hospital, use only the intake complaint, history and clinical signs to select a proportionate maximum wait for the veterinarian to assess it. Balance dangerous delay against unnecessary emergency escalation.",
        "criteria":WONG_RULES}

def run(args):
    import torch
    from julia import load_model
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    cases=load_cases(args.task)
    classes=FARROW_KEYS if args.task=="farrow" else WONG_KEYS
    questions={"triage":rubric(args.task)}
    print("JULIA_REAL_WEIGHTS_START",json.dumps({"task":args.task,"n":min(len(cases),args.limit),
          "model":"SupersonicLabs/Julia-1","primitive":"choice","device":"cpu",
          "no_test_training":True}),flush=True)
    started=time.perf_counter()
    model=load_model(args.checkpoint,device="cpu",strict_encoding=True,
           max_length=8192,head_length=512)
    load_seconds=time.perf_counter()-started
    values=[];lat=[];fails=collections.Counter();count=0;critical_hits=critical=0
    for i,case in enumerate(cases[:args.limit],1):
        try:
            t=time.perf_counter()
            response=model.predict(state=case["state"],questions=questions)
            elapsed=time.perf_counter()-t
            answers=response["answers"]["triage"]
            p=answers["probabilities"]
            if set(p)!=set(classes):
                raise ValueError("Returned option set did not match frozen rubric")
            pred=[float(p[k]) for k in classes]
            if not all(math.isfinite(v) and v>=0 for v in pred) or abs(sum(pred)-1)>.02:
                raise ValueError("Probabilities invalid")
            pred=[v/sum(pred) for v in pred]
            modal=max(range(len(pred)),key=lambda j:pred[j])
            if args.task=="farrow":
                gold=classes.index(case["reference"])
                q=[int(j==gold) for j in range(len(classes))]
                reference_modes=[gold]
            else:
                votes=[WONG_LABELS.index(v) for v in case["vet_votes"]]
                q=[votes.count(j)/3 for j in range(5)]
                reference_modes=sorted(votes)
                gold=reference_modes[1]
            if gold==0:
                critical+=1;critical_hits+=int(modal==0)
            values.append({"brier":sum((pred[j]-q[j])**2 for j in range(len(classes))),
                "accuracy":int(modal==gold),
                "undertriage":sum(modal>v for v in reference_modes)/len(reference_modes),
                "overtriage":sum(modal<v for v in reference_modes)/len(reference_modes),
                "ordinal_mae":sum(abs(modal-v) for v in reference_modes)/len(reference_modes)})
            lat.append(elapsed*1000)
        except Exception as e:
            fails[type(e).__name__]+=1
            print("JULIA_CASE_INVALID",args.task,i,type(e).__name__,flush=True)
        count+=1
        if i==1 or i%10==0 or i==min(len(cases),args.limit):
            print("JULIA_PROGRESS",args.task,i,"/",min(len(cases),args.limit),flush=True)
    report={"model":"SupersonicLabs/Julia-1","checkpoint_source":args.checkpoint,
        "task":args.task,"source_n":len(cases),"attempted":count,"scored":len(values),
        "coverage":len(values)/count if count else 0,"errors":dict(fails),
        "engine":"Julia-1 native named question, genuine torch CPU weights, no training",
        "primitive":"choice","load_seconds":round(load_seconds,3),
        "median_cpu_decision_ms":round(statistics.median(lat),3) if lat else None,
        "metrics":{k:round(statistics.mean(v[k] for v in values),6) for k in values[0]} if values else {},
        "critical_n":critical,"critical_recall":critical_hits/critical if critical else None,
        "limits":"Source Wong lacks animal IDs; Farrow has only 30 unique vignettes; not clinical deployment evidence."}
    path=Path("reports/index_l3_vet")/("cpu_julia_"+args.task+"_choice.json")
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n")
    print("JULIA_REAL_EVAL_RESULT",json.dumps(report,sort_keys=True),flush=True)
    if not values or len(values)!=count:raise RuntimeError("CPU model evaluation incomplete")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",required=True,choices=["farrow","wong"])
    ap.add_argument("--checkpoint",default="Julia-1")
    ap.add_argument("--limit",type=int,default=999)
    run(ap.parse_args())
