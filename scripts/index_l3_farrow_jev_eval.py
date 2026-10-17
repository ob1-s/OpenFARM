"""Farrow 30-case Jev Choice research evaluation, aggregate metrics only."""
import json,math,os,statistics,time,urllib.request
from pathlib import Path
NAMES=["within_1h","within_12h","within_24h","within_7d","within_14d","self_care"]
RULES={
"within_1h":"Seek veterinary care immediately, within one hour.",
"within_12h":"Seek veterinary care within 12 hours.",
"within_24h":"Seek veterinary care within 24 hours.",
"within_7d":"Seek veterinary care within seven days.",
"within_14d":"Seek veterinary care within fourteen days.",
"self_care":"Manage at home for now, without immediately seeking veterinary care."
}
def main():
    key=os.environ.get("TYPESAFE_API_KEY")
    if not key:raise RuntimeError("Actions API credential unavailable")
    p=Path("reports/index_l3_farrow/cases_local_reference.json")
    cases=sorted(json.loads(p.read_text()),key=lambda x:x["id"])
    if len(cases)!=30:raise ValueError("Farrow original case count mismatch")
    score=[];serving=set();tokens=0
    n=int(os.environ.get("FARROW_JEV_N","5"))
    if not 1<=n<=30:raise ValueError("case budget outside 1..30")
    for case in cases[:n]:
        req={"model":"jev-1.13.0","state":case["state"],
             "questions":{"urgency":{"type":"choice","instructions":"For this owner-reported dog case, how soon should the owner seek veterinary advice and care? Use only the vignette, and choose the safest proportionate option.","criteria":RULES}}}
        request=urllib.request.Request("https://api.typesafe.ai/v1/systemone",method="POST",data=json.dumps(req).encode("utf-8"),
          headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
        started=time.perf_counter()
        with urllib.request.urlopen(request,timeout=60) as r: out=json.loads(r.read(300000))
        ans=out["answers"]["urgency"];probs=ans["probabilities"]
        if ans.get("type")!="choice" or set(probs)!=set(NAMES) or abs(sum(probs.values())-1)>.005:raise ValueError("API Choice response invalid")
        gold=NAMES.index(case["reference"]);pred=max(range(6),key=lambda i:probs[NAMES[i]])
        score.append({"exact":int(gold==pred),"under":int(pred>gold),"over":int(pred<gold),
                      "ordinal_error":abs(pred-gold),"brier":sum((probs[NAMES[i]]-int(i==gold))**2 for i in range(6)),
                      "latency_ms":(time.perf_counter()-started)*1000})
        serving.add(out.get("model","unknown"));tokens+=int(out.get("usage",{}).get("input_tokens",0))
        print("FARROW_JEV_PROGRESS",len(score),"/",n,flush=True)
    result={"n":len(score),"served_model":sorted(serving),"input_tokens":tokens,
            "mean":{k:statistics.mean(x[k] for x in score) for k in score[0]},
            "reference":"original source-derived veterinarian consensus; 30 independent scenarios",
            "caution":"Retrospective owner triage, not clinical deployment evidence"}
    f=Path("reports/index_l3_vet/farrow_jev_aggregate.json");f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(result,indent=2)+"\n")
    print("FARROW_JEV_AGGREGATE",json.dumps(result),flush=True)
if __name__=="__main__":main()
