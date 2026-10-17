"""Inspect only the response envelope on the previously failing Farrow item."""
import json,os,urllib.request
from pathlib import Path
from index_l3_farrow_jev_eval import RULES
cases=sorted(json.loads(Path("reports/index_l3_farrow/cases_local_reference.json").read_text()),key=lambda c:c["id"])
case=cases[15]
body={"model":"jev-1.13.0","state":case["state"],"questions":{"urgency":{"type":"choice","instructions":"For this owner-reported dog case, how soon should the owner seek veterinary advice and care? Use only the vignette, and choose the safest proportionate option.","criteria":RULES}}}
req=urllib.request.Request("https://api.typesafe.ai/v1/systemone",method="POST",data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+os.environ["TYPESAFE_API_KEY"],"Content-Type":"application/json"})
with urllib.request.urlopen(req,timeout=50) as r: reply=json.loads(r.read(128000))
ans=reply.get("answers",{}).get("urgency",{})
probs=ans.get("probabilities")
print("JEV_RESPONSE_SCHEMA_ONLY",json.dumps({"answer_type":ans.get("type"),"answer_keys":sorted(ans),"probability_keys":sorted(probs) if isinstance(probs,dict) else type(probs).__name__,"prob_sum":sum(probs.values()) if isinstance(probs,dict) else None,"model":reply.get("model")}))
