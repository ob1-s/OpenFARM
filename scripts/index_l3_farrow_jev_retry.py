"""Farrow Jev: retry invalid Choice responses and account for missing cases."""
import collections,json,math,os,statistics,time,urllib.request,urllib.error
from pathlib import Path
CATS=['within_1h','within_12h','within_24h','within_7d','within_14d','self_care']
RULES={'within_1h':'Seek veterinary advice within one hour.',
       'within_12h':'Seek veterinary advice within 12 hours.',
       'within_24h':'Seek veterinary advice within 24 hours.',
       'within_7d':'Seek veterinary advice within 7 days.',
       'within_14d':'Seek veterinary advice within 14 days.',
       'self_care':'No immediate veterinary advice; manage the symptoms at home.'}
def main():
    key=os.getenv('TYPESAFE_API_KEY')
    if not key:raise ValueError('Actions API secret unavailable')
    cases=sorted(json.loads(Path('reports/index_l3_farrow/cases_local_reference.json').read_text()),key=lambda c:c['id'])
    if len(cases)!=30 or len(set(c['id'] for c in cases))!=30:raise ValueError('unexpected case manifest')
    scores=[];fail=collections.Counter();tokens=0;served=set();lat=[];pred_classes=collections.Counter()
    for i,case in enumerate(cases,1):
        gold=CATS.index(case['reference'])
        body={'model':'jev-1.13.0','state':case['state'],'questions':{'urgency':{'type':'choice','instructions':'Based on this dog owner’s report, how soon should they seek professional veterinary care? Choose the safest proportionate published time interval.','criteria':RULES}}}
        request=urllib.request.Request('https://api.typesafe.ai/v1/systemone',data=json.dumps(body).encode(),method='POST',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        result=None;usage={};model=None
        for attempt in range(4):
            start=time.perf_counter()
            try:
                with urllib.request.urlopen(request,timeout=60) as response: msg=json.loads(response.read(128000))
            except urllib.error.HTTPError as e:
                if e.code not in (429,529):raise RuntimeError('Jev HTTP '+str(e.code)+'; case text suppressed') from None
                fail['rate_limit_retry']+=1;time.sleep(2**attempt);continue
            ans=msg.get('answers',{}).get('urgency',{});p=ans.get('probabilities')
            if ans.get('type')=='choice' and isinstance(p,dict) and set(p)==set(CATS) and all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in p.values()) and abs(sum(p.values())-1)<=.025:
                total=sum(p.values());result=[p[x]/total for x in CATS]
                usage=msg.get('usage',{});model=msg.get('model')
                lat.append((time.perf_counter()-start)*1000);break
            fail['invalid_schema_retry']+=1;time.sleep(.2)
        if result is None:
            fail['unscorable_cases']+=1;print('JEV_UNSCORABLE_CASE',i,'/30',flush=True);continue
        pred=max(range(6),key=lambda x:result[x]);pred_classes[CATS[pred]]+=1
        scores.append({'exact':int(pred==gold),'undertriage':int(pred>gold),'overtriage':int(pred<gold),
            'ordinal_mae':abs(pred-gold),
            'brier':sum((result[j]-(1 if j==gold else 0))**2 for j in range(6))})
        tokens+=int(usage.get('input_tokens',0));served.add(str(model))
        if i%5==0 or i==1 or i==30:print('JEV_FARROW_PROGRESS',i,'/30',flush=True)
    out={'attempted':30,'scored':len(scores),'coverage':len(scores)/30,
         'retry_diagnostics':dict(fail),'model':sorted(served),'input_tokens':tokens,
         'mean_metrics_on_scored':{k:statistics.mean(s[k] for s in scores) for k in scores[0]} if scores else {},
         'predicted_classes':dict(pred_classes),'median_latency_ms':statistics.median(lat) if lat else None,
         'reference':'30 Farrow author-scored veterinarian consensus categories; 14 individual votes unavailable',
         'limitations':'Retrospective owner-first-contact triage, no deployment validity. Unscorable predictions excluded but explicitly counted.'}
    f=Path('reports/index_l3_vet/farrow_jev_aggregate.json');f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(out,indent=2)+'\n')
    print('FARROW_JEV_FINAL',json.dumps(out,sort_keys=True),flush=True)
    if not scores:raise RuntimeError('No valid model responses')
if __name__=='__main__':main()
