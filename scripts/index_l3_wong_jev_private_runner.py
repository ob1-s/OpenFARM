#!/usr/bin/env python3
"""Wong heldout veterinarian triage Jev evaluation from GitHub Actions secrets.

WONG_TEST_CASES_B64 carries 102 user-provided cases privately (gzipped JSON).
Nothing case-level or credential-related is written to the artifact/logs.
"""
import base64,collections,gzip,hashlib,json,math,os,random,statistics,time
import urllib.error,urllib.request
from pathlib import Path

LABELS=['Immediate','15 mins','30-60 mins','120 mins','240 mins']
KEYS=['immediate','within_15min','within_30to60min','within_120min','within_240min']
RUBRIC={
'immediate':'Immediate veterinarian assessment, no acceptable delay; severe unstable emergency.',
'within_15min':'Veterinarian assessment within 15 minutes; very urgent.',
'within_30to60min':'Veterinarian assessment within 30 to 60 minutes; urgent.',
'within_120min':'Veterinarian assessment within 120 minutes; stable non-critical presentation.',
'within_240min':'Veterinarian assessment within 240 minutes; lowest urgency.'
}
QUESTION={'type':'choice',
'instructions':'For this dog already presenting to an emergency veterinary hospital, use only the intake complaint, history and clinical signs to select a proportionate maximum wait for the veterinarian to assess it. Balance dangerous delay against unnecessary emergency escalation.',
'criteria':RUBRIC}
DOI='10.25910/rbk1-p833'
SRC='4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027'

def load():
    encoded=os.getenv('WONG_TEST_CASES_B64')
    if not encoded:
        print('WONG_SOURCE_SECRET_MISSING: safely skipped all model calls.')
        return None
    raw=gzip.decompress(base64.b64decode(encoded,validate=True))
    d=json.loads(raw)
    cases=d.get('cases',[])
    if d.get('source_sha256')!=SRC or d.get('split')!='test' or len(cases)!=102:
        raise ValueError('Wong pinned source/split validation failed')
    if len({c['id'] for c in cases})!=102:raise ValueError('duplicate unit in heldout file')
    if any(not c.get('state') or len(c.get('vet_votes',[]))!=3 or not all(v in LABELS for v in c['vet_votes']) for c in cases):
        raise ValueError('malformed independent expert references')
    prior=d.get('train_vet_prior',{})
    if set(prior)!=set(LABELS) or abs(sum(prior.values())-1)>.00001:
        raise ValueError('train-only baseline probabilities invalid')
    return d

def ask(case,secret,model):
    body={'model':model,'state':case['state'],'questions':{'vet_triage':QUESTION}}
    request=urllib.request.Request(
        'https://api.typesafe.ai/v1/systemone',
        data=json.dumps(body,ensure_ascii=False).encode('utf-8'),
        method='POST',
        headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json',
                 'Idempotency-Key':hashlib.sha256((model+case['id']+'openfarm-wong').encode()).hexdigest()})
    for retry in range(4):
        started=time.perf_counter()
        try:
            with urllib.request.urlopen(request,timeout=50) as r:
                response=json.loads(r.read(128000))
            a=response.get('answers',{}).get('vet_triage',{})
            p=a.get('probabilities')
            if a.get('type')=='choice' and isinstance(p,dict) and set(p)==set(KEYS) and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in p.values()):
                den=sum(p.values())
                if .975<=den<=1.025:
                    return [p[k]/den for k in KEYS],(time.perf_counter()-started)*1000,response.get('usage',{}),str(response.get('model'))
        except urllib.error.HTTPError as e:
            if e.code not in (429,529,502,503):raise RuntimeError('Jev API HTTP '+str(e.code)) from None
        except urllib.error.URLError:
            pass
        time.sleep(min(.25*2**retry,3))
    return None,None,{},None

def metrics(cases,probs):
    items=[];emergencies=0;emergencies_hit=0
    for case in cases:
        x=probs.get(case['id'])
        if x is None:continue
        r=[LABELS.index(y) for y in case['vet_votes']]
        q=[sum(v==i for v in r)/3 for i in range(5)]
        pred=max(range(5),key=lambda i:x[i])
        med=sorted(r)[1]
        if med==0:
            emergencies+=1
            emergencies_hit+=int(pred==0)
        items.append({'brier':sum((x[i]-q[i])**2 for i in range(5)),
             'median_exact':int(pred==med),
             'vote_undertriage':sum(pred>v for v in r)/3,
             'vote_overtriage':sum(pred<v for v in r)/3,
             'vote_ordinal_mae':sum(abs(pred-v) for v in r)/3,
             'logloss_to_votes':-sum(q[i]*math.log(max(x[i],1e-12)) for i in range(5))})
    res={'scored':len(items),'coverage':len(items)/102,
         'mean':{key:statistics.mean(x[key] for x in items) for key in items[0]} if items else {},
         'majority_immediate_test_cases_scored':emergencies,
         'majority_immediate_sensitivity':emergencies_hit/emergencies if emergencies else None}
    return res,items

def boot(a,b):
    vals=[a[i]['brier']-b[i]['brier'] for i in range(len(a))]
    rng=random.Random(170102)
    bmeans=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for j in range(4000))
    return {'brier_delta_first_minus_second':statistics.mean(vals),
            'case_bootstrap_95pct':[bmeans[100],bmeans[3900]]}

def main():
    d=load()
    if d is None:return
    secret=os.getenv('TYPESAFE_API_KEY')
    if not secret:raise RuntimeError('TYPESAFE_API_KEY missing; no inference')
    cases=sorted(d['cases'],key=lambda c:c['id'])
    outputs={};costs={};model_summaries={}
    for model in ('jev-1.13.0','jev-1.14.0'):
        predictions={};latencies=[];tokens=collections.Counter();served=set();invalid=0
        for j,case in enumerate(cases,1):
            p,ms,usage,name=ask(case,secret,model)
            if p is None:
                invalid+=1
                continue
            predictions[case['id']]=p
            latencies.append(ms)
            tokens.update({'input_tokens':int(usage.get('input_tokens',0)),
                           'output_tokens':int(usage.get('output_tokens',0))})
            served.add(name)
            if j%25==0:print('WONG_JEV_PROGRESS',model,j,'/',len(cases),flush=True)
        res,rows=metrics(cases,predictions)
        res.update(model_requested=model,models_served=sorted(served),invalid_responses=invalid,
                   median_latency_ms=statistics.median(latencies) if latencies else None,
                   usage=dict(tokens))
        model_summaries[model]=res
        outputs[model]=predictions
    # Fair paired metrics on cases scored by both (excludes failed predictions but records coverage above).
    common=[c for c in cases if c['id'] in outputs['jev-1.13.0'] and c['id'] in outputs['jev-1.14.0']]
    m13,r13=metrics(common,outputs['jev-1.13.0'])
    m14,r14=metrics(common,outputs['jev-1.14.0'])
    prior=[d['train_vet_prior'][n] for n in LABELS]
    prior_stats,_=metrics(cases,{c['id']:prior for c in cases})
    urgent=[1.,0.,0.,0.,0.]
    always_stats,_=metrics(cases,{c['id']:urgent for c in cases})
    result={'source_doi':DOI,'original_sha256':SRC,
        'n_unique_cases':102,'three_clinician_ratings_per_case':True,
        'split':'pinned_case_exclusive_test; animal identity unavailable',
        'model_scores':model_summaries,'paired_n':len(common),
        'paired_brier':boot(r14,r13) if common else None,
        'train_prior_baseline':prior_stats,'always_immediate_baseline':always_stats,
        'note':'Research-only retrospective veterinary hospital queue priority; not real-world outcome safety. Predictions and case text not published.'}
    dest=Path('reports/index_l3_vet/wong_jev_secret_heldout_aggregate.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print('WONG_JEV_SUMMARY',json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
