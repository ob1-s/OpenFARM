#!/usr/bin/env python3
"""Original Liquid Farrow failure forensics: predeclared criteria-order and rubric tests.

30 *original* owner-facing vignettes rebuilt from checksum-pinned source.
Output per-vignette (public ID only) probabilities + aggregate no clinical text.
This is diagnostic reuse, not a new blind leaderboard.
"""
import collections,json,math,os,statistics,time,urllib.request,urllib.error
from pathlib import Path

CATS=['within_1h','within_12h','within_24h','within_7d','within_14d','self_care']
BASE={
'within_1h':'Seek veterinary care immediately, within one hour.',
'within_12h':'Seek veterinary care within 12 hours.',
'within_24h':'Seek veterinary care within 24 hours.',
'within_7d':'Seek veterinary care within seven days.',
'within_14d':'Seek veterinary care within fourteen days.',
'self_care':'Manage at home for now, without immediately seeking veterinary care.'}
BASE_INSTR='For this owner-reported dog case, how soon should the owner seek veterinary advice and care? Use only the vignette, and choose the safest proportionate option.'
CLARIFY_INSTR=("This is an OWNER AT HOME asking when to contact a veterinarian, NOT a patient already waiting in the emergency hospital. "
    "Estimate the delay after which the owner should obtain professional veterinary assessment. Use the published six urgency options. "
    "Decide from the reported signs and history only, without assuming a diagnosis.")
ORDINAL=[
    'Self-care at home without seeking veterinary advice immediately.',
    'Seek veterinary care within 14 days.',
    'Seek veterinary care within 7 days.',
    'Seek veterinary care within 24 hours.',
    'Seek veterinary care within 12 hours.',
    'Seek veterinary care within 1 hour immediately.'
]
VARIANTS=['choice_original','choice_reversed','choice_clarified','score_ordered']
def format_q(v):
    if v=='choice_original':return {'type':'choice','instructions':BASE_INSTR,'criteria':BASE}
    if v=='choice_reversed':return {'type':'choice','instructions':BASE_INSTR,'criteria':{k:BASE[k] for k in reversed(CATS)}}
    if v=='choice_clarified':return {'type':'choice','instructions':CLARIFY_INSTR,'criteria':BASE}
    if v=='score_ordered':return {'type':'score','instructions':BASE_INSTR,'criteria':ORDINAL}
    raise ValueError('not preregistered')

def parse(answer,variant):
    p=answer.get('probabilities')
    if variant=='score_ordered':
        vals=[p.get(str(i)) for i in range(6)] if isinstance(p,dict) else p
        if answer.get('type')!='score' or not isinstance(vals,list) or len(vals)!=6:return None
        vals=list(reversed(vals))
    else:
        if answer.get('type')!='choice' or not isinstance(p,dict) or set(p)!=set(CATS):return None
        vals=[p[k] for k in CATS]
    if any(not isinstance(v,(float,int)) or v<0 or not math.isfinite(v) for v in vals):return None
    den=sum(vals)
    if not .975<=den<=1.025:return None
    return [v/den for v in vals]

def send(url,case,model,variant,secret):
    q=format_q(variant);body={'state':case['state'],'questions':{'urgency':q}}
    if url.startswith('https'):body['model']=model
    head={'Content-Type':'application/json'}
    if secret:head['Authorization']='Bearer '+secret
    request=urllib.request.Request(url,data=json.dumps(body).encode('utf-8'),headers=head,method='POST')
    for retry in range(3):
        try:
            start=time.perf_counter()
            with urllib.request.urlopen(request,timeout=70) as response:
                result=json.loads(response.read(300000))
            decoded=parse(result.get('answers',{}).get('urgency',{}),variant)
            if decoded is not None:return decoded,1000*(time.perf_counter()-start),'valid'
        except urllib.error.HTTPError as e:
            if e.code not in (429,502,503):return None,None,'http_'+str(e.code)
        except urllib.error.URLError:
            pass
        time.sleep(.3*(retry+1))
    return None,None,'invalid_or_transport'

def main():
    cases=json.loads(Path('reports/index_l3_farrow/cases_local_reference.json').read_text())
    if len(cases)!=30 or len({c['id'] for c in cases})!=30:raise ValueError('30 original public vignettes required')
    cases=sorted(cases,key=lambda x:x['id'])
    target=os.environ.get('OPENFARM_MODEL_BACKEND','liquid')
    if target not in ('liquid','jev'):raise ValueError('unsupported backend')
    api='http://127.0.0.1:8080/v1/systemone' if target=='liquid' else 'https://api.typesafe.ai/v1/systemone'
    secret=None if target=='liquid' else os.environ.get('TYPESAFE_API_KEY')
    if target=='jev' and not secret:raise ValueError('Jev API secret required')
    model='LiquidAI/d1-omni-600M-GGUF:Q8_0' if target=='liquid' else 'jev-1.14.0'
    detail={};group={}
    for variant in VARIANTS:
        rows=[];errors=collections.Counter();lat=[]
        for i,c in enumerate(cases,1):
            p,ms,state=send(api,c,model,variant,secret)
            if p is None:
                errors[state]+=1
                rows.append({'case_id':c['id'],'reference':c['reference'],'status':state})
                continue
            gold=CATS.index(c['reference']);prediction=max(range(6),key=lambda i:p[i])
            rows.append({'case_id':c['id'],'reference':c['reference'],'status':'valid',
                  'prediction':CATS[prediction],'probabilities':dict(zip(CATS,p)),
                  'brier':sum((p[k]-int(k==gold))**2 for k in range(6)),
                  'undertriage':int(prediction>gold),'overtriage':int(prediction<gold),
                  'exact':int(prediction==gold)})
            lat.append(ms)
            if i%10==0:print('FARROW_FORENSICS_PROGRESS',target,variant,i,'/30',flush=True)
        valid=[r for r in rows if r['status']=='valid']
        out={'attempted':len(rows),'valid':len(valid),'validity_fraction':len(valid)/30,
             'errors':dict(errors),'predicted_class_histogram':dict(collections.Counter(r['prediction'] for r in valid)),
             'majority_critical_1h_count':sum(x['reference']==CATS[0] for x in valid),
             'correct_1h':sum(x['reference']==CATS[0] and x['prediction']==CATS[0] for x in valid),
             'mean':{key:statistics.mean(r[key] for r in valid) for key in ('exact','undertriage','overtriage','brier')} if valid else {},
             'median_request_ms':statistics.median(lat) if lat else None}
        group[variant]=out;detail[variant]=rows
    # Important confound: reverse option order changes no semantic key/text, so any
    # change is a pure presentation-order effect.
    def flip_summary(a,b):
        rows_a={r['case_id']:r for r in detail[a] if r['status']=='valid'}
        rows_b={r['case_id']:r for r in detail[b] if r['status']=='valid'}
        keys=set(rows_a)&set(rows_b)
        return {'paired_n':len(keys),
          'changed_top_class':sum(rows_a[k]['prediction']!=rows_b[k]['prediction'] for k in keys),
          'changed_fraction':sum(rows_a[k]['prediction']!=rows_b[k]['prediction'] for k in keys)/len(keys) if keys else None,
          'mean_brier_delta_b_minus_a':statistics.mean(rows_b[k]['brier']-rows_a[k]['brier'] for k in keys) if keys else None}
    result={'backend':target,'actual_weights':model,'n_original_vignettes':30,
      'dataset':'Farrow original 30 independently authored owner scenarios; author-coded vet-panel consensus',
      'predeclared_variants':VARIANTS,
      'group_metrics':group,
      'paired_original_vs_reversed':flip_summary('choice_original','choice_reversed'),
      'paired_original_vs_clarified':flip_summary('choice_original','choice_clarified'),
      'research_only':'All 30 cases have been scored previously. This is a forensic stability diagnostic, not a new blinded ranking.'}
    out=Path('reports/decision-index');out.mkdir(parents=True,exist_ok=True)
    (out/('farrow_'+target+'_forensics.json')).write_text(json.dumps(result,indent=2)+'\n')
    (out/('farrow_'+target+'_public_case_predictions.json')).write_text(json.dumps(detail,indent=2)+'\n')
    print('FARROW_FORENSICS_SUMMARY',json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
