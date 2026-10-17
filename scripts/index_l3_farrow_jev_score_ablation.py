"""Farrow native ordinal Score vs Choice exploratory ablation (original 30 cases).

Does not print vignette text, owner records, credentials, or individual predictions.
No clinical deployment claims; panel consensus is 1 category per vignette.
"""
import collections,json,math,os,statistics,urllib.request,time
from pathlib import Path
URGENCY=['within_1h','within_12h','within_24h','within_7d','within_14d','self_care']
SCORE_LEVELS=[
 'Self-care at home for now without promptly contacting the veterinarian',
 'Seek veterinary advice and care within 14 days',
 'Seek veterinary advice and care within 7 days',
 'Seek veterinary advice and care within 24 hours',
 'Seek veterinary advice and care within 12 hours',
 'Seek veterinary advice and care immediately, within 1 hour'
]
def main():
    model=os.getenv('FARROW_SCORE_MODEL','jev-1.14.0')
    if model not in ('jev-1.13.0','jev-1.14.0'):raise ValueError('unapproved version')
    secret=os.getenv('TYPESAFE_API_KEY')
    if not secret:raise ValueError('TYPESAFE_API_KEY unavailable')
    cases=sorted(json.loads(Path('reports/index_l3_farrow/cases_local_reference.json').read_text()),key=lambda x:x['id'])
    if len(cases)!=30:raise ValueError('case count mismatch')
    counts=collections.Counter();rows=[];records=[];errors=collections.Counter()
    for i,case in enumerate(cases,1):
        request={'model':model,'state':case['state'],
        'questions':{'urgency_score':{'type':'score','instructions':'Rate how urgently the owner should seek veterinary advice and care based only on the symptoms in this vignette. Use this ordered scale from least urgent to most urgent.','criteria':SCORE_LEVELS}}}
        req=urllib.request.Request('https://api.typesafe.ai/v1/systemone',method='POST',data=json.dumps(request).encode(),
            headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=60) as r: response=json.loads(r.read(128000))
            ans=response.get('answers',{}).get('urgency_score',{})
            p=ans.get('probabilities')
            if isinstance(p,dict):
                values=[p.get(str(i)) for i in range(6)]
            elif isinstance(p,list):
                values=p
            else:values=[]
            if ans.get('type')!='score' or len(values)!=6 or not all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in values) or abs(sum(values)-1)>.03:
                errors['invalid_schema']+=1
                print('FARROW_SCORE_SCHEMA_DIAG',json.dumps({'type':ans.get('type'),'probabilities_kind':type(p).__name__,'answer_keys':sorted(ans),'valid_n':len(values)}),flush=True)
                continue
            prob=list(reversed([x/sum(values) for x in values]))
            label=URGENCY.index(case['reference']);pred=max(range(6),key=lambda x:prob[x])
            counts[URGENCY[pred]]+=1
            records.append({'vignette_id':case['id'],'reference':case['reference'],'prediction':URGENCY[pred],'probabilities':dict(zip(URGENCY,prob))})
            rows.append({'exact':int(label==pred),'undertriage':int(pred>label),'overtriage':int(pred<label),
                'ordinal_mae':abs(pred-label),'brier':sum((prob[j]-int(j==label))**2 for j in range(6)),
                'logloss':-math.log(max(prob[label],1e-12))})
        except Exception as e:
            errors['api_failure']+=1
            print('FARROW_SCORE_ERROR type='+type(e).__name__,flush=True)
        if i%10==0:print('FARROW_SCORE_PROGRESS',i,'/30',flush=True)
    out={'model':model,'primitive':'Score','attempted':30,'scored':len(rows),
         'mean':{k:statistics.mean(row[k] for row in rows) for k in rows[0]} if rows else {},
         'predicted_classes':dict(counts),'errors':dict(errors),
         'research_only':True,'note':'Same public clinical cases; Score-specific wording and ordinal criteria; not final held-out result.'}
    dest=Path('reports/index_l3_vet/farrow_'+model.replace('.','_')+'_score_ablation.json');dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2)+'\n')
    (dest.parent/('farrow_'+model.replace('.','_')+'_score_predictions.json')).write_text(json.dumps(records,indent=2)+'\n')
    print('FARROW_SCORE_FINAL',json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
