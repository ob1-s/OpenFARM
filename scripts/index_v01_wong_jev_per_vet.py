#!/usr/bin/env python3
"""Actual Jev vs each original veterinarian on Wong's 102 test clinical cases.

Analyzes multi-rater clinical severity thresholds, not one unquestionable truth.
No case text, credentials or per-case predictions written to artifacts/logs.
"""
import collections,json,math,os,statistics
from pathlib import Path
from scripts.index_l3_wong_jev_private_runner import load,LABELS,ask as choice_ask
from scripts.index_l3_wong_jev_score_private import ask as score_ask

def entropy(q):
    return -sum(float(v)*math.log(v) for v in q if v>0)
def pearson(a,b):
    if len(a)<3:return None
    ma=statistics.mean(a);mb=statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    sa=sum((x-ma)**2 for x in a);sb=sum((y-mb)**2 for y in b)
    return cov/math.sqrt(sa*sb) if sa and sb else None
def per_model(cases,model,key,kind):
    fn=choice_ask if kind=='choice' else score_ask
    rec=[];invalid=0;served=set()
    for ix,case in enumerate(cases,1):
        p,lat,usage,served_model=fn(case,key,model)
        if p is None:
            invalid+=1;continue
        if not isinstance(p,list) or len(p)!=5:raise ValueError('Invalid model probability vector')
        votes=[LABELS.index(x) for x in case['vet_votes']]
        q=[votes.count(j)/3 for j in range(5)]
        pred=max(range(5),key=lambda j:p[j])
        true_med=sorted(votes)[1]
        rec.append({'votes':votes,'prob':p,'pred':pred,'med':true_med,
                    'human_uncertainty':entropy(q),'model_uncertainty':entropy(p),
                    'brier_q':sum((p[j]-q[j])**2 for j in range(5)),
                    'vote_variance':1-sum(x*x for x in q),
                    'vet_brier':[sum((p[j]-int(j==v))**2 for j in range(5)) for v in votes]})
        served.add(served_model)
        if ix%25==0:print('WONG_RATER_MODEL_PROGRESS',kind,ix,'/102',flush=True)
    if not rec:raise RuntimeError('No valid model outputs')
    by_vet=[]
    for vet in range(3):
        by_vet.append({'rater':vet+1,
            'immediate_source_cases':sum(r['votes'][vet]==0 for r in rec),
            'exact_fraction':statistics.mean(r['pred']==r['votes'][vet] for r in rec),
            'pred_more_urgent_fraction':statistics.mean(r['pred']<r['votes'][vet] for r in rec),
            'pred_less_urgent_fraction':statistics.mean(r['pred']>r['votes'][vet] for r in rec),
            'mean_signed_pred_rank_minus_vet_rank':statistics.mean(r['pred']-r['votes'][vet] for r in rec),
            'mean_onehot_brier':statistics.mean(r['vet_brier'][vet] for r in rec)})
    reference_groups={
        'unanimous_immediate':[r for r in rec if all(v==0 for v in r['votes'])],
        'majority_immediate':[r for r in rec if r['med']==0],
        'any_vet_immediate':[r for r in rec if 0 in r['votes']],
        'no_vet_immediate':[r for r in rec if 0 not in r['votes']],
        'unanimous_all_labels':[r for r in rec if len(set(r['votes']))==1],
        'any_disagreement':[r for r in rec if len(set(r['votes']))>1]
    }
    slice_scores={name:{'cases':len(rows),'fraction_model_immediate':statistics.mean(r['pred']==0 for r in rows) if rows else None,
       'mean_p_immediate':statistics.mean(r['prob'][0] for r in rows) if rows else None,
       'mean_brier_q':statistics.mean(r['brier_q'] for r in rows) if rows else None}
         for name,rows in reference_groups.items()}
    by_t=[]
    for t in (.05,.1,.2,.3,.5,.7):
        escalated=[r for r in rec if r['prob'][0]>=t]
        urgent=reference_groups['majority_immediate']
        by_t.append({'p_immediate_threshold':t,'escalations':len(escalated),
             'majority_immediate_detected':sum(r['prob'][0]>=t for r in urgent),'majority_immediate_total':len(urgent),
             'any_vet_immediate_detected':sum(r['prob'][0]>=t for r in reference_groups['any_vet_immediate'])})
    return {'model':model,'type':kind,'n':len(rec),'invalid':invalid,'served_models':sorted(served),
        'per_vet':by_vet,'critical_reference_slices':slice_scores,
        'mean_brier_vs_3vet_frequency':statistics.mean(x['brier_q'] for x in rec),
        'mean_human_reference_noise_floor':statistics.mean(x['vote_variance'] for x in rec),
        'mean_onehot_vet_brier':statistics.mean(statistics.mean(x['vet_brier']) for x in rec),
        'brier_algebra_residual':statistics.mean(statistics.mean(x['vet_brier'])-x['brier_q']-x['vote_variance'] for x in rec),
        'model_entropy_by_expert_disagreement_correlation':pearson([x['human_uncertainty'] for x in rec],[x['model_uncertainty'] for x in rec]),
        'posthoc_immediate_threshold_curve_diagnostic_only':by_t,
        'note':'Threshold curve computed on reused test cases is diagnostic, not calibration- or prospective-approved.'}
def main():
    original=load()
    if original is None:raise RuntimeError('private Wong test source missing')
    key=os.getenv('TYPESAFE_API_KEY')
    if not key:raise RuntimeError('TypeSafe key missing')
    cases=sorted(original['cases'],key=lambda r:r['id'])
    reports=[per_model(cases,'jev-1.14.0',key,kind) for kind in ('choice','score')]
    out={'n_cases':102,'source_sha256':original['source_sha256'],
         'rater_votes_per_case':3,'analyses':reports,
         'limitations':'Retrospective 102 reused Wong case units with unknown unique patient IDs; 9 majority-immediate; rater-specific agreement not clinical outcome correctness.'}
    file=Path('reports/decision-index/wong_jev_vs_individual_veterinarians.json')
    file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(out,indent=2)+'\n')
    print('WONG_JEV_RATER_REFERENCE_RESULT',json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
