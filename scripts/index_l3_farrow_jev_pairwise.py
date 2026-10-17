"""Paired original Farrow 30-vignette Jev model comparison; no case text."""
import json,random,statistics
from pathlib import Path
CATS=['within_1h','within_12h','within_24h','within_7d','within_14d','self_care']
ROOT=Path('reports/index_l3_vet')
def read(ver):
    p=ROOT/('farrow_jev_'+ver+'_case_predictions.json')
    rows=json.loads(p.read_text())
    return {x['vignette_id']:x for x in rows}
def brier(x):
    return sum((x['probabilities'][c]-int(x['reference']==c))**2 for c in CATS)
def main():
    a=read('jev-1_13_0');b=read('jev-1_14_0')
    if len(a)!=30 or set(a)!=set(b):raise ValueError('not paired 30 independent vignettes')
    keys=sorted(a)
    valid=[id for id in keys if a[id]['status']=='valid' and b[id]['status']=='valid']
    diff=[brier(b[id])-brier(a[id]) for id in valid]
    da=[int(a[id]['prediction']==a[id]['reference']) for id in valid]
    db=[int(b[id]['prediction']==b[id]['reference']) for id in valid]
    rng=random.Random(20261017); samples=[]
    for i in range(10000):
        subset=[rng.choice(diff) for _ in diff]
        samples.append(statistics.mean(subset))
    samples.sort()
    out={
        'n_paired':len(valid),'n_all':len(keys),
        'brier_jev_1_13':statistics.mean(brier(a[k]) for k in valid),
        'brier_jev_1_14':statistics.mean(brier(b[k]) for k in valid),
        'delta_brier_1_14_minus_1_13':statistics.mean(diff),
        'case_bootstrap_95pct_delta_brier':[samples[250],samples[9750]],
        'accuracy_jev_1_13':statistics.mean(da),'accuracy_jev_1_14':statistics.mean(db),
        'correct_both':sum(x and y for x,y in zip(da,db)),
        'only_1_13_correct':sum(x and not y for x,y in zip(da,db)),
        'only_1_14_correct':sum(y and not x for x,y in zip(da,db)),
        'wrong_both':sum(not x and not y for x,y in zip(da,db)),
        'uniform_six_way_brier':5/6,
        'reference_class_prior_brier_in_sample':1-sum((sum(a[k]['reference']==c for k in keys)/len(keys))**2 for c in CATS),
        'warning':'Tiny 30-item retrospective set; paired bootstrap is exploratory and ignores any source-clinic sampling bias. Same task repeatedly evaluated; no final benchmark tuning.'
    }
    target=ROOT/'farrow_jev_paired_comparison.json';target.write_text(json.dumps(out,indent=2)+'\n')
    print('FARROW_JEV_PAIRED',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
