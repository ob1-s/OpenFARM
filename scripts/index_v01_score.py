#!/usr/bin/env python3
"""Reference scorer for OpenFARM Decision Index v0.1.

Scoring input is DATA-FREE metadata plus model predictions and independent
human/outcome labels, never image/text state. No cross-task score averaging.

Input JSONL per line:
 {"task_id":"...", "unit_id":"...", "group_id":"animal_or_video",
  "kind":"choice"|"score"|"noul", "options":[...],
  "prediction":{"probabilities":{option:p,...}} OR {"probability":p},
  "reference":{"votes":[option,option,...]} OR {"label":option|0|1},
  "priority_order":"most_urgent_first" OPTIONAL,
  "status":"scorable"|"unscorable"|"abstain"}
Group IDs must reflect each source's scientifically justified split unit.
"""
import argparse,collections,json,math,random,statistics
from pathlib import Path

INDEX=Path('benchmarks/decision-index/v0.1/tasks.json')
def finite(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def decode(record):
    task=record['task_id'];kind=record['kind']
    unit=record['unit_id'];group=record['group_id']
    assert task and unit and group and kind in ('choice','score','noul')
    if 'state' in record or 'image' in record or 'clinical_text' in record:
        raise ValueError('Raw source states forbidden in results scorer')
    status=record.get('status','scorable')
    if status not in ('scorable','unscorable','abstain'):raise ValueError('unrecognized observation quality status')
    if status!='scorable':return {'task_id':task,'group_id':group,'unit_id':unit,'status':status}
    prediction=record['prediction'];ref=record['reference']
    if kind=='noul':
        p=prediction['probability'];y=ref['label']
        if not finite(p) or not 0<=p<=1 or y not in (0,1,False,True):
            raise ValueError('Bad binary probability/target')
        return {'task_id':task,'group_id':group,'unit_id':unit,'status':'scorable',
                'kind':kind,'brier':(p-float(y))**2,'log_loss':-(float(y)*math.log(max(p,1e-12))+
               (1-float(y))*math.log(max(1-p,1e-12))),'exact':int((p>=0.5)==bool(y)),
                'reference_label':int(y),'prediction_label':int(p>=0.5)}
    options=record['options'];pmap=prediction['probabilities']
    if len(options)<2 or len(set(options))!=len(options) or set(pmap)!=set(options):
        raise ValueError('Option/reference schema mismatch')
    probs=[pmap[x] for x in options]
    if any(not finite(x) or not 0<=x<=1 for x in probs) or abs(sum(probs)-1)>.005:
        raise ValueError('Bad multiclass probability distribution')
    probs=[x/sum(probs) for x in probs]
    votes=ref.get('votes')
    if votes is None:votes=[ref['label']]
    if not votes or any(x not in options for x in votes):raise ValueError('Missing/out-of-schema references')
    indices=[options.index(x) for x in votes]
    q=[indices.count(i)/len(indices) for i in range(len(options))]
    predicted=max(range(len(options)),key=lambda i:probs[i])
    modal=sorted(indices)[len(indices)//2] if len(indices)%2 else None
    result={'task_id':task,'unit_id':unit,'group_id':group,'status':'scorable','kind':kind,
            'n_reference_votes':len(indices),
            'brier':sum((probs[i]-q[i])**2 for i in range(len(options))),
            'log_loss':-sum(q[i]*math.log(max(probs[i],1e-12)) for i in range(len(options))),
            'mean_human_vote_variance':1-sum(x*x for x in q),
            'exact':None if modal is None else int(predicted==modal),
            'reference_label':None if modal is None else options[modal],
            'prediction_label':options[predicted]}
    if record.get('priority_order')=='most_urgent_first':
        result.update(vote_undertriage=sum(predicted>v for v in indices)/len(indices),
                      vote_overtriage=sum(predicted<v for v in indices)/len(indices),
                      vote_ordinal_mae=sum(abs(predicted-v) for v in indices)/len(indices),
                      reference_majority_immediate=int(modal==0) if modal is not None else None,
                      caught_majority_immediate=int(predicted==0 and modal==0) if modal is not None else None)
    return result
def score(rows,bootstrap=500):
    assert 0<=bootstrap<=10000
    units=set();tasks=collections.defaultdict(list);meta={}
    for row in rows:
        key=(row['task_id'],row['unit_id'])
        if key in units:raise ValueError('Duplicate task/unit ID')
        units.add(key)
        d=decode(row)
        if d['task_id'] in meta:
            if meta[d['task_id']]['kind']!=row['kind']:raise ValueError('Mixed typed heads in same task')
            if row['kind']!='noul' and meta[d['task_id']]['options']!=row['options']:
                raise ValueError('Option order changed within model task')
        else:meta[d['task_id']]={'kind':row['kind'],'options':row.get('options')}
        tasks[d['task_id']].append(d)
    valid_ids={x['id'] for x in json.loads(INDEX.read_text())['tasks']}
    if not set(tasks).issubset(valid_ids):raise ValueError('Unknown task identifier')
    rng=random.Random(20261017)
    output={}
    for task_id,items in tasks.items():
        valid=[r for r in items if r['status']=='scorable']
        group=collections.defaultdict(list)
        for row in valid:group[row['group_id']].append(row)
        if not valid:raise ValueError('No scorable rows for task '+task_id)
        numeric=('brier','log_loss','exact','vote_undertriage','vote_overtriage','vote_ordinal_mae')
        means={key:statistics.mean(x[key] for x in valid if x.get(key) is not None)
               for key in numeric if any(x.get(key) is not None for x in valid)}
        heads=[r for r in valid if r.get('reference_majority_immediate')==1]
        if heads:
            means['majority_immediate_recall']=sum(x['caught_majority_immediate'] for x in heads)/len(heads)
        if any('mean_human_vote_variance' in x for x in valid):
            means['human_vote_variance_term']=statistics.mean(x['mean_human_vote_variance'] for x in valid)
        groups=list(group)
        if bootstrap and len(groups)>=2:
            sampled=[]
            for j in range(bootstrap):
                picked=[row for g in rng.choices(groups,k=len(groups)) for row in group[g]]
                sampled.append(statistics.mean(x['brier'] for x in picked))
            sampled.sort()
            ci=[sampled[int(.025*(bootstrap-1))],sampled[int(.975*(bootstrap-1))]]
        else:ci=None
        output[task_id]={
            'attempted':len(items),'scored':len(valid),
            'unscorable':sum(x['status']=='unscorable' for x in items),
            'abstained':sum(x['status']=='abstain' for x in items),
            'n_independence_groups_claimed':len(group),
            'n_majority_immediate':len(heads),
            'scored_only_means':means,
            'cluster_bootstrap_95pct_brier':ci,
            'warning':'Group IDs must reflect proven independence; no cross-task aggregate. Urgency disagreement is not clinical outcome harm.'}
    return {'schema_version':'openfarm-decision-index-score-v0.1',
            'task_metrics':output,'cross_task_aggregate':None}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    a=p.parse_args()
    rows=[json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    result=score(rows,a.bootstrap)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('INDEX_V01_SCORE',json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
