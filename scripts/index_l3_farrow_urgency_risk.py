"""Source-free clinical risk analysis on four Farrow Jev benchmarks, no case text."""
import collections,json
from pathlib import Path
ROOT=Path('reports/index_l3_vet')
KIND={
'choice_1_13':'farrow_jev_jev-1_13_0_case_predictions.json',
'choice_1_14':'farrow_jev_jev-1_14_0_case_predictions.json',
'score_1_13':'farrow_jev-1_13_0_score_predictions.json',
'score_1_14':'farrow_jev-1_14_0_score_predictions.json'}
CATS=['within_1h','within_12h','within_24h','within_7d','within_14d','self_care']
def main():
    results={}
    for task,fn in KIND.items():
        rows=json.loads((ROOT/fn).read_text())
        if len(rows)!=30 or len({r['vignette_id'] for r in rows})!=30:
            raise ValueError('corrupted case-level predictions')
        ncritical=hits=0
        false_urgent=0
        severe_under=0
        predicted=collections.Counter()
        risk_for_urgent=[]
        for r in rows:
            if r.get('status','valid')!='valid':continue
            gold=CATS.index(r['reference'])
            p=CATS.index(r['prediction'])
            predicted[CATS[p]]+=1
            if gold==0:
                ncritical+=1
                hits+=int(p==0)
            if gold>=2 and p==0:false_urgent+=1
            if gold<=1 and p>=3:severe_under+=1
            if gold<=1:risk_for_urgent.append(int(p>gold))
        results[task]={'n_scored':sum(predicted.values()),
          'critical_vignettes':ncritical,
          'critical_within_1h_sensitivity':hits/ncritical if ncritical else None,
          'critical_undertriaged':ncritical-hits,
          'immediate_predictions_on_24h_or_less_urgent_reference':false_urgent,
          'urgency_within_12h_missed_to_7d_or_later':severe_under,
          'undertriage_among_within_12h_references':sum(risk_for_urgent),
          'within_12h_reference_count':len(risk_for_urgent),
          'prediction_counts':dict(predicted)}
    result={'n_unique_cases':30,'source_gold':'author-scored veterinarian consensus',
        'risk_metrics':results,
        'caution':'Only five source cases have within-hour consensus. Patient outcome impacts not observed.'}
    target=ROOT/'farrow_jev_urgency_risk_report.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('FARROW_CLINICAL_RISK',json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
