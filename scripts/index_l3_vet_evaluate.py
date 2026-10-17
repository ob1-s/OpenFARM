#!/usr/bin/env python3
"""Case-level veterinary triage agreement metrics; expects original expert votes.

Inputs are output from index_l3_vet_build.py and predictions from an independent
model or an explicit constant baseline. Not a medical device.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

CATEGORIES = ('red','orange','yellow','green','blue')


def evaluate(cases: list[dict], predictions: dict[str, dict[str, float]]) -> dict:
    if not cases:
        raise ValueError('no patient cases')
    seen = set()
    n = 0
    brier = ordinal_mae = under = over = 0.0
    red_total = red_recognized = 0
    unanimous = 0
    for case in cases:
        cid = case['case_id']
        if cid in seen:
            raise ValueError(f'duplicate case prediction unit {cid}')
        seen.add(cid)
        votes = [v for v in case['veterinarian_votes'] if v is not None]
        if not votes:
            continue
        if cid not in predictions:
            raise ValueError(f'no prediction for labeled case {cid}')
        p = predictions[cid]
        if set(p) != set(CATEGORIES):
            raise ValueError('prediction must have five category probabilities')
        if any(not isinstance(p[k], (int,float)) or p[k] < 0 or p[k] > 1 for k in CATEGORIES):
            raise ValueError('invalid category probability')
        if abs(sum(p.values())-1) > 1e-6:
            raise ValueError('probabilities must sum to 1')
        ref = {k: sum(v == k for v in votes)/len(votes) for k in CATEGORIES}
        brier += sum((p[k]-ref[k])**2 for k in CATEGORIES)
        p_class = max(CATEGORIES, key=lambda k: p[k])
        pred_rank = CATEGORIES.index(p_class)
        for v in votes:
            v_rank = CATEGORIES.index(v)
            ordinal_mae += abs(pred_rank-v_rank)/len(votes)
            under += (pred_rank>v_rank)/len(votes)
            over += (pred_rank<v_rank)/len(votes)
        if all(v==votes[0] for v in votes):
            unanimous += 1
        if ref['red'] > 0.5:
            red_total += 1
            red_recognized += int(p_class == 'red')
        n += 1
    if n == 0:
        raise ValueError('no expert-labeled cases')
    if set(predictions) - seen:
        raise ValueError('unknown case predictions (possible join error)')
    return {'n_unique_evaluable_cases':n,'n_unanimous_cases':unanimous,
            'brier_vs_vet_vote_distribution':brier/n,
            'mean_ordinal_abs_error_vs_votes':ordinal_mae/n,
            'veterinarian_vote_weighted_undertriage_rate':under/n,
            'veterinarian_vote_weighted_overtriage_rate':over/n,
            'reference_majority_red_cases':red_total,
            'sensitivity_for_majority_red_cases':red_recognized/red_total if red_total else None,
            'note':'Retrospective clinical expert preference agreement, NOT validated patient outcome benefit.'}


def always_red(cases: list[dict]) -> dict[str,dict[str,float]]:
    return {c['case_id']:{k:float(k=='red') for k in CATEGORIES} for c in cases}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',required=True,type=Path)
    ap.add_argument('--predictions',type=Path,help='optional JSON mapping case_id to 5-category probabilities')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    cases=json.loads(args.cases.read_text())
    predictions=json.loads(args.predictions.read_text()) if args.predictions else always_red(cases)
    result=evaluate(cases,predictions)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
