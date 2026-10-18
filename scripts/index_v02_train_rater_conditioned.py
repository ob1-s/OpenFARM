#!/usr/bin/env python3
"""Rater-aware small specialist research test, source pinned and training-only.

Requires user-supplied original Wong Case_data.xlsx OUTSIDE repository.
Preserves the 171 train / 67 calibration / 102 exploratory test case split.
Trains on three original vet votes, chooses settings exclusively on calibration,
and logs only aggregate metrics; no personal clinical case text or model weights.
"""
import argparse,json,math,os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wong_benchmark import parse,splits,CATS

def make_vote_matrix(rows):
    return np.asarray([[sum(x==k for x in r['vet_votes'])/3 for k in CATS] for r in rows])
def brier(p,q):
    return float(np.mean(np.sum((p-q)**2,axis=1)))
def votes_in_order(rows):
    return np.array([CATS.index(x) for r in rows for x in r['vet_votes']])
def model_input(x,scale,train):
    if scale==0:return x[np.repeat(np.arange(x.shape[0]),3)] if train else x
    if train:
        x=x[np.repeat(np.arange(x.shape[0]),3)]
        raters=np.tile(np.arange(3),x.shape[0]//3)
        o=csr_matrix((np.full(len(raters),scale),(np.arange(len(raters)),raters)),shape=(len(raters),3))
        return hstack([x,o],format='csr')
    return [hstack([x,csr_matrix((np.full(x.shape[0],scale),(np.arange(x.shape[0]),np.full(x.shape[0],r))),shape=(x.shape[0],3))],format='csr') for r in range(3)]
def predict(model,x,scale):
    z=model_input(x,scale,False)
    if scale==0:return model.predict_proba(z)
    return np.mean([model.predict_proba(zz) for zz in z],axis=0)
def choose_threshold(cal,p):
    positive=np.array([i for i,c in enumerate(cal) if c['median']==CATS[0]])
    if len(positive)!=6:raise ValueError('Original 6 urgent calibration cases changed')
    candidates=[]
    for threshold in np.unique(np.r_[0,1,p[:,0]])[::-1]:
        flags=p[:,0]>=threshold
        if sum(flags[positive])>=5:candidates.append((sum(flags),-threshold))
    n,negative=min(candidates)
    return float(-negative),int(n)
def policy(rows,p,t):
    prediction=np.argmax(p,axis=1)
    prediction[p[:,0]>=t]=0
    med=[CATS.index(r['median']) for r in rows]
    votes=[[CATS.index(v) for v in r['vet_votes']] for r in rows]
    return {'exact_vet_median':float(np.mean(prediction==med)),
            'vote_undertriage':float(np.mean([sum(a>v for v in vs)/3 for a,vs in zip(prediction,votes)])),
            'vote_overtriage':float(np.mean([sum(a<v for v in vs)/3 for a,vs in zip(prediction,votes)])),
            'immediate_median_reference_n':sum(m==0 for m in med),
            'immediate_median_detected':sum(a==0 and m==0 for a,m in zip(prediction,med)),
            'cases_sent_immediate':int(np.sum(prediction==0)),
            'any_vet_immediate_n':sum(0 in vs for vs in votes),
            'any_vet_immediate_detected':sum(a==0 and 0 in vs for a,vs in zip(prediction,votes))}
def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--source',type=Path,required=True)
    arg.add_argument('--report',type=Path,required=True)
    args=arg.parse_args()
    cases,meta=parse(args.source)
    part,_=splits(cases)
    train=[x for x in cases if part[x['id']]=='train']
    cal=[x for x in cases if part[x['id']]=='cal']
    test=[x for x in cases if part[x['id']]=='test']
    assert (len(train),len(cal),len(test))==(171,67,102)
    v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=12000,sublinear_tf=True,strip_accents='unicode')
    Xt=v.fit_transform([c['state'] for c in train])
    Xv=v.transform([c['state'] for c in cal])
    Xz=v.transform([c['state'] for c in test])
    Qv=make_vote_matrix(cal);Qz=make_vote_matrix(test);Y=votes_in_order(train)
    prior=make_vote_matrix(train).mean(axis=0)
    trials=[]
    for scale in (0.,.3,1.,3.):
        for C in (.3,3.):
            m=LogisticRegression(C=C,max_iter=600)
            m.fit(model_input(Xt,scale,True),Y)
            probs=predict(m,Xv,scale)
            for shrink in (0.,.15,.3):
                pv=(1-shrink)*probs+shrink*prior
                threshold,n=choose_threshold(cal,pv)
                trials.append({'rater_strength':scale,'C':C,'shrink':shrink,
                  'validation_brier':brier(pv,Qv),'validation_immediate_escalations':n,
                  'threshold':threshold,'model':m})
    winner=min(trials,key=lambda v:(v['validation_brier'],v['validation_immediate_escalations'],
                                   v['rater_strength'],v['C'],v['shrink']))
    out=(1-winner['shrink'])*predict(winner['model'],Xz,winner['rater_strength'])+winner['shrink']*prior
    result={'source_sha256':meta['source_sha256'],'train':171,'cal':67,'exploratory_test':102,
            'n_candidate_settings':len(trials),
            'selection_rule':'Brier on 67 calibration cases only, secondary fewer immediate escalations',
            'selected':{k:float(winner[k]) for k in ('rater_strength','C','shrink','validation_brier','validation_immediate_escalations','threshold')},
            'exploratory_test_results':{'brier_to_three_vets':brier(out,Qz),'policy':policy(test,out,winner['threshold'])},
            'warnings':['Source lacks patient identity/site heldout','Never interpret clinician votes as treatment outcome',
                 'Existing test repeatedly examined; no new independent benchmarking',
                 'Raw clinical text, vocabulary and model parameters deliberately not persisted']}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print('OPENFARM_RATER_CONDITIONED_RESULT',json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
