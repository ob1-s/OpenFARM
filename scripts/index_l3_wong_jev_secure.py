#!/usr/bin/env python3
"""Wong Jev CI; sealed source subset, never logs case text or secrets."""
import argparse,base64,hashlib,json,math,os,statistics,time,urllib.request,urllib.error
from pathlib import Path
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey,X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

CONTEXT=b'OpenFARM Wong X25519 transport v1 | ob1-s/OpenFARM'
MAGIC=b'OFWONG1'
SEALED=Path('data/index_l3_vet/wong_test_subset.sealed')
CATS=['Immediate','15 mins','30-60 mins','120 mins','240 mins']
KEYS=['immediate','within_15min','within_30to60min','within_120min','within_240min']
CRITERIA={
 'immediate':'Immediate assessment; highest acuity; cannot safely wait',
 'within_15min':'Very urgent; veterinarian within 15 minutes',
 'within_30to60min':'Urgent assessment within 30 to 60 minutes',
 'within_120min':'Stable; veterinarian within approximately 120 minutes',
 'within_240min':'Lowest urgency; veterinarian within approximately 240 minutes'
}
QUESTION={'type':'choice','instructions':('Retrospective veterinary emergency triage: the dog has already arrived at the emergency hospital. Based only on presenting complaint, history and intake signs, what is the appropriate priority/maximum waiting time for a veterinarian? Respond with the most proportionate safe urgency category.'),'criteria':CRITERIA}
ENDPOINT='https://api.typesafe.ai/v1/systemone'

def private_key(secret):
    seed=HKDF(algorithm=hashes.SHA256(),length=32,salt=b'OpenFARM Wong v1 purpose-specific seed',info=CONTEXT).derive(secret.encode())
    return X25519PrivateKey.from_private_bytes(seed)
def decrypt(blob_str,secret):
    b=base64.b64decode(blob_str.strip(),validate=True)
    if b[:7]!=MAGIC:raise ValueError('bad transport envelope')
    pub,salt,nonce,cipher=b[7:39],b[39:55],b[55:67],b[67:]
    k=HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,info=CONTEXT).derive(private_key(secret).exchange(X25519PublicKey.from_public_bytes(pub)))
    return AESGCM(k).decrypt(nonce,cipher,MAGIC)

def call(case,token,model):
    state=case['state']
    request={'model':model,'state':state,'questions':{'vet_triage':QUESTION}}
    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
    req=urllib.request.Request(ENDPOINT,data=json.dumps(request,ensure_ascii=False).encode(),headers=headers,method='POST')
    for retry in range(3):
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=50) as r:
                if r.status!=200:raise RuntimeError('unexpected status')
                response=json.loads(r.read(256_000))
            ans=response.get('answers',{}).get('vet_triage',{})
            if ans.get('type')!='choice' or set(ans.get('probabilities',{}))!=set(KEYS):
                raise ValueError('unexpected Jev schema (case withheld)')
            probs=ans['probabilities']
            if not all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in probs.values()):
                raise ValueError('invalid probability')
            if abs(sum(probs.values())-1)>.005:raise ValueError('bad probability sum')
            return [probs[k]/sum(probs.values()) for k in KEYS],(time.perf_counter()-start)*1000,response.get('usage',{}),response.get('model')
        except urllib.error.HTTPError as e:
            if e.code not in (429,529) or retry==2:raise RuntimeError('Jev API HTTP '+str(e.code)+' (no clinical text logged)') from None
            time.sleep(2**retry)
        except urllib.error.URLError:
            if retry==2:raise RuntimeError('Jev transport failure (clinical text suppressed)') from None
            time.sleep(2**retry)

def run(token,max_cases,model):
    if not SEALED.exists():
        print('SEALED_SOURCE_NOT_PRESENT: key is available; no billable inference attempted')
        return
    payload=json.loads(decrypt(SEALED.read_text(),token))
    if payload.get('source_sha256')!='4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027':raise ValueError('source fingerprint mismatch')
    cases=payload.get('cases',[])
    if payload.get('split')!='test' or not isinstance(cases,list) or len(cases)<max_cases:
        raise ValueError('bad frozen test subset')
    if len({x['id'] for x in cases})!=len(cases):raise ValueError('duplicate cases')
    if any(not x.get('state') or len(x.get('vet_votes',[]))!=3 or not all(v in CATS for v in x['vet_votes']) for x in cases):
        raise ValueError('invalid clinical state or vet votes')
    results=[];lat=[];tokens=0;served=set()
    print('WONG_JEV_START n='+str(max_cases)+' original_sha_verified=true',flush=True)
    for i,case in enumerate(cases[:max_cases],1):
        probabilities,ms,usage,model_id=call(case,token,model)
        q=[sum(v==cat for v in case['vet_votes'])/3 for cat in CATS]
        pred=max(range(5),key=lambda j:probabilities[j])
        expected=sorted([CATS.index(v) for v in case['vet_votes']])[1]
        r={
            'brier':sum((probabilities[j]-q[j])**2 for j in range(5)),
            'median_exact':int(pred==expected),
            'vote_undertriage':sum(pred>CATS.index(v) for v in case['vet_votes'])/3,
            'vote_overtriage':sum(pred<CATS.index(v) for v in case['vet_votes'])/3,
            'vote_ordinal_mae':sum(abs(pred-CATS.index(v)) for v in case['vet_votes'])/3,
            'log_loss':-sum(q[j]*math.log(max(probabilities[j],1e-12)) for j in range(5)),
            'entropy':-sum(x*math.log(max(x,1e-12)) for x in probabilities)
        }
        results.append(r);lat.append(ms);tokens+=int(usage.get('input_tokens',0));served.add(str(model_id))
        print('WONG_JEV_PROGRESS '+str(i)+'/'+str(max_cases),flush=True)
    agg={k:sum(x[k] for x in results)/len(results) for k in results[0]}
    summary={'n_cases':len(results),'model_requested':model,'models_served':sorted(served),
        'dataset':'Wong original 340 canine cases, frozen case-exclusive heldout split',
        'source_sha256':payload['source_sha256'],
        'protocol':'five-way Choice against 3 original independent vet ratings; not a clinical safety claim',
        'means':agg,'median_latency_ms':statistics.median(lat),
        'input_tokens':tokens}
    out=Path('reports/index_l3_vet/wong_jev_aggregate.json')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2)+'\n')
    print('WONG_JEV_AGGREGATE '+json.dumps(summary,sort_keys=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['pubkey','run'],required=True)
    parser.add_argument('--n',type=int,default=5)
    parser.add_argument('--model',default='jev-1.13.0')
    args=parser.parse_args()
    key=os.environ.get('TYPESAFE_API_KEY')
    if not key:raise SystemExit('TYPESAFE_API_KEY missing from GitHub Actions secrets; no calls made')
    if args.mode=='pubkey':
        pk=private_key(key).public_key().public_bytes(encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
        print('TRANSPORT_PUBKEY='+base64.b64encode(pk).decode())
        print('TYPESAFE_SECRET_PRESENT=true (value never printed)')
    else:
        if args.n<1 or args.n>102:raise ValueError('n outside 1..102')
        run(key,args.n,args.model)
