"""Perceptual cross-split near-duplicate screening for Sheep APR.

Audits original image bytes via dHash + grayscale Pearson similarity. A low
dHash Hamming distance is a CANDIDATE, not proof of same photograph; review
examples with actual images. Doesn't modify datasets.
"""
from __future__ import annotations
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from datasets import load_dataset,Image as HFImage
from huggingface_hub import HfApi

DATASET="oliveirabruno01/sheep-facial-expression-benchmark"

def getrows(split,rev):
    ds=load_dataset(DATASET,split=split,revision=rev,streaming=True)
    ds=ds.cast_column('image',HFImage(decode=False))
    result=[]
    for r in ds:
        im=r['image']
        if not isinstance(im,dict) or not isinstance(im.get('bytes'),bytes):
            raise RuntimeError(f'{split}: missing image bytes')
        picture=Image.open(BytesIO(im['bytes'])).convert('RGB')
        gray=np.array(picture.convert('L').resize((9,8)))
        dhash=sum(int(z)<<i for i,z in enumerate((gray[:,1:]>gray[:,:-1]).flatten()))
        preview=np.array(picture.convert('L').resize((96,96)),dtype=np.float32)
        result.append({
            'hash64':dhash, 'label':str(r['label']),
            'filename':str(r.get('source_filename','')),
            'original_dhash':str(r.get('dhash','')),
            'image_bytes_sha256':__import__('hashlib').sha256(im['bytes']).hexdigest(),
            'gray96':preview
        })
    return result

def pearson(a,b):
    x=a.ravel();y=b.ravel()
    nx=np.linalg.norm(x-x.mean());ny=np.linalg.norm(y-y.mean())
    if nx*ny<1e-9:return 0.0
    return float(np.dot(x-x.mean(),y-y.mean())/(nx*ny))

def summary(a,b,max_dist=4):
    pairs=[]
    for x in a:
        for y in b:
            d=(x['hash64']^y['hash64']).bit_count()
            if d<=max_dist:
                pairs.append({
                    'dist':d,
                    'same_label':x['label']==y['label'],
                    'label_a':x['label'],'label_b':y['label'],
                    'grayscale_pearson':round(pearson(x['gray96'],y['gray96']),3),
                    'same_original_filename':x['filename']==y['filename'],
                    'filename_a':x['filename'],'filename_b':y['filename'],
                    'same_embedded_bytes':x['image_bytes_sha256']==y['image_bytes_sha256'],
                    'sha_a':x['image_bytes_sha256'][:12], 'sha_b':y['image_bytes_sha256'][:12]
                })
    pairs.sort(key=lambda p:(p['dist'],-p['grayscale_pearson']))
    return {'candidate_pairs_dhash_at_most_4':len(pairs),
            'opposite_labels':sum(not p['same_label'] for p in pairs),
            'high_corr_at_least_0_8':sum(p['grayscale_pearson']>=0.8 for p in pairs),
            'top_examples':pairs[:12]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rev=HfApi().dataset_info(DATASET).sha
    records={k:getrows(k,rev) for k in ('train','test','train_raw','test_raw')}
    result={'dataset':DATASET,'revision':rev,
            'pairings':{f'{a}_vs_{b}':summary(records[a],records[b])
                        for a,b in (('train','test'),('train_raw','test_raw'),
                                    ('train','test_raw'),('train_raw','test'))}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print('SHEEP IMAGE PAIR AUDIT: '+json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
