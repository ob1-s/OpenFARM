"""Inspect real image/audio media and generate small labeled research contact sheets.

No model inference. No dataset mutations. Public Hugging Face media only; review
copyright before redistributing. Outputs remain CI artifacts, not committed files.

Usage: python scripts/audit_openfarm_media.py --output-dir audit-media
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import soundfile as sf

HORSE="oliveirabruno01/openfarm-horse-grimace-region"
SHEEP="oliveirabruno01/sheep-facial-expression-benchmark"
UNGULATE="oliveirabruno01/openfarm-ungulate-valence"

def open_rows(dataset,split,max_items=None):
    from datasets import load_dataset,Image as HFImage, Audio
    from huggingface_hub import HfApi
    rev=HfApi().dataset_info(dataset).sha
    ds=load_dataset(dataset,split=split,revision=rev,streaming=True)
    for col,feature in ds.features.items():
        if isinstance(feature,HFImage):
            ds=ds.cast_column(col,HFImage(decode=False))
        elif isinstance(feature,Audio):
            ds=ds.cast_column(col,Audio(decode=False))
    out=[]
    for i,r in enumerate(ds):
        if max_items is not None and i>=max_items:break
        out.append(r)
    return out,rev

def read_image(row):
    item=row.get('image')
    if not isinstance(item,dict) or not isinstance(item.get('bytes'),bytes):
        raise ValueError("Image bytes unavailable in streaming row")
    return Image.open(BytesIO(item['bytes'])).convert('RGB')

def image_dhash(im):
    im=im.convert('L').resize((9,8))
    a=np.asarray(im)
    bits=(a[:,1:]>a[:,:-1]).flatten()
    return sum(int(bit)<<idx for idx,bit in enumerate(bits))

def collage(items,out_path,title):
    """items = [(image, lines)]"""
    W,H=240,200
    cols=4
    rows=(len(items)+cols-1)//cols
    out=Image.new('RGB',(W*cols,H*rows+30),'#fafafa')
    dr=ImageDraw.Draw(out)
    dr.text((8,8),title,fill='#151515')
    for i,(image,lines) in enumerate(items):
        x=(i%cols)*W;y=(i//cols)*H+30
        thumb=ImageOps.contain(image,(224,154))
        out.paste(thumb,(x+(W-thumb.width)//2,y))
        for k,t in enumerate(lines[:3]):
            dr.text((x+6,y+158+k*13),str(t)[:38],fill='#202020')
    out.save(out_path)

def audit_horse(out):
    rows,revision=open_rows(HORSE,'test',max_items=279)
    selected=[]
    counts=Counter()
    for row in rows:
        cls=str(row.get('pain_score'))
        if counts[cls]>=4:continue
        try:im=read_image(row)
        except Exception:continue
        selected.append((im,[f"score={cls}, region={row.get('face_region')}",
                             f"horse={row.get('horse_id')}",
                             f"source={str(row.get('source_sha256'))[:10]}"]))
        counts[cls]+=1
    collage(selected,out/'horse_sample_grid.png','Horse grimace regional test examples')
    # Investigate repeated source_sha256: parent image vs repeated crop.
    raw,_=open_rows(HORSE,'train_raw',max_items=160)
    by_parent=defaultdict(list)
    for r in raw:
        key=r.get('source_sha256')
        if key and len(by_parent[key])<3:
            try:
                im=read_image(r)
                by_parent[key].append((im,
                    [f"score={r.get('pain_score')}; region={r.get('face_region')}",
                     f"horse={r.get('horse_id')}",
                     f"crop-bytes={sha256(r['image']['bytes']).hexdigest()[:10]}"]))
            except Exception:pass
    pairs=[]
    for values in by_parent.values():
        if len(values)>=2:
            pairs.extend(values[:2])
        if len(pairs)>=12:break
    if pairs:collage(pairs[:12],out/'horse_shared_source_grids.png',
                     'Same source_sha256 across multiple crop rows')
    return {'revision':revision,'test_rows_seen':len(rows),'test_samples_per_score':dict(counts),
            'duplicate_parent_samples_shown':len(pairs)//2}

def audit_sheep(out):
    train,revision=open_rows(SHEEP,'train',max_items=172)
    test,_=open_rows(SHEEP,'test',max_items=74)
    shown=[];counts=Counter();loaded={'train':[],'test':[]}
    for split,rows in [('train',train),('test',test)]:
        for r in rows:
            try:im=read_image(r)
            except Exception:continue
            label=str(r.get('label'))
            hh=image_dhash(im)
            loaded[split].append((hh,im,label,r))
            if split=='test' and counts[label]<6:
                shown.append((im,[f"proxy_label={label}, split=test",f"recomputed_dhash={hh:016x}"]))
                counts[label]+=1
    collage(shown,out/'sheep_sample_grid.png','Sheep facial-expression proxy labels')
    neighbors=[]
    for ht,it,lt,rt in loaded['test']:
        nearest=min(((int((ht^hr).bit_count()),ir,lr,rr) for hr,ir,lr,rr in loaded['train']),
                    key=lambda x:x[0],default=None)
        if nearest:
            dist,ir,lr,rr=nearest
            neighbors.append((dist,it,ir,lt,lr))
    neighbors.sort(key=lambda x:x[0])
    pair_cells=[]
    for dist,it,ir,lt,lr in neighbors[:6]:
        pair_cells.extend([
            (it,[f"TEST label={lt}",f"dHash Hamming={dist}"]),
            (ir,[f"TRAIN label={lr}",f"dHash Hamming={dist}"])
        ])
    if pair_cells:collage(pair_cells,out/'sheep_nearest_train_test.png',
                          'Nearest train/test image pairs by 64-bit dHash')
    return {'revision':revision,'train_decoded':len(loaded['train']),
            'test_decoded':len(loaded['test']),
            'test_samples_per_label':dict(counts),
            'closest_cross_split_dhash_distances':[n[0] for n in neighbors[:15]],
            'number_test_with_nearest_dist_at_most_4':sum(n[0]<=4 for n in neighbors),
            'warning':'Perceptual dHash similarity is a screening signal, not proof of duplicated original photographs.'}

def spectrogram(wav,sr):
    wav=np.asarray(wav,dtype=np.float32)
    if wav.ndim>1:wav=wav.mean(axis=1)
    if wav.size<1024:wav=np.pad(wav,(0,1024-wav.size))
    frame=1024;hop=256
    x=np.lib.stride_tricks.sliding_window_view(wav,frame)[::hop][:240]
    spec=np.log1p(np.abs(np.fft.rfft(x*np.hanning(frame),axis=1)).T)
    lo,hi=np.percentile(spec,[5,99])
    if hi<=lo:hi=lo+1e-6
    im=np.uint8(np.flipud(np.clip((spec-lo)/(hi-lo),0,1))*255)
    return Image.fromarray(im,mode='L').convert('RGB').resize((224,150))

def audit_ungulate(out):
    # Small real audio sample from the already-known contaminated evaluation
    rows,revision=open_rows(UNGULATE,'test',max_items=136)
    counts=Counter();selected=[];sample_rates=Counter()
    for r in rows:
        label=r.get('valence')
        if counts[label]>=6:continue
        audio=r.get('audio')
        if not isinstance(audio,dict) or not isinstance(audio.get('bytes'),bytes):
            continue
        try:
            wav,sr=sf.read(BytesIO(audio['bytes']))
            im=spectrogram(wav,sr)
        except Exception:continue
        sample_rates[sr]+=1
        counts[label]+=1
        selected.append((im,[f"label={label}, sr={sr}",
             f"animal_group={r.get('source_group_id')}",
             f"audio={str(r.get('audio_sha256'))[:12]}"]))
    collage(selected,out/'ungulate_spectrogram_samples.png',
            'Ungulate actual audio as spectrograms (test)')
    return {'revision':revision,'decoded':len(selected),'samples_per_label':dict(counts),
            'sample_rates':dict(sample_rates),
            'warning':'Samples have compromised train/test byte-overlap; gallery is not an independent benchmark.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results={}
    for name,fn in [('horse',audit_horse),('sheep',audit_sheep),('ungulate',audit_ungulate)]:
        try:results[name]=fn(args.output_dir)
        except Exception as e:
            results[name]={'error':f'{type(e).__name__}: {e}'}
    (args.output_dir/'report.json').write_text(json.dumps(results,indent=2))
    print('MEDIA AUDIT SUMMARY: '+json.dumps(results))
    if any('error' in value for value in results.values()):
        raise SystemExit(1)
if __name__=='__main__':main()
