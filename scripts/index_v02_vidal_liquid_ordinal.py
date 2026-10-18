#!/usr/bin/env python3
"""External 2022 original visual L2 score evaluation, independent of GrimACE.

Original 8-mouse orbital tightening annotations and author-hosted 470MB PNG
archive. Actual Liquid d1 multimodal ggml decision-head CPU inference.
No image bytes, embeddings, models or sample screenshots are distributed.
96 label-stratified sample images across all 8 mice (not natural prevalence).
"""
import base64,collections,hashlib,io,json,math,statistics,tempfile,urllib.request,urllib.error,zipfile
from pathlib import Path
import PIL.Image

ANNOTATION="https://raw.githubusercontent.com/AndreaVidal/WhiteFurredMice_Dataset/main/Annotation_json/eyes_detection_data.json"
ARCHIVE="https://lab-msp.com/WhiteFurredMiceDataset/Eyes_detection.zip"
ANNSHA="754d3442b056b14aee137643adab101c166dee27f42338397ef7a1f869cb9ad2"
ZIP_SHA="57c8a2dc112232544b25726a9d4a594795c44c2c528956128d9a2ad647c4064e"
CLASSES=["0","1","2"]
def source(url):
    req=urllib.request.Request(url,headers={'User-Agent':'OpenFARM-External-Research-Only/0.2'})
    return urllib.request.urlopen(req,timeout=100)
def load():
    with source(ANNOTATION) as r:src=r.read(5_000_000)
    if hashlib.sha256(src).hexdigest()!=ANNSHA:raise ValueError("external annotation bytes not pinned")
    ann=json.loads(src)
    if len(ann)!=1087:raise ValueError('expected 1087 original annotations')
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for name,entries in ann.items():
        if not name.endswith(".png") or len(entries)!=1:raise ValueError("ambiguous annotations")
        one=entries[0]
        animal=str(one["AnimalNumber"]);label=str(one["GrimaceScale"])
        if label not in CLASSES:raise ValueError('invalid source score')
        groups[animal][label].append(name)
    if len(groups)!=8:raise ValueError("expected original eight subjects")
    return ann,groups
def pick(groups):
    selected=[]
    for subject,classes in sorted(groups.items()):
        # 12 images per mouse; preferential balance classes without oversampling.
        taken=set()
        for y in CLASSES:
            keys=sorted(classes[y],key=lambda name:hashlib.sha256(('openfarm-vidal-v02:'+name).encode()).hexdigest())
            chosen=keys[:4]
            taken.update(chosen)
        rest=sorted([name for keys in classes.values() for name in keys if name not in taken],
            key=lambda name:hashlib.sha256(('openfarm-vidal-v02-fallback:'+name).encode()).hexdigest())
        taken.update(rest[:max(0,12-len(taken))])
        selected.extend((name,subject,str(y)) for y,keys in classes.items() for name in sorted(taken) if name in keys)
    if len(selected)!=96:raise ValueError("cannot sample exactly 12 images per 8 animals")
    return sorted(selected,key=lambda v:(v[1],hashlib.sha256(v[0].encode()).hexdigest()))
def image_payload(raw):
    with PIL.Image.open(io.BytesIO(raw)) as img:
        rgb=img.convert('RGB')
        rgb.thumbnail((448,448))
        out=io.BytesIO()
        rgb.save(out,format='JPEG',quality=83)
    return 'data:image/jpeg;base64,'+base64.b64encode(out.getvalue()).decode('ascii')
def predict(image):
    question={"state":"An original eye-region image of a laboratory mouse in a standardized experiment.",
      "images":[image],"questions":{"orbital_tightening":{
        "type":"score",
        "instructions":"Score the visible Mouse Grimace Scale orbital tightening ONLY. The ordered scale is: 0 no orbital tightening, eyes wide/open; 1 moderately narrowed eyes; 2 clearly narrowed or closed eyes. Do not infer the surgery or intervention.",
        "criteria":["0 = absent orbital tightening, eyes open","1 = moderate narrowing of eyes","2 = obvious orbital tightening, strongly narrowed eyes"]}}}
    req=urllib.request.Request('http://127.0.0.1:8080/v1/systemone',
       method='POST',data=json.dumps(question).encode(),
       headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as r:res=json.loads(r.read(160000))
    answer=res.get('answers',{}).get('orbital_tightening',{})
    p=answer.get('probabilities')
    p=[p.get(str(i)) for i in range(3)] if isinstance(p,dict) else p
    if answer.get('type')!='score' or not isinstance(p,list) or len(p)!=3 or any(
        not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in p):
        raise ValueError("Invalid native ordinal Score response")
    total=sum(p)
    if not .975<=total<=1.025:raise ValueError("Original model probabilities do not sum to one")
    return [x/total for x in p]
def main():
    ann,groups=load()
    examples=pick(groups)
    d={'status':'STARTED','original_source_images':1087,'n_source_animals':8,
      'evaluation_sample':'96 label-stratified original eye frames, 12/mouse (enriched)',
      'source_annotation_sha256':ANNSHA,'source_archive_sha256':ZIP_SHA,
      'model':'LiquidAI/d1-omni-600M-GGUF Q8 + exact multimodal projector on GitHub Actions CPU',
      'primitive':'Score[0,1,2]','media_rights':'unknown; non-redistributed research pilot, no commercial training',
      'per_mouse':{},'invalid_count':0}
    rows=[];errors=collections.Counter()
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'author-original-eyes.zip'
        h=hashlib.sha256()
        with source(ARCHIVE) as response,open(path,'wb') as out:
            while buf:=response.read(4*1024*1024):
                h.update(buf);out.write(buf)
        if h.hexdigest()!=ZIP_SHA:raise ValueError("Source archive bytes changed")
        with zipfile.ZipFile(path) as archive:
            entries={Path(name).name:name for name in archive.namelist() if name.endswith('.png')}
            if set(entries)!=set(ann):raise ValueError("Original annotation image keys no longer align")
            for i,(name,subject,y) in enumerate(examples,1):
                try:
                    p=predict(image_payload(archive.read(entries[name])))
                    idx=CLASSES.index(y);hat=max(range(3),key=lambda n:p[n])
                    rows.append({'mouse':subject,'gold':idx,'pred':hat,'brier':sum((p[j]-int(j==idx))**2 for j in range(3)),
                                 'ordinal_mae':abs(hat-idx),'correct':int(idx==hat)})
                except Exception as e:
                    errors[type(e).__name__]+=1
                if i%12==0:print('VIDAL_L2_VISION_PROGRESS',i,'/96',flush=True)
    mouse=collections.defaultdict(list)
    for x in rows:mouse[x['mouse']].append(x)
    for key,values in mouse.items():
        d['per_mouse'][key]={'valid':len(values),'exact':statistics.mean(x['correct'] for x in values),
            'brier':statistics.mean(x['brier'] for x in values),'ordinal_mae':statistics.mean(x['ordinal_mae'] for x in values)}
    d['errors']=dict(errors);d['invalid_count']=sum(errors.values())
    d['evaluated']=len(rows)
    if rows:
        d['aggregate']={'exact':statistics.mean(x['correct'] for x in rows),
          'brier':statistics.mean(x['brier'] for x in rows),
          'ordinal_mae':statistics.mean(x['ordinal_mae'] for x in rows),
          'pred_class_counts':dict(collections.Counter(str(x['pred']) for x in rows)),
          'gold_class_counts':dict(collections.Counter(str(x['gold']) for x in rows)),
          'uniform_3way_onehot_brier':2/3}
    d['status']='RESEARCH_EVAL_COMPLETE' if len(rows)==96 else 'INCOMPLETE_RESEARCH_EVAL'
    d['limit']='96 original frames but just 8 independent animals; source label validity and license must be separately reviewed. This test cannot establish animal clinical pain outcomes.'
    dest=Path('reports/decision-index/v02_vidal_actual_visual_model_96.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(d,indent=2)+'\n')
    print('VIDAL_L2_LIQUID_RESULT',json.dumps(d,sort_keys=True),flush=True)
    if len(rows)!=96:raise RuntimeError('96-image L2 evaluation incomplete')
if __name__=='__main__':main()
