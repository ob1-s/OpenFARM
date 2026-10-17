#!/usr/bin/env python3
"""Six source-verified broiler 10s windows to real Liquid d1 multimodal Noul.

Diagnostic only: four original chronological frames represent each 10-second
clip, so this is sparse video-derived vision, not continuous video encoding.
"""
import base64,csv,hashlib,io,json,math,re,statistics,subprocess,tempfile,urllib.request,urllib.error
from pathlib import Path
from remotezip import RemoteZip
SRC="https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
FILES={
"037_1_10_1920_157_0_01":"572aed483640572111579c888bc275e8a4a8d32b9debd387e6ed2a2ddb441231",
"092_2_20_1920_011_1_01":"e50c5c22d40fcc0d1f4efa69f8a0bd09dbfb25fa954b54d778f8bab6a1f5cfd6",
"090_3_32_1920_010_0_10":"b7338a9d4b9cf93f1568769414d80b8c82fe3b8627f86070b5478b8d3e9b44eb"}
def cli(parts):
    p=subprocess.run(parts,capture_output=True,timeout=100)
    if p.returncode:raise RuntimeError('ffmpeg source decode error')
    return p.stdout
def windows(z,ann):
    seen=set();pos=set();bad=set();maxf=0
    reader=csv.DictReader(io.TextIOWrapper(z.open(ann),encoding='utf-8-sig',newline=''))
    for r in reader:
        f=int(r['frame']);maxf=max(maxf,f);seen.add(f)
        labels=re.split('[,;|]',r['actions_ground'] or '')
        if 'Feather_Pecking' in labels:pos.add(f)
        if 'Video_Problem' in labels:bad.add(f)
    opts=[]
    for end in range(249,maxf,25):
        seq=set(range(end-249,end+1))
        if len(seq&seen)<200 or seq&bad:continue
        opts.append((end,int(bool(seq&pos))))
    positives=[f for f,y in opts if y==1]
    negatives=[f for f,y in opts if y==0]
    if not positives or not negatives:raise ValueError('cannot select scorable positive and negative windows')
    positive=positives[0]
    negative=next((f for f in negatives if abs(f-positive)>=250),negatives[0])
    return [(positive,1),(negative,0)],len(opts)
def frame(path,f):
    b=cli(['ffmpeg','-hide_banner','-loglevel','error','-ss',f'{f/25:.4f}',
        '-i',str(path),'-frames:v','1','-vf','scale=320:-2','-q:v','8',
        '-f','image2pipe','-vcodec','mjpeg','-'])
    if not b.startswith(bytes.fromhex('ffd8')):raise ValueError('bad source JPEG')
    return 'data:image/jpeg;base64,'+base64.b64encode(b).decode()
def ask(images):
    def request(one_or_many):
        req={"state":"Four chronological 2.5-second-spaced frames from the same 10-second broiler flock video.",
             "images":one_or_many,"questions":{"feather_pecking":{"type":"noul",
             "instructions":"Does any chicken visibly peck or pull feathers of another chicken in the sampled sequence? Self-preening does not count."}}}
        q=urllib.request.Request('http://127.0.0.1:8080/v1/systemone',
            data=json.dumps(req).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(q,timeout=120) as response:
                return json.loads(response.read(130000))
        except urllib.error.HTTPError as e:
            body=e.read(900).decode(errors='replace')
            print('MULTIFRAME_HTTP_SCHEMA_DIAGNOSTIC',json.dumps({'status':e.code,'body':body[:700]}),flush=True)
            raise
    kind='four_sequential_images'
    try:
        response=request(images)
    except urllib.error.HTTPError as e:
        if e.code!=501:raise
        from PIL import Image
        # Fall back to ONE image containing all four frames in chronological order.
        # This tests image capability but is NOT a true video or multi-image model.
        tiles=[Image.open(io.BytesIO(base64.b64decode(x.split(',',1)[1]))).convert('RGB') for x in images]
        size=tiles[0].size
        canvas=Image.new('RGB',(size[0]*2,size[1]*2))
        for i,im in enumerate(tiles):
            canvas.paste(im,((i%2)*size[0],(i//2)*size[1]))
        buffer=io.BytesIO();canvas.save(buffer,format='JPEG',quality=78)
        single='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()
        response=request([single])
        kind='single_temporal_2x2_montage_fallback'
    ans=response['answers']['feather_pecking']
    p=ans.get('noul')
    if ans.get('type')!='noul' or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:
        raise ValueError('invalid multimodal model response')
    return float(p),kind

def main():
    scores=[];source=[]
    with RemoteZip(SRC,initial_buffer_size=4*1024*1024) as z:
        names=set(z.namelist())
        for stem,sha in FILES.items():
            video=next((n for n in names if n.endswith('/videos/'+stem+'.mp4')),None)
            annot=next((n for n in names if n.endswith('/annotations/'+stem+'.csv')),None)
            if not video or not annot:raise ValueError('missing video/source-label exact pair')
            selected,nscorable=windows(z,annot)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'source.mp4';h=hashlib.sha256()
                with z.open(video) as inp,open(path,'wb') as out:
                    while block:=inp.read(4*1024*1024):h.update(block);out.write(block)
                if h.hexdigest()!=sha:raise ValueError('video SHA different from earlier validation')
                for end,gold in selected:
                    # Fixed without inspecting inside-window annotations.
                    imgs=[frame(path,end-249+i) for i in (0,62,125,187)]
                    p,representation=ask(imgs)
                    scores.append({'setup':int(stem.split('_')[1]),'source_video':stem,
                                   'end_frame':end,'expert_window_label':gold,'p_model':p,'representation_used':representation})
                    print('MULTIFRAME_LIQUID_CASE',stem,'gold',gold,'prob',round(p,4),flush=True)
            source.append({'stem':stem,'source_sha256':sha,'n_scorable_windows':nscorable})
    mean=statistics.mean((r['p_model']-r['expert_window_label'])**2 for r in scores)
    summary={'n':len(scores),'n_positives':sum(x['expert_window_label'] for x in scores),
             'n_negatives':sum(1-x['expert_window_label'] for x in scores),
             'brier':mean,'correct_at_0_5':sum((x['p_model']>=.5)==x['expert_window_label'] for x in scores),
             'source_setups':len(source)}
    out={'source':SRC,'model':'LiquidAI/d1-omni-600M-GGUF:Q8_0',
         'primitive':'Noul','input':'four original chronological frames from each 10s video; no label/box metadata',
         'no_continuous_video_encoding':True,'positive_enriched_by_source_label':True,
         'source_videos':source,'windows':scores,'aggregate':summary,
         'limitation':'6 deliberately selected highly correlated windows, not a valid generalization benchmark; 4 frames may miss behavior between sampled frames.'}
    target=Path('reports/decision-index/broiler_real_multiframe_d1_pilot.json')
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2)+'\n')
    print('MULTIFRAME_LIQUID_SUMMARY',json.dumps(summary),flush=True)
    if len(scores)!=6:raise RuntimeError('pilot incomplete')
if __name__=='__main__':main()
