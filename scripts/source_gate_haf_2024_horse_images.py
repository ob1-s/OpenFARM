#!/usr/bin/env python3
"""Original HAF 2024 equine pain-face images source gate, distinct from clinical outcomes.

605 expert-rated frame images from 22 videos of reportedly 11 horses. Video
IDs in filenames are NOT animal IDs. No person/media data published. Verify
author archive md5, label filename construction and image decode; source
overlap with prior OpenFARM horse seeds is a separate admission gate.
"""
import collections,hashlib,io,json,re,tempfile,urllib.request,zipfile
from pathlib import Path
from PIL import Image

REC="https://zenodo.org/api/records/14541977"
MD5="be09a4ac6c1d3db3c4908cad1aeca075"
SOURCE="https://zenodo.org/records/14541977/files/Pain%20and%20NoPain.zip?download=1"
PAT=re.compile(r"cropped_(pain|nopain)_video(\d+)_1_(\d+)\.png",re.I)
def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'OpenFARM-Horse-L2-SourceGate/0.2'}),timeout=90)
def main():
    out={'status':'UNVERIFIED','study_doi':'10.5281/zenodo.14541977',
      'original_archive':SOURCE,'expected_md5':MD5,
      'claimed_animals_in_paper':11,'claimed_videos':22,
      'source_overlap_with_existing_openfarm_horse_seeds':'NOT_YET_CHECKED'}
    with request(REC) as reply:meta=json.loads(reply.read(1_000_000))
    files=meta.get('files',[])
    rights=meta.get('metadata',{}).get('rights') or meta.get('metadata',{}).get('license')
    out['repository_rights_metadata']=rights
    out['published_file_names']=[x.get('key') or x.get('filename') for x in files]
    if not any("Pain and NoPain.zip" in str(x) for x in out['published_file_names']):
        raise ValueError('source API did not contain original named media file')
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'original-equine-images.zip';digest=hashlib.md5();sha=hashlib.sha256();nbytes=0
        with request(SOURCE) as response,open(path,'wb') as dest:
            while b:=response.read(4*1024*1024):
                dest.write(b);digest.update(b);sha.update(b);nbytes+=len(b)
                if nbytes>450_000_000:raise ValueError('unexpected large archive')
        out.update(archive_sha256=sha.hexdigest(),archive_bytes=nbytes,archive_md5=digest.hexdigest())
        if digest.hexdigest()!=MD5:raise ValueError("published MD5 content mismatch")
        with zipfile.ZipFile(path) as z:
            names=[x for x in z.namelist() if x.lower().endswith('.png')]
            if len(names)!=605:raise ValueError('expected original 605 frames')
            by_video=collections.defaultdict(list)
            labels=collections.Counter()
            for name in names:
                m=PAT.fullmatch(Path(name).name)
                if not m:raise ValueError('file naming does not match source paper')
                label,video,frame=m.groups()
                by_video[int(video)].append((name,label,int(frame)))
                labels[label.lower()]+=1
            if len(by_video)!=22:raise ValueError('source video counts inconsistent')
            samples=[]
            for video,items in sorted(by_video.items()):
                item=sorted(items,key=lambda x:x[2])[0]
                b=z.read(item[0])
                with Image.open(io.BytesIO(b)) as img:
                    w,h=img.size
                    img.verify()
                samples.append({'video_id':video,'width':w,'height':h,'source_image_sha256':hashlib.sha256(b).hexdigest()})
        out.update(original_png_frames=len(names),distinct_original_video_ids=len(by_video),
            label_counts=dict(labels),video_frame_counts_sorted=sorted(len(v) for v in by_video.values()),
            sample_decodes=samples)
    out['status']='ORIGINAL_EXPERT_FRAME_MEDIA_VERIFIED_ANIMAL_MAP_AND_OVERLAP_PENDING'
    out['clinical_limit']='expert eye judgment about pain face from sparse video frames, not independent analgesic outcome or clinical care advice'
    p=Path('reports/source_gates/haf2024_horse_original_expert_images.json')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
    print('HAF_HORSE_L2_SOURCE_GATE',json.dumps({k:out.get(k) for k in ('status','archive_md5','original_png_frames','distinct_original_video_ids','label_counts','repository_rights_metadata')},sort_keys=True),flush=True)
if __name__=='__main__':main()
