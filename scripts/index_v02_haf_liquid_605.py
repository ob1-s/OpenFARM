#!/usr/bin/env python3
"""Genuine Liquid d1 multimodal Q8 on all 605 original expert-labeled horse faces.

A retrospective L2 source benchmark, not pain physiology/diagnosis or welfare
action. Animal identities are NOT inferred from 22 video filenames; group by
source video until independent author mapping is fully validated.
"""
import base64,collections,hashlib,io,json,math,re,statistics,tempfile,urllib.request,zipfile
from pathlib import Path
from PIL import Image
SOURCE="https://zenodo.org/records/14541977/files/Pain%20and%20NoPain.zip?download=1"
EXPECTED_MD5="be09a4ac6c1d3db3c4908cad1aeca075"
PAT=re.compile(r"cropped_(pain|nopain)_video(\d+)_1_(\d+)\.png",re.I)
def original():
    req=urllib.request.Request(SOURCE,headers={"User-Agent":"OpenFARM-HAF2024-original-images-research/0.2"})
    return urllib.request.urlopen(req,timeout=100)
def payload(blob):
    with Image.open(io.BytesIO(blob)) as img:
        im=img.convert("RGB");im.thumbnail((448,448))
        b=io.BytesIO();im.save(b,format="JPEG",quality=82)
    return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode("ascii")
def evaluate(image):
    import urllib.error
    ask={"state":"An original close-cropped RGB facial photograph of a horse recorded in a veterinary pain-face study.",
         "images":[image],
         "questions":{"visible_pain_face":{
           "type":"noul",
           "instructions":"According to facial appearance alone, does this horse display an equine pain face? Judge ears, orbital narrowing, tension around eyes/muzzle and nostrils when visible. Do not assume medical treatment, clinical outcome, or background context."}}}
    req=urllib.request.Request("http://127.0.0.1:8080/v1/systemone",data=json.dumps(ask).encode("utf-8"),
        method="POST",headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=110) as f:answer=json.loads(f.read(250000))
    ans=answer.get("answers",{}).get("visible_pain_face",{})
    p=ans.get("noul")
    if ans.get("type")!="noul" or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:
        raise ValueError("Expected native Liquid binary probability")
    return float(p)
def main():
    with tempfile.TemporaryDirectory(prefix="openfarm_605_horse_") as temp:
        path=Path(temp)/"horse-author-original.zip";hash=hashlib.md5();nbytes=0
        with original() as inp,open(path,"wb") as out:
            while block:=inp.read(4*1024*1024):
                out.write(block);hash.update(block);nbytes+=len(block)
        if hash.hexdigest()!=EXPECTED_MD5:raise ValueError("Source media MD5 mismatch")
        print("HAF_L2_HORSE_SOURCE_READY","bytes",nbytes,flush=True)
        records=[];failures=collections.Counter()
        with zipfile.ZipFile(path) as z:
            media=[n for n in z.namelist() if n.lower().endswith(".png")]
            if len(media)!=605:raise ValueError("605 source images expected")
            labels=collections.Counter();videos=collections.defaultdict(list)
            for name in media:
                m=PAT.fullmatch(Path(name).name)
                if not m:raise ValueError("Unrecognized original image filename")
                y,video,frame=m.groups()
                videos[int(video)].append((name,int(y.lower()=="pain"),int(frame)))
                labels[y.lower()]+=1
            if len(videos)!=22 or labels!=collections.Counter({"pain":314,"nopain":291}):
                raise ValueError("Source video/label counts inconsistent")
            for vi,items in sorted(videos.items()):
                for name,label,frame in sorted(items,key=lambda x:x[2]):
                    try:
                        p=evaluate(payload(z.read(name)))
                        records.append({"video_id":vi,"reference":label,"p_pain_face":p,
                          "argmax":int(p>=.5),"brier":(p-label)**2})
                    except Exception as e:
                        failures[type(e).__name__]+=1
                    if len(records)%40==0 and len(records)>0:print("HAF_L2_HORSE_PROGRESS",len(records),"/605",flush=True)
        by_video=collections.defaultdict(list)
        for r in records:by_video[r["video_id"]].append(r)
        per_video={}
        for vid,rows in by_video.items():
            per_video[str(vid)]={"n":len(rows),"prevalence_pain":statistics.mean(x["reference"] for x in rows),
                "brier":statistics.mean(x["brier"] for x in rows),
                "accuracy":statistics.mean(x["argmax"]==x["reference"] for x in rows),
                "mean_predicted_pain":statistics.mean(x["p_pain_face"] for x in rows)}
        n=len(records);prevalence=314/605
        summary={
            "source":"10.5281/zenodo.14541977","original_archive_md5":EXPECTED_MD5,
            "model":"LiquidAI/d1-omni-600M-Q8-GGUF + official multimodal projector; llama.cpp CPU",
            "task":"visible expert-rated equine pain-face Noul; no clinical outcome",
            "n_source_images":605,"source_videos":22,"reported_horses":11,
            "animal_mapping_status":"unverified; 22 video units are not 22 independent horses",
            "source_label_counts":dict(labels),"n_valid":n,"failures":dict(failures),
            "results":{
                "accuracy_at_0_5":statistics.mean(x["argmax"]==x["reference"] for x in records) if records else None,
                "brier":statistics.mean(x["brier"] for x in records) if records else None,
                "predict_positive_count":sum(x["argmax"] for x in records),
                "mean_probability_pain":statistics.mean(x["p_pain_face"] for x in records) if records else None,
                "class_prior_brier":prevalence*(1-prevalence),
                "all_nopain_accuracy":291/605,
                "all_pain_accuracy":314/605,
                "equal_half_brier":.25},
            "per_video_aggregate":per_video,
            "limitations":"605 frames correlated over 22 source videos/11 horses. Same source label per image, no observed treatment benefit or objective pain physiology. Zero-shot only. No images in GH artifacts."
        }
        output=Path("reports/decision-index/v02_haf605_liquid_original_images.json")
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(summary,indent=2)+"\n")
        print("HAF_L2_LIQUID_605_RESULT",json.dumps({k:summary[k] for k in ("n_valid","failures","results")},sort_keys=True),flush=True)
        if n!=605:raise RuntimeError("Not all original source images successfully scored")
if __name__=="__main__":main()
