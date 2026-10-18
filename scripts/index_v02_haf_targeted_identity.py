#!/usr/bin/env python3
"""Targeted original-image identity forensic for the three HAF failed videos.

Previous MD5-pinned 605 -> author-2025 release visual-link source audit matched
19/22 videos; exactly 1,4,9 were unmatched under its sparse candidates.
Here recover from possible off-by-one frame indexing and temporal shifts,
never by relaxing dHash threshold or trusting shared numeric video IDs alone.
"""
import collections,hashlib,io,json,tempfile,zipfile
from pathlib import Path
from PIL import Image
from remotezip import RemoteZip
from index_v02_haf_crossrelease_frame_verify import request,OLD,NEW,REC,ORIG,ORIGMD5,profile,dist

TARGET={1,4,9}
OFF=(-25,-24,-5,-2,-1,0,1,2,5,24,25)
def main():
    with request(REC) as r:meta=json.loads(r.read(500000))
    urls=[(x["key"],x["links"]["self"]) for x in meta["files"] if "med id" in x["key"].lower()]
    if len(urls)!=2:raise ValueError("Expected two original 25fps source archives")
    out={"source_status":"FORENSIC_NOT_YET_VERIFIED","source_605_md5":ORIGMD5,
         "earlier_missing_video_ids":sorted(TARGET),"decision_threshold":"byte-exact OR <=8/64 dHash; no near-match accepted merely for ID","videos":{}}
    with tempfile.TemporaryDirectory(prefix="openfarm_haf_targeted_") as t:
        zipname=Path(t)/"old.zip";h=hashlib.md5()
        with request(ORIG) as s,open(zipname,"wb") as f:
            while b:=s.read(4*1024*1024):
                f.write(b);h.update(b)
        if h.hexdigest()!=ORIGMD5:raise ValueError("Original MD5 changed")
        with zipfile.ZipFile(zipname) as old:
            examples=collections.defaultdict(list)
            for fn in old.namelist():
                m=OLD.fullmatch(Path(fn).name)
                if m and int(m.group(2)) in TARGET:
                    label,vid,frame=m.groups()
                    examples[int(vid)].append((int(frame),label,fn))
            if set(examples)!=TARGET:raise ValueError("Missing targeted source videos")
            attempts=collections.defaultdict(list)
            for archive,url in urls:
                with RemoteZip(url,initial_buffer_size=262144) as high:
                    keys=collections.defaultdict(dict)
                    for fn in high.namelist():
                        m=NEW.fullmatch(Path(fn).name)
                        if m and int(m.group(2)) in TARGET:
                            label,vid,frame,horse=m.groups()
                            keys[(int(vid),label)][int(frame)]=(fn,horse)
                    for vid in sorted(TARGET):
                        rows=sorted(examples[vid])
                        select={0,len(rows)//4,len(rows)//2,(3*len(rows))//4,len(rows)-1}
                        for rindex in sorted(select):
                            frame,label,fn=rows[rindex]
                            available=keys.get((vid,label),{})
                            if not available:continue
                            pivots=[frame,frame*25,1+(frame-1)*25]
                            candidates=sorted({p+off for p in pivots for off in OFF if p+off in available})
                            src=old.read(fn);ph=profile(src)
                            best=None
                            for cand in candidates:
                                dst,hid=available[cand]
                                im=high.read(dst)
                                exact=hashlib.sha256(src).digest()==hashlib.sha256(im).digest()
                                dh=dist(ph,profile(im))
                                item={"source_old_frame":frame,"matched_new_frame":cand,
                                     "horse_id":hid,"dhash":dh,"byte_exact":exact,"archive":archive}
                                if best is None or (not exact,dh)<(not best["byte_exact"],best["dhash"]):
                                    best=item
                                if exact:break
                            if best is not None: attempts[vid].append(best)
    for vid in sorted(TARGET):
        a=attempts.get(vid,[])
        passed=[x for x in a if x["byte_exact"] or x["dhash"]<=8]
        horse_ids=sorted({x["horse_id"] for x in passed})
        out["videos"][str(vid)]={
            "old_source_frames":len(examples[vid]),"attempted_samples":len(a),
            "matches_threshold":len(passed),
            "byte_exact":sum(x["byte_exact"] for x in passed),
            "min_dhash":min((x["dhash"] for x in a),default=None),
            "matched_horse_ids":horse_ids,
            "best_samples":sorted(a,key=lambda x:(not x["byte_exact"],x["dhash"]))[:4],
            "verified":bool(passed) and len(horse_ids)==1
        }
    ok=all(v["verified"] for v in out["videos"].values())
    out["source_status"]="TARGETED_3_OF_3_LINKED" if ok else "TARGETED_SOURCE_ID_LINK_PARTIAL"
    output=Path("reports/source_gates/haf2024_three_video_targeted_match.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("HAF_TARGETED_IDENTITY_SUMMARY",json.dumps(out,sort_keys=True),flush=True)
    if not ok:raise RuntimeError("3 source video pixel links still not all verified")
if __name__=="__main__":main()
