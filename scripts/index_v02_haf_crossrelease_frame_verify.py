#!/usr/bin/env python3
"""Verify original HAF 605 1fps frames link to the same 2025 25fps horse-ID frames.

Only samples 3 original frames/video and corresponding higher-rate images from
author ZIP with HTTP 206. Original 605 archive fully MD5 pinned. No media exits CI.
"""
import collections,hashlib,io,json,re,tempfile,urllib.request,zipfile
from pathlib import Path
from PIL import Image,ImageOps
from remotezip import RemoteZip

REC="https://zenodo.org/api/records/17389614"
ORIG="https://zenodo.org/records/14541977/files/Pain%20and%20NoPain.zip?download=1"
ORIGMD5="be09a4ac6c1d3db3c4908cad1aeca075"
OLD=re.compile(r"cropped_(pain|nopain)_video(\d+)_1_(\d+)\.png",re.I)
NEW=re.compile(r"cropped_(pain|nopain)_video(\d+)_25_(\d+)_ID=(\d+)\.png",re.I)
def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"OpenFARM-independent-frame-link-verifier/0.2"}),timeout=100)
def profile(data):
    with Image.open(io.BytesIO(data)) as im:
        pic=ImageOps.grayscale(im).resize((9,8),resample=Image.Resampling.LANCZOS)
        raw=list(pic.getdata());return tuple(int(raw[i*9+j]>raw[i*9+j+1]) for i in range(8) for j in range(8))
def dist(a,b):return sum(x!=y for x,y in zip(a,b))
def main():
    with request(REC) as r:meta=json.loads(r.read(500000))
    src_files={x["key"]:x for x in meta["files"]}
    remote=[]
    for name,item in src_files.items():
        if "med ID" in name or "med id" in name.lower():
            url=item["links"]["self"]
            remote.append((name,url))
    if len(remote)!=2:raise ValueError("Original 2025 horse ID source list changed")
    with tempfile.TemporaryDirectory(prefix="openfarm_haf_linkage_") as temp:
        path=Path(temp)/"original605.zip";h=hashlib.md5()
        with request(ORIG) as r,open(path,"wb") as f:
            while b:=r.read(4*1024*1024):f.write(b);h.update(b)
        if h.hexdigest()!=ORIGMD5:raise ValueError("Original 605 source checksum mismatch")
        with zipfile.ZipFile(path) as old:
            oldgroups=collections.defaultdict(list)
            for name in old.namelist():
                m=OLD.fullmatch(Path(name).name)
                if not m:continue
                label,vid,frame=m.groups()
                oldgroups[int(vid)].append((int(frame),label,name))
            if len(oldgroups)!=22:raise ValueError("source old video IDs changed")
            checks=collections.defaultdict(list)
            for archivename,url in remote:
                with RemoteZip(url,initial_buffer_size=262144) as z:
                    high=collections.defaultdict(dict)
                    for pathstr in z.namelist():
                        m=NEW.fullmatch(Path(pathstr).name)
                        if not m:continue
                        label,vid,frame,horse=m.groups()
                        high[(int(vid),label)][int(frame)]=(pathstr,horse)
                    for vid,frames in sorted(oldgroups.items()):
                        rows=sorted(frames)
                        selected=[rows[0],rows[len(rows)//2],rows[-1]]
                        for oldframe,label,oldname in selected:
                            key=(vid,label);available=high.get(key)
                            if not available:continue
                            # Filename indices are given as temporal frame counters at 1 vs 25 fps.
                            candidates=set()
                            for pivot in (oldframe*25,oldframe):
                                for off in (-25,-5,-1,0,1,5,25):
                                    if pivot+off in available:candidates.add(pivot+off)
                            if not candidates:continue
                            original=old.read(oldname);orig_phash=profile(original)
                            best=None
                            for candidate in sorted(candidates):
                                name,horse=available[candidate]
                                new=z.read(name)
                                diff=dist(orig_phash,profile(new))
                                direct=hashlib.sha256(original).digest()==hashlib.sha256(new).digest()
                                item={"old_frame":oldframe,"new_frame":candidate,"video":vid,"horse":horse,
                                      "byte_exact":direct,"dhash_distance":diff,"source":archivename}
                                if best is None or (int(not direct),diff)<(int(not best["byte_exact"]),best["dhash_distance"]):
                                    best=item
                            if best:checks[vid].append(best)
    by_video={}
    for vid,items in sorted(checks.items()):
        matched=[x for x in items if x["byte_exact"] or x["dhash_distance"]<=10]
        horses={x["horse"] for x in matched}
        by_video[str(vid)]={"attempted_samples":len(items),"matched_samples":len(matched),
          "byte_exact_samples":sum(x["byte_exact"] for x in items),
          "best_dhash":min(x["dhash_distance"] for x in items),
          "horse_ids_from_matched":sorted(horses),
          "sample_indices":[{"original_frame":x["old_frame"],"candidate_highfps_frame":x["new_frame"],
              "dhash_distance":x["dhash_distance"],"exact":x["byte_exact"]} for x in items]}
    all_ok=len(by_video)==22 and all(v["matched_samples"]>=1 and len(v["horse_ids_from_matched"])==1 for v in by_video.values())
    output={"original_record":"10.5281/zenodo.14541977","higherfps_id_record":"10.5281/zenodo.17389614",
      "original_archive_md5":ORIGMD5,"cross_release_pixel_match_threshold":"byte exact OR 8x8 dHash distance <=10",
      "status":"CROSS_RELEASE_22_VIDEO_TO_11_HORSE_VISUAL_LINK_VERIFIED" if all_ok else "CROSS_RELEASE_PARTIAL_OR_FAILED",
      "videos":by_video,"n_matched_videos":sum(v["matched_samples"]>=1 for v in by_video.values())}
    dest=Path("reports/source_gates/haf2024_605_to_25fps_frame_linkage.json")
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(output,indent=2)+"\n")
    print("HAF_RELEASE_VISUAL_LINK_RESULT",json.dumps({"status":output["status"],"n_matched_videos":output["n_matched_videos"],
      "videos":len(by_video),"examples":list(by_video.items())[:3]}),flush=True)
    if not all_ok:raise RuntimeError("Cross-release visual link not established for all 22 videos; keep animal mapping provisional")
if __name__=="__main__":main()
