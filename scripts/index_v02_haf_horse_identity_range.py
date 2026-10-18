#!/usr/bin/env python3
"""HAF 2024 equine image video->animal identity recovery from the author's 25fps follow-up.

Never download the ~5.5 GB media archive. Verify HTTP 206 byte-range first,
then query ZIP central directory for original filenames containing ID=n.
Author filenames independently document mapping; each video maps to 1 horse ID.

Do NOT assume this 2025 revision adds independent animals/cases to the original
605 images. This is a potential identity metadata source for the SAME 22 videos.
"""
import collections,json,re,urllib.request
from pathlib import Path
from remotezip import RemoteZip

REC="https://zenodo.org/api/records/17389614"
PAT=re.compile(r"cropped_(pain|nopain)_video(\d+)_25_(\d+)_ID=(\d+)\.png",re.I)
def fetch_json(url):
    req=urllib.request.Request(url,headers={"User-Agent":"OpenFARM-original-study-animal-ID-audit/0.2"})
    with urllib.request.urlopen(req,timeout=40) as r:return json.loads(r.read(500000))
def probe_206(url):
    req=urllib.request.Request(url,headers={"Range":"bytes=-65536","User-Agent":"OpenFARM-source-ZIP-range-check/0.2"})
    with urllib.request.urlopen(req,timeout=45) as r:
        status=r.status;limit=r.read(120000)
        if status!=206 or len(limit)>70000:raise RuntimeError("NON_RANGED_BIG_ARCHIVE (refuse 2.7GB transfer)")
        return {"code":status,"content_range":r.headers.get("Content-Range")}
def main():
    meta=fetch_json(REC)
    entries=meta.get("files",[])
    chosen=[x for x in entries if "25 fps" in (x.get("key") or "").lower() or "id.zip" in (x.get("key") or "").lower()]
    if len(chosen)!=2:
        chosen=[x for x in entries if "med ID" in (x.get("key") or "")]
    if len(chosen)!=2:raise ValueError("Original 2025 source file inventory ambiguous")
    out={"record_id":17389614,"doi":"10.5281/zenodo.17389614",
         "status":"SOURCE_RANGE_UNVERIFIED","ranged_files":[],"video_to_animal":{},
         "note":"Horse identity metadata from same 22 videos; not additional independent samples."}
    mapping=collections.defaultdict(set);counts=collections.Counter()
    for item in chosen:
        name=item["key"];url=(item.get("links",{}).get("self") or ("https://zenodo.org/records/17389614/files/"+urllib.parse.quote(name)+"?download=1"))
        ranged=probe_206(url)
        print("HORSE_ID_SOURCE_RANGE_OK",json.dumps({"name":name,"range":ranged}),flush=True)
        info={"file":name,"source_size":item.get("size"),"range_status":ranged,
              "parsed_entries":0,"unparsed_entries":0,"video_ids":[]}
        with RemoteZip(url,initial_buffer_size=262144) as archive:
            for path in archive.namelist():
                if not path.lower().endswith(".png"):continue
                m=PAT.fullmatch(Path(path).name)
                if not m:
                    info["unparsed_entries"]+=1;continue
                pain,video,frame,animal=m.groups()
                mapping[int(video)].add(animal);counts[(int(video),animal)]+=1
                info["parsed_entries"]+=1
        info["video_ids"]=sorted({video for (video,_) in counts})
        out["ranged_files"].append(info)
        print("HORSE_ID_ZIP_DIRECTORY",json.dumps({"file":name,"parsed":info["parsed_entries"],"unparsed":info["unparsed_entries"]}),flush=True)
    unique_anon=set().union(*mapping.values())
    out["video_to_animal"]={str(k):list(v) for k,v in sorted(mapping.items())}
    out["distinct_video_ids"]=len(mapping);out["distinct_horse_ids"]=len(unique_anon)
    out["conflicting_video_horse_mapping"]={str(k):list(v) for k,v in mapping.items() if len(v)!=1}
    out["status"]="SOURCE_VIDEO_TO_ANIMAL_MAP_VERIFIED" if len(mapping)==22 and len(unique_anon)==11 and not out["conflicting_video_horse_mapping"] else "MAPPING_INCOMPLETE"
    dest=Path("reports/source_gates/haf2024_605frame_animal_mapping.json")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2)+"\n")
    print("HORSE_ID_RECOVERY_SUMMARY",json.dumps({"status":out["status"],"videos":len(mapping),"horses":len(unique_anon),
           "conflicts":len(out["conflicting_video_horse_mapping"])}),flush=True)
    if out["status"]!="SOURCE_VIDEO_TO_ANIMAL_MAP_VERIFIED":raise RuntimeError("Animal identity evidence incomplete")
if __name__=="__main__":main()
