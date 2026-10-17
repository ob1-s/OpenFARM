#!/usr/bin/env python3
"""Original Big Broiler MP4 ingestion proof from official 32.5 GB release.

Remote ZIP central directory -> one source MP4 + matching action CSV per setup.
Check genuine 25-fps decoding and positive label/timestamp correspondence.
No media/annotated rows are committed or uploaded; only file fingerprints and
aggregate decode/sync metadata. Does not claim model video understanding.
"""
import argparse,collections,csv,hashlib,io,json,os,re,shutil,subprocess,tempfile,time
from pathlib import Path
from remotezip import RemoteZip

SRC="https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
FPS=25.0
STEM=re.compile(r"(?P<vid>\d{3})_(?P<setup>[123])_\d+_\d+_\d+_[01]_(?:01|05|10)$")
def cmd(argv,timeout=120):
    run=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
    if run.returncode!=0:raise RuntimeError("decoder command failed: "+argv[0]+" status="+str(run.returncode))
    return run.stdout
def inspect(z,name):
    stem=Path(name).stem
    q=next((x for x in z.namelist() if x.endswith("/annotations/"+stem+".csv")),None)
    if not q:raise ValueError("original MP4 has no exact-stem annotation pair")
    maxframe=-1;pos=[];lastaction=collections.Counter();allrows=0
    sourcebytes=z.open(q).read()
    with io.TextIOWrapper(io.BytesIO(sourcebytes),encoding="utf-8-sig",newline="") as fp:
        reader=csv.DictReader(fp)
        if not {"frame","actions_ground"}.issubset(reader.fieldnames or []):raise ValueError("annotation schema missing")
        for row in reader:
            allrows+=1
            fr=int(row["frame"]);maxframe=max(maxframe,fr)
            action=row["actions_ground"] or ""
            labels=[x.strip() for x in re.split(r"[;,|]",action) if x.strip()]
            if "Feather_Pecking" in labels and len(pos)<64:pos.append(fr)
            if allrows<10000:
                for x in labels:lastaction[x]+=1
    return {"annotation_member":q,"n_annotation_rows":allrows,
            "max_annotated_frame":maxframe,"first_positive_frame":min(pos) if pos else None,
            "source_annotation_sha256":hashlib.sha256(sourcebytes).hexdigest()}
def candidate(z):
    entries=[]
    for name in z.namelist():
        if not name.endswith(".mp4") or "/videos/" not in name:continue
        stem=Path(name).stem;m=STEM.fullmatch(stem)
        if not m:continue
        info=z.getinfo(name)
        entries.append({"name":name,"setup":int(m["setup"]),"stem":stem,
                        "bytes":info.file_size,"compressed_bytes":info.compress_size})
    return entries

def main():
    p=argparse.ArgumentParser();p.add_argument("--max-per-setup",type=int,default=1)
    p.add_argument("--limit-bytes",type=int,default=600_000_000)
    a=p.parse_args()
    if a.max_per_setup!=1:raise ValueError("this bounded protocol allows one media per setup")
    out={"original_archive":SRC,"source_repo":"https://github.com/decide-ugent/big-broiler-dataset",
       "audio_video_note":"Video-native source-ingestion & annotation alignment only, NO visual model inference",
       "source_license":"AGPL-3.0 upstream repo; verify exact video/media redistribution rights separately",
       "media":[],"candidate_counts":{}}
    print("BROILER_VIDEO_PROBE_REMOTE_ZIP_CONNECT",flush=True)
    with RemoteZip(SRC,initial_buffer_size=4*1024*1024) as z:
        files=candidate(z)
        out["total_mp4_members"]=len(files)
        out["candidate_counts"]=dict(collections.Counter(str(x["setup"]) for x in files))
        if len(files)!=102 or sorted(set(x["setup"] for x in files))!=[1,2,3]:
            raise ValueError("Expected original 102 videos over exactly 3 setups")
        for setup in (1,2,3):
            options=sorted([x for x in files if x["setup"]==setup],key=lambda r:(r["bytes"],r["stem"]))
            filtered=[x for x in options if x["bytes"]<=a.limit_bytes]
            if not filtered:
                out["media"].append({"setup":setup,"status":"NO_SOURCE_CLIP_UNDER_LIMIT",
                  "smallest_source_bytes":options[0]["bytes"]})
                continue
            selected=None;meta=None
            for x in filtered[:min(12,len(filtered))]:
                try:
                    m=inspect(z,x["name"])
                    if m["first_positive_frame"] is not None:
                        selected=x;meta=m;break
                    if selected is None:selected=x;meta=m
                except Exception as ex:
                    print("BROILER_VIDEO_ANNOTATION_CANDIDATE_FAILED",setup,type(ex).__name__,flush=True)
            if selected is None:raise RuntimeError("No source annotation pairs in setup "+str(setup))
            with tempfile.TemporaryDirectory(prefix="openfarm_broiler_") as td:
                path=Path(td)/(selected["stem"]+".mp4")
                h=hashlib.sha256();count=0
                with z.open(selected["name"]) as source, open(path,"wb") as fp:
                    while True:
                        buffer=source.read(4*1024*1024)
                        if not buffer:break
                        fp.write(buffer);h.update(buffer);count+=len(buffer)
                if count!=selected["bytes"]:raise ValueError("MP4 extracted byte size mismatch")
                probe=json.loads(cmd(["ffprobe","-v","error","-select_streams","v:0",
                    "-show_entries","stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration",
                    "-of","json",str(path)]))
                video=(probe.get("streams") or [{}])[0]
                rate=video.get("avg_frame_rate","0/0")
                if "/" in rate:
                    num,den=[int(x) for x in rate.split("/")]
                    fps=num/den if den else 0
                else:fps=float(rate)
                if not 24.9<=fps<=25.1:raise ValueError("Source MP4 FPS does not match original 25 fps annotation protocol")
                nframes=int(video.get("nb_frames") or 0)
                if nframes<=0:raise ValueError("FFprobe cannot verify decoded frame count")
                # one normal frame plus one human-annotated feather-pecking frame if available
                t=[0,min(max(meta["max_annotated_frame"],0),nframes-1)//2]
                if meta["first_positive_frame"] is not None and meta["first_positive_frame"]<nframes:
                    t.append(meta["first_positive_frame"])
                stamps=sorted(set(int(x) for x in t))
                checked=[]
                for frame in stamps:
                    secs=frame/fps
                    # seek at known absolute timestamp and prove actual H.264 sample decoding.
                    digest=cmd(["ffmpeg","-hide_banner","-loglevel","error",
                        "-ss",f"{secs:.4f}","-i",str(path),"-frames:v","1",
                        "-f","md5","-"],timeout=90).decode().strip()
                    if not digest.startswith("MD5="):raise ValueError("No decoded frame from original MP4")
                    checked.append({"frame_index":frame,"timestamp_seconds":round(secs,4),
                                    "decoded_frame_md5":digest.split("=",1)[1]})
                out["media"].append({
                    "setup":setup,"status":"SOURCE_MP4_DECODED",
                    "source_video_stem":selected["stem"],"source_bytes":count,
                    "mp4_sha256":h.hexdigest(),"width":video.get("width"),"height":video.get("height"),
                    "source_fps":fps,"source_frame_count":nframes,
                    "annotation_rows":meta["n_annotation_rows"],
                    "annotation_member_sha256":meta["source_annotation_sha256"],
                    "annotation_max_frame":meta["max_annotated_frame"],
                    "feather_pecking_annotation_frame":meta["first_positive_frame"],
                    "decoded_frames":checked,
                    "observations":"Camera setup independently represented; decoding does not mean model behavior recognition."})
            print("BROILER_VIDEO_SOURCE_PROOF",json.dumps(out["media"][-1]),flush=True)
    summary={"n_verified_media":sum(x["status"]=="SOURCE_MP4_DECODED" for x in out["media"]),
         "n_decoded_source_frames":sum(len(x.get("decoded_frames",[])) for x in out["media"]),
         "n_setups":len(out["candidate_counts"]),"original_videos_in_archive":out["total_mp4_members"]}
    out["summary"]=summary
    dest=Path("reports/decision-index/big_broiler_original_video_probe.json")
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+"\n")
    print("BROILER_VIDEO_PROBE_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    if summary["n_verified_media"]!=3:raise RuntimeError("Did not verify an original MP4 in every camera setup")
if __name__=="__main__":main()
