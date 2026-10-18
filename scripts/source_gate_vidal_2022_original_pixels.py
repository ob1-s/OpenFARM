#!/usr/bin/env python3
"""Vidal 2022 external 8-mouse orbital-tightening L2 image identity gate.

Downloads ORIGINAL 470 MB author archive to temporary GH runner; matches every
annotated image ID to its source PNG; decodes two per mouse. Never uploads
original pixels, images, image embeddings, or annotation rows. Rights still
require publisher/lab permission assessment for commercial training.
"""
import collections,hashlib,io,json,os,random,tempfile,urllib.request,zipfile
from pathlib import Path

ANN="https://raw.githubusercontent.com/AndreaVidal/WhiteFurredMice_Dataset/main/Annotation_json/eyes_detection_data.json"
ZIP="https://lab-msp.com/WhiteFurredMiceDataset/Eyes_detection.zip"
EXPECTED_ANN_SHA="754d3442b056b14aee137643adab101c166dee27f42338397ef7a1f869cb9ad2"
def fetch(url,maxbytes=3_000_000):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"OpenFARM-L2-SourceIntegrity/0.2"}),timeout=50) as f:return f.read(maxbytes)
def main():
    import PIL.Image
    ann=fetch(ANN)
    if hashlib.sha256(ann).hexdigest()!=EXPECTED_ANN_SHA:raise ValueError("Original annotations changed")
    source=json.loads(ann)
    expected=set(source)
    if len(expected)!=1087:raise ValueError("Original 1087 human grimace records mismatch")
    stats={"study":"Vidal et al. 2022, DOI 10.1016/j.mlwa.2022.100312",
      "source_zip_url":ZIP,"original_annotation_sha256":EXPECTED_ANN_SHA,
      "n_annotated_frames":1087,"n_source_mice":8,
      "source_license_status":"Unresolved: public directory availability is not redistribution permission.",
      "status":"UNVERIFIED", "source_image_dimensions":{}, "sampled_original_image_hashes":[]}
    with tempfile.TemporaryDirectory(prefix="openfarm_vidal_external_") as dirname:
        path=Path(dirname)/"Eyes_detection.zip"
        h=hashlib.sha256();total=0
        req=urllib.request.Request(ZIP,headers={"User-Agent":"OpenFARM-L2-SourceIntegrity/0.2"})
        with urllib.request.urlopen(req,timeout=120) as response,open(path,"wb") as out:
            while data:=response.read(2*1024*1024):
                out.write(data);h.update(data);total+=len(data)
                if total>800_000_000:raise ValueError("Original eye archive exceeds review budget")
        if not zipfile.is_zipfile(path):raise ValueError("Author download not a ZIP")
        stats.update(original_zip_sha256=h.hexdigest(),original_zip_bytes=total)
        print("VIDAL_PIXEL_ARCHIVE_DOWNLOADED",total,flush=True)
        with zipfile.ZipFile(path) as z:
            mapping=collections.defaultdict(list)
            for name in z.namelist():
                if name.lower().endswith(".png"):mapping[Path(name).name].append(name)
            matches=expected&set(mapping)
            duplicates=[name for name in expected if len(mapping[name])!=1]
            missing=expected-set(mapping)
            stats.update(archive_png_images=sum(map(len,mapping.values())),
                n_unique_archive_basenames=len(mapping),matched_original_annotations=len(matches),
                duplicate_source_names=len(duplicates),missing_annotation_images=len(missing))
            if missing or duplicates:raise ValueError("Original author images not one-to-one with annotated 1087 files")
            by_mouse=collections.defaultdict(list)
            for name,entries in source.items():
                r=entries[0]
                by_mouse[str(r["AnimalNumber"])].append(name)
            selected=[]
            for mouse,names in sorted(by_mouse.items()):
                selected.extend(random.Random(20261018+int(mouse)).sample(sorted(names),min(2,len(names))))
            for name in selected:
                b=z.read(mapping[name][0])
                with PIL.Image.open(io.BytesIO(b)) as im:
                    w,height=im.size
                    im.verify()
                r=source[name][0]
                box=r["Boundingbox"]
                x,y,bw,bh=(int(box[k]) for k in ("x","y","w","h"))
                bbox_valid=(0<=x<w and 0<=y<height and bw>0 and bh>0
                            and (x+bw<=w and y+bh<=height or bw<=w and bh<=height))
                stats["sampled_original_image_hashes"].append({
                    "subject_pseudonym":"mouse_"+str(r["AnimalNumber"]),
                    "width":w,"height":height,"bbox_plausible":bbox_valid,
                    "source_image_sha256":hashlib.sha256(b).hexdigest()})
                stats["source_image_dimensions"][f"{w}x{height}"]=stats["source_image_dimensions"].get(f"{w}x{height}",0)+1
            if not all(x["bbox_plausible"] for x in stats["sampled_original_image_hashes"]):
                raise ValueError("Original image-to-box geometry mismatch")
            stats["status"]="ORIGINAL_PIXELS_AND_1087_LABEL_KEYS_VERIFIED_RIGHTS_PENDING"
    dest=Path("reports/source_gates/vidal_2022_independent_pixels_verified.json")
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(stats,indent=2)+"\n")
    print("VIDAL_PIXELS_SOURCE_GATE",json.dumps({key:stats.get(key) for key in (
      "status","original_zip_sha256","original_zip_bytes","archive_png_images",
      "matched_original_annotations","sampled_original_image_hashes")},sort_keys=True),flush=True)
    if stats["status"]!="ORIGINAL_PIXELS_AND_1087_LABEL_KEYS_VERIFIED_RIGHTS_PENDING":
        raise SystemExit(2)
if __name__=="__main__":main()
