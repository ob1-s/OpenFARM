#!/usr/bin/env python3
"""Independent Grimace L2 cross-lab source audit: Vidal 2022 white furred mice.

Audit exact author-released grimace/face reference JSON and source image-host
availability without redistributing images or committing animal-level rows.
"""
import collections,hashlib,json,urllib.request,urllib.error,re
from pathlib import Path

BASE="https://raw.githubusercontent.com/AndreaVidal/WhiteFurredMice_Dataset/main/Annotation_json/"
IMAGE_HOST="https://lab-msp.com/WhiteFurredMiceDataset/"
DOI="10.1016/j.mlwa.2022.100312"
def get(url,lim=5_000_000):
    req=urllib.request.Request(url,headers={"User-Agent":"OpenFARM-scientific-source-audit/0.2"})
    with urllib.request.urlopen(req,timeout=30) as r:return r.read(lim)
def summarize(blob,kind):
    records=json.loads(blob)
    if not isinstance(records,dict):raise ValueError("Original annotation file must be keyed by image name")
    stats=collections.Counter();animal=collections.Counter();schema=collections.Counter()
    examples=set()
    for key,values in records.items():
        if not key.lower().endswith(".png") or not isinstance(values,list) or not values:
            raise ValueError("Non-image or empty source annotation entry")
        for x in values:
            if not isinstance(x,dict) or "AnimalNumber" not in x or "Boundingbox" not in x:raise ValueError("source detection schema mismatch")
            animal[str(x['AnimalNumber'])]+=1
            bbox=x['Boundingbox']
            if not set(("x","y","w","h")).issubset(bbox):raise ValueError("required localization absent")
            schema.update(x.keys())
            if kind=="eyes":
                if x.get("GrimaceScale") not in ("0","1","2"):raise ValueError("Non-ordinal grimace label")
                stats[x['GrimaceScale']]+=1
        examples.add(key)
    return {'unique_image_keys':len(examples),'annotations':sum(animal.values()),
      'animal_count':len(animal),'per_animal_annotation_counts_sorted':sorted(animal.values()),
      'ordinal_grimace_distribution':dict(stats),'source_fields':sorted(schema),
      'sha256_original_json':hashlib.sha256(blob).hexdigest()}
def main():
    out={'doi':DOI,'annotation_repo':'https://github.com/AndreaVidal/WhiteFurredMice_Dataset',
      'source_lab_images_url':IMAGE_HOST,'source_license':'Not defined in GitHub repository: verify original image and label reuse rights before redistribution/commercial training.',
      'interpretation':'GrimaceScale refers to orbital tightening/eye region in Vidal 2022, not five independent facial action units like GrimACE.',
      'status':'NOT_VERIFIED','original_files':{}}
    try:
        for key,part in [('eyes','eyes_detection_data.json'),('face','face_detection_data.json')]:
            x=get(BASE+part)
            out['original_files'][key]=summarize(x,key)
        if out['original_files']['eyes']['animal_count']!=8:raise ValueError('unexpected animal identity count')
        try:
            page=get(IMAGE_HOST,1_000_000).decode(errors='replace')
            links=re.findall(r'href=[\"\']([^\"\']+?\\.zip(?:\\?[^\"\']*)?)[\"\']',page,re.I)
            out['source_image_download_site']={'accessible':True,'html_bytes':len(page),
              'zip_archive_link_count':len(links),'archive_filenames':[x.split('/')[-1] for x in links],
              'license_metadata_present':bool(re.search(r'Creative Commons|license|research only',page,re.I)),
              'landing_html_excerpt':page[:850]}
        except urllib.error.HTTPError as error:
            out['source_image_download_site']={'accessible':False,'status_code':error.code}
        except Exception as error:
            out['source_image_download_site']={'accessible':False,'error_type':type(error).__name__}
        out['status']='ANNOTATION_BYTES_VERIFIED_IMAGES_RIGHTS_UNVERIFIED'
    except Exception as error:
        out['status']='ANNOTATION_GATE_FAILED';out['error_type']=type(error).__name__
    file=Path('reports/source_gates/vidal_2022_mouse_orbital_grimace.json')
    file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(out,indent=2)+'\n')
    print('VIDAL_INDEPENDENT_GRIMACE_GATE',json.dumps(out,sort_keys=True),flush=True)
    if out['status']!='ANNOTATION_BYTES_VERIFIED_IMAGES_RIGHTS_UNVERIFIED':raise SystemExit(2)
if __name__=='__main__':main()
