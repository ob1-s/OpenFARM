#!/usr/bin/env python3
"""Ask et al Scientific Reports 2024 published supplementary data SOURCE GATE.

Original 348 video-sequence EquiFACS summaries vs gait-derived orthopedic
pain intensity. Supplements are not the raw 37,072 per-AU start/stop events.
Read workbook XML with Python stdlib; do not publish per-horse observations.
"""
import collections,hashlib,json,tempfile,urllib.request,zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

URL="https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-023-50383-y/MediaObjects/41598_2023_50383_MOESM1_ESM.xlsx"
DOI="10.1038/s41598-023-50383-y"
NS="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
def fetch():
    req=urllib.request.Request(URL,headers={"User-Agent":"OpenFARM-scientific-source-validation/0.2","Accept":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
    with urllib.request.urlopen(req,timeout=90) as resp: blob=resp.read(20_000_000)
    if not blob.startswith(b"PK") or not 10_000<len(blob)<10_000_000:
        raise ValueError("Invalid original supplementary workbook transfer")
    return blob
def col_index(code):
    out=0
    for x in code:
        if not x.isalpha():break
        out=out*26+ord(x.upper())-ord("A")+1
    return out
def main():
    raw=fetch();digest=hashlib.sha256(raw).hexdigest()
    out={"study":"Ask et al 2024 Scientific Reports 14:129","doi":DOI,
      "source_uri":URL,"file_bytes":len(raw),"original_workbook_sha256":digest,
      "reported_animals":8,"reported_video_sequences":348,
      "reported_equifacs_action_event_count":37072,
      "only_aggregated_AU_features_original_raw_AU_timings_not_publicly_verified":True,
      "workbook_sheets":[]}
    with zipfile.ZipFile(__import__('io').BytesIO(raw)) as z:
        shared=[]
        if "xl/sharedStrings.xml" in z.namelist():
            ss=ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall(NS+"si"):
                shared.append("".join((t.text or "") for t in si.iter(NS+"t")))
        w=ET.fromstring(z.read("xl/workbook.xml"))
        names=[x.attrib.get("name","?") for x in w.find(NS+"sheets").findall(NS+"sheet")]
        files=sorted((n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")))
        if len(files)!=len(names):raise ValueError("Sheet inventory mapping differs from source")
        for sheet_name,path in zip(names,files):
            root=ET.fromstring(z.read(path))
            table=root.find(NS+"sheetData")
            lines=table.findall(NS+"row") if table is not None else []
            max_columns=0;first=[]
            for row in lines[:15]:
                cells=[]
                for c in row.findall(NS+"c"):
                    rawv=c.find(NS+"v");v=rawv.text if rawv is not None else ""
                    if c.attrib.get("t")=="s" and v:
                        v=shared[int(v)]
                    elif c.attrib.get("t")=="inlineStr":
                        v="".join(x.text or "" for x in c.iter(NS+"t"))
                    cells.append((c.attrib.get("r"),str(v)[:100]))
                    max_columns=max(max_columns,col_index(c.attrib.get("r","")))
                if cells:first.append(cells[:30])
            # Only sparse titles/header hints. Do not expose individual data rows.
            data_header_cells=[{"coordinate":ref,"label":v} for row in first[:5] for ref,v in row
               if any(c.isalpha() for c in v) and len(v)<100][:80]
            out["workbook_sheets"].append({"name":sheet_name,"path":path,"xml_row_count":len(lines),
                "top_header_hints":data_header_cells,"max_columns_in_first_15_rows":max_columns})
    p=Path("reports/source_gates/ask2024_equiFACS_original_supplement_gate.json")
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+"\n")
    print("ASK2024_SOURCE_GATE",json.dumps({"sha256":digest,"bytes":len(raw),
        "sheets":[{"name":x["name"],"rows":x["xml_row_count"],"header_hints":x["top_header_hints"][:12]} for x in out["workbook_sheets"]]}),flush=True)
    if not out["workbook_sheets"]:raise RuntimeError("Original workbook has no sheets")
if __name__=="__main__":main()
