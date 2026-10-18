#!/usr/bin/env python3
"""Original 2026 Dryad 48-calf L2 independent-physiology SOURCE GATE.

Retrieves two primary XLSX datasets, hashes original bytes, audits animal/date
identities and candidate longitudinal join keys. Does not assign model-ready
pain/welfare labels, train a model, or upload clinical/wound raw records.
"""
from __future__ import annotations
import collections,hashlib,json,os,re,time,urllib.request,urllib.error,zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
DOI="10.5061/dryad.k3j9kd5qt"
URLS={
"mnt":"https://datadryad.org/downloads/file_stream/4850856",
"tissue":"https://datadryad.org/downloads/file_stream/4850857"}
NAMES={"mnt":"Disbudding_Calves_2016_MNT_Dryad.xlsx",
       "tissue":"Disbudding_Calves_2016_Tissue_Dryad.xlsx"}
S="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"OpenFARM-Research-SourceAudit/0.2","Accept":"application/octet-stream"})
    with urllib.request.urlopen(req,timeout=60) as r:
        return r.read(10_000_000)
def source_file(key):
    last=None
    candidates=[URLS[key]]
    # Primary Dryad download links from its published HTML. Try verified
    # API mirror only when the named stable file_stream endpoint fails.
    for url in candidates:
        try:
            blob=fetch(url)
            if not zipfile.is_zipfile(__import__("io").BytesIO(blob)):
                raise ValueError("download did not return an XLSX ZIP")
            return blob,url
        except Exception as e:last=type(e).__name__
    raise RuntimeError("Dryad source fetch inaccessible: "+key+" "+str(last))
def spreadsheet(blob):
    import io
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=set(z.namelist())
        if "xl/worksheets/sheet1.xml" not in names:raise ValueError("first data sheet missing")
        strings=[]
        if "xl/sharedStrings.xml" in names:
            root=ET.fromstring(z.read("xl/sharedStrings.xml"))
            strings=["".join(node.text or "" for node in item.iter(S+"t")) for item in root.findall(S+"si")]
        sheet=ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows=[]
    for row in sheet.findall(".//"+S+"sheetData/"+S+"row"):
        cols={}
        for c in row.findall(S+"c"):
            ref=c.get("r") or ""
            match=re.match(r"([A-Z]+)[0-9]+",ref)
            if not match:continue
            ix=0
            for char in match.group(1):ix=ix*26+ord(char)-64
            ix-=1
            v=c.find(S+"v")
            if c.get("t")=="inlineStr":txt="".join(x.text or "" for x in c.iter(S+"t"))
            elif v is None:txt=""
            elif c.get("t")=="s":txt=strings[int(v.text)]
            else:txt=v.text or ""
            cols[ix]=txt
        rows.append(cols)
    if not rows:raise ValueError("Empty original study worksheet")
    header=rows[0]
    names=[header.get(i,"") for i in range(max(header)+1)]
    if not {"Calf","Rep","Treatment","Side"}.issubset(names):
        raise ValueError("Study group identifier columns missing from original")
    records=[{names[i]:v for i,v in row.items() if i<len(names)} for row in rows[1:]]
    return names,records
def summarize(records,key):
    def series(field):
        return [r.get(field,"") for r in records]
    calves=series("Calf");reps=series("Rep")
    if any(not x for x in calves):raise ValueError("Calf IDs missing; no reliable animal grouping")
    cols=set().union(*(set(x) for x in records))
    fields={name:{"nonblank":sum(str(r.get(name,"")).strip() not in ("","#N/A") for r in records)}
            for name in ["Calf","Rep","Side","Treatment","Days_disbudded","MNT_N","Rate_N_s","Test_date","Date","Exudate","Epithelium"] if name in cols}
    datefield=next((x for x in ("Test_date","Date","Day","Days_disbudded") if x in cols),None)
    return {"rows":len(records),"unique_calf_ids":len(set(calves)),
      "unique_replicate_cohorts":len(set(reps)),"unique_treatments":dict(collections.Counter(series("Treatment"))),
      "columns":sorted(cols),"selected_field_completeness":fields,"temporal_field":datefield,
      "side_counts":dict(collections.Counter(series("Side")))}
def join_diagnostic(a,b):
    # Only count overlap of animal+horn-side ID; temporal joins require later review
    pa=set((r.get("Calf"),r.get("Side")) for r in a)
    pb=set((r.get("Calf"),r.get("Side")) for r in b)
    ca=set(x[0] for x in pa);cb=set(x[0] for x in pb)
    return {"calves_in_both_sources":len(ca&cb),"calves_only_mnt":len(ca-cb),
       "calves_only_tissue":len(cb-ca),"matching_animal_side_pairs":len(pa&pb),
       "note":"No same-time or future-dependent feature join is approved by this diagnostic."}
def main():
    out={"source_doi":DOI,"status":"SOURCE_ACCESS_UNCHECKED",
         "project":"UC Davis 2016 calf original instrument+healing datasets, Dryad posted 2026",
         "required_study_animals":48,"source_information":{},
         "clinical_target_caution":"MNT_N is actual nociceptive algometer force, not expert visual pain intensity or analgesic intervention outcome"}
    caches={}
    try:
        for key in ("mnt","tissue"):
            data,url=source_file(key)
            columns,records=spreadsheet(data)
            caches[key]=records
            out["source_information"][key]={"file_name":NAMES[key],
                "source_url":url,"byte_count":len(data),
                "sha256":hashlib.sha256(data).hexdigest(),
                **summarize(records,key)}
            print("CALF_ORIGINAL_SOURCE_FETCH",key,"rows",len(records),"calves",len(set(r['Calf'] for r in records)),flush=True)
        out["join_gate"]=join_diagnostic(caches["mnt"],caches["tissue"])
        if out["join_gate"]["calves_in_both_sources"]!=48:
            raise ValueError("Original source IDs do not establish 48 overlapping calves")
        out["status"]="ORIGINAL_BYTES_VERIFIED_JOIN_CANDIDATE_NOT_BENCHMARK"
    except Exception as error:
        out["status"]="SOURCE_GATE_FAILED"
        out["error_type"]=type(error).__name__
        out["error_message"]=str(error)[:200]
    report=Path("reports/source_gates/dryad_2026_calves_original_gate.json")
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(out,indent=2)+"\n")
    print("CALF_SOURCE_GATE_REPORT",json.dumps({"status":out['status'],"source_n":len(out["source_information"]),
       "join":out.get("join_gate"),"error_type":out.get("error_type"),"error_message":out.get("error_message")},sort_keys=True))
    if out["status"]!="ORIGINAL_BYTES_VERIFIED_JOIN_CANDIDATE_NOT_BENCHMARK":raise SystemExit(2)
if __name__=="__main__":main()
