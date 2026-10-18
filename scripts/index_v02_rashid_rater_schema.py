#!/usr/bin/env python3
"""Original PLOS 2020 EquiFACS XLSX: aggregate rater/label-structure forensic.

No subject-level rows, no workbook cell contents, no names, or original source
material are included in output artifacts or logs. This script determines
whether source contains separable individual veterinarians, or only study's
aggregated pain labels, as far as the tabular structure allows.
"""
import hashlib,io,json,re,urllib.request
from collections import Counter
from pathlib import Path
from openpyxl import load_workbook

BASE="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0231608."
SOURCES={
    "experiment":("s002","2df498362259cbd75a5dc7eb40c580fef5368c53314a1d6451ffc22efe613612"),
    "clinical":("s003","f33fe32b8bd1283072df222038a36e6d9c34308adba7a174c6d6d47f960d2214"),
}
SMALL_SAFE={"pain","no pain","moderate pain","severe pain","yes","no","positive","negative",
            "true","false","0","1","2","3","p","np"}
def dtype(v):
    if v is None or str(v).strip()=="":return "empty"
    if isinstance(v,bool):return "bool"
    if isinstance(v,(float,int)):return "numeric"
    if getattr(v,"year",None) is not None:return "date"
    return "text"
def shape_column(values):
    nonempty=[v for v in values if dtype(v)!="empty"]
    typ=Counter(dtype(v) for v in nonempty)
    unique={str(v).strip().casefold() for v in nonempty}
    safe=bool(nonempty) and len(unique)<=8 and unique<=SMALL_SAFE
    return {
        "nonempty":len(nonempty),"unique_values":len(unique),"types":dict(typ),
        "allowlisted_small_label_counts":dict(sorted(Counter(str(v).strip().casefold() for v in nonempty).items())) if safe else None,
        "raw_data_exported":False,
    }
def main():
    report={"doi":"10.1371/journal.pone.0231608","status":"SOURCE_LEVEL_ONLY_NOT_ADMITTED",
            "source_rights":"Open paper; downstream reuse as allowed by applicable data provenance",
            "original_aggregate_labels_from_paper":{"clinical_horses":21,"majority_pain":7,"majority_nopain":14,
                 "three_veterinarians":True},
            "files":{}}
    for key,(suffix,digest) in SOURCES.items():
        url=BASE+suffix+"&type=supplementary"
        req=urllib.request.Request(url,headers={"User-Agent":"OpenFARM-original-rater-forensics/0.2"})
        with urllib.request.urlopen(req,timeout=90) as f:
            raw=f.read(3*1024*1024)
        assert hashlib.sha256(raw).hexdigest()==digest,"Pinned original supplemental XLSX changed"
        wb=load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
        sheets={}
        for ws in wb.worksheets:
            # Material is original public supplemental XLSX but do not export rows.
            cols=list(zip(*ws.values))
            info={"n_rows":ws.max_row,"n_columns":ws.max_column,"columns":[]}
            for i,col in enumerate(cols):
                # first row may be column header OR a data value; capture only if
                # it is a generic schema token (not a name/location/patient ID).
                header=str(col[0] or "").strip()
                safe_header=(len(header)<=40 and
                  bool(re.fullmatch(r"(?i)(?:AU|EAD|AD|FACS|Pain|Video|Horse|ID|Vet|Annotator|Rater|Time|Start|End|Label|Duration|Code)[A-Za-z0-9 _.-]*",header)))
                inf=shape_column(col[1:])
                inf["column_index_1based"]=i+1
                inf["safe_generic_header"]=header if safe_header else None
                info["columns"].append(inf)
            sheets[ws.title]=info
        wb.close()
        report["files"][key]={"sha256":digest,"sheets":sheets}
    out=Path("reports/source_gates/rashid2020_rater_schema_metadata.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for key,item in report["files"].items():
        for name,sh in item["sheets"].items():
            print("RASHID_SAFE_SCHEMA",json.dumps({
                "source":key,"sheet":name,"rows":sh["n_rows"],"cols":sh["n_columns"],
                "column_types":[{"index":c["column_index_1based"],"header":c["safe_generic_header"],
                  "types":c["types"],"nonempty":c["nonempty"],"uniques":c["unique_values"],
                  "safe_labels":c["allowlisted_small_label_counts"]} for c in sh["columns"]]
            },sort_keys=True))
if __name__=="__main__":main()
