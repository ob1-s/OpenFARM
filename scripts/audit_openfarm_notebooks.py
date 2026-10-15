"""Static source-provenance archaeology for surviving OpenFARM data notebooks.

Only scans *cell source code* (not outputs, credentials, image attachments, or
full notebooks). Emits counts plus potentially relevant code locations.
Usage: python scripts/audit_openfarm_notebooks.py
"""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
KEYWORDS=re.compile(
    r"train_test_split|group.*split|GroupShuffle|StratifiedGroup|"
    r"audio_sha256|source_sha256|source_group_id|source_filename|"
    r"\bdhash\b|_raw\b|drop_duplicates|duplicated\(|"
    r"source_row_index|animal_id|cat_id|horse_id|"
    r"balanced|balance|valence|stress_binary|push_to_hub|"
    r"to_parquet|dataset_dict|shuffle\(",
    re.IGNORECASE
)
SENSITIVE=re.compile("token|secret|password|api.key|authorization|bearer",re.IGNORECASE)

def summarize(path):
    nb=json.loads(path.read_text(encoding="utf-8"))
    match_lines=[]
    for index,cell in enumerate(nb.get("cells",[])):
        if cell.get("cell_type")!="code":continue
        src=cell.get("source",[])
        if isinstance(src,list):src="".join(src)
        for line_no,line in enumerate(src.splitlines(),1):
            if KEYWORDS.search(line) and not SENSITIVE.search(line):
                safe=line.strip()
                if len(safe)>180:safe=safe[:180]+"…"
                match_lines.append({"cell":index,"line":line_no,"source":safe})
    return {
        "path":str(path.relative_to(ROOT)),
        "cells":len(nb.get("cells",[])),
        "code_cells":sum(c.get("cell_type")=="code" for c in nb.get("cells",[])),
        "executed_outputs":sum(bool(c.get("outputs")) for c in nb.get("cells",[])),
        "interesting_lines":match_lines,
    }

def main():
    paths=sorted((ROOT/"notebooks"/"data").glob("*.ipynb"))
    paths.extend(sorted((ROOT/"notebooks"/"unsloth").glob("*.ipynb")))
    for path in paths:
        report=summarize(path)
        lines=report["interesting_lines"]
        print(f"NOTEBOOK {report['path']} cells={report['cells']} "
              f"outputs={report['executed_outputs']} relevant_lines={len(lines)}")
        for r in lines[:30]:
            print(f"  [{r['cell']}:{r['line']}] {r['source']}")
        if len(lines)>30:
            print(f"  ... {len(lines)-30} further matches (not printed)")
if __name__=="__main__":main()
