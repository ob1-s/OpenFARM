#!/usr/bin/env python3
"""Metadata-only source gate for 2020 Rashid et al. EquiFACS horse pain data.

Original paper: https://doi.org/10.1371/journal.pone.0231608
S1 data: 12 experimental video annotations (6 horses, 2 conditions).
S2 data: 21 clinical video annotations (3 expert veterinarian ratings reported).
This does NOT establish availability of the original 21 clinical videos.
Do not commit raw data or clinical records. No model inference.
"""
import argparse
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

from openpyxl import load_workbook

DOI = "10.1371/journal.pone.0231608"
SOURCE = "https://journals.plos.org/plosone/article/file?id=" + DOI
SUPPLEMENTS = {
    "experimental_annotations": SOURCE + ".s002&type=supplementary",
    "clinical_annotations": SOURCE + ".s003&type=supplementary",
}
VIDEO_ZIP = SOURCE + ".s001&type=supplementary"
MAX_XLSX_BYTES = 5 * 1024 * 1024


def request(url, range_header=None):
    headers = {"User-Agent": "OpenFARM-original-equine-PLOS-supplement-source-gate/0.2"}
    if range_header:
        headers["Range"] = range_header
    return urllib.request.Request(url, headers=headers)


def download_small_xlsx(url):
    with urllib.request.urlopen(request(url), timeout=75) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_XLSX_BYTES:
            raise ValueError("Unexpected oversize published XLSX")
        data = response.read(MAX_XLSX_BYTES + 1)
        if len(data) > MAX_XLSX_BYTES:
            raise ValueError("Source exceeds bounded XLSX download")
        if data[:2] != b"PK":
            raise ValueError("Original source was not a ZIP-format XLSX")
        return data, response.geturl()


def workbook_metadata(raw):
    wb = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    try:
        sheets = []
        for ws in wb.worksheets:
            # Do not expose ANY worksheet cell values, even a potential
            # 'header' row: the first row might contain research subjects.
            sheets.append({
                "sheet": ws.title,
                "reported_rows": ws.max_row,
                "reported_columns": ws.max_column,
            })
        return sheets
    finally:
        wb.close()


def video_range_probe(url):
    # Never stream the full ~880 MB video archive. Close after headers.
    with urllib.request.urlopen(request(url, "bytes=0-0"), timeout=50) as response:
        return {
            "http_status": response.status,
            "content_range": response.headers.get("Content-Range"),
            "content_length": response.headers.get("Content-Length"),
            "range_supported": response.status == 206,
            "final_url": response.geturl().split("?")[0],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("reports/source_gates/rashid_2020_equifacs_original_workbooks.json"))
    args = ap.parse_args()
    out = {
        "doi": DOI,
        "status": "ORIGINAL_CASE_LEVEL_BYTES_NOT_YET_VERIFIED",
        "published_claims": {
            "experiment_animals": 6,
            "experiment_condition_videos": 12,
            "clinical_animals": 21,
            "clinical_vet_raters": 3,
        },
        "source_rights": "PLOS paper CC BY; check supplement-level rights and permitted downstream reuse separately",
        "scope": "L2 horse facial AU and clinically rated video interpretation; not owner triage L3",
        "workbooks": {},
        "media_probe": None,
        "scientific_gates": [
            "Verify actual individual rating/vote columns vs preaggregated labels",
            "Validate unique horse identity and experimental/clinical overlap",
            "Determine whether raw 21 clinical videos are publicly accessible; do not invent them",
            "Preserve assessment boundary: human-coded FACS is not raw-video model perception",
            "No raw clinical/media/source table rows committed",
        ],
    }
    errors = []
    for name, url in SUPPLEMENTS.items():
        try:
            raw, final = download_small_xlsx(url)
            out["workbooks"][name] = {
                "access": "VERIFIED_ORIGINAL_XLSX_BYTES",
                "source_url": url,
                "redirected_host_or_path": final.split("?")[0],
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "sheets": workbook_metadata(raw),
            }
        except (Exception) as exc:
            out["workbooks"][name] = {"access": "BLOCKED", "source_url": url,
                                      "reason": type(exc).__name__ + ": " + str(exc)[:150]}
            errors.append(name)
    try:
        out["media_probe"] = video_range_probe(VIDEO_ZIP)
    except Exception as exc:
        out["media_probe"] = {"access": "BLOCKED_OR_RANGE_UNSUPPORTED",
                              "reason": type(exc).__name__ + ": " + str(exc)[:150]}
    out["status"] = (
        "TWO_ORIGINAL_ANNOTATION_WORKBOOKS_AUTHENTICATED_METADATA_ONLY"
        if not errors else "SOURCE_ACCESS_INCOMPLETE"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("RASHID2020_SOURCE_GATE_RESULT", json.dumps(out, sort_keys=True), flush=True)
    if errors:
        raise RuntimeError("Original EquiFACS published XLSX not yet authenticated: " + ",".join(errors))


if __name__ == "__main__":
    main()
