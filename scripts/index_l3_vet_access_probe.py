#!/usr/bin/env python3
"""External download verification ONLY; no cases or vignette texts committed.

Run on a network-enabled GitHub Actions worker. Results contain status codes,
safe metadata, SHA256 and DOCX structural counts, not original patient records.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests

OUT = Path("reports/index_l3_vet_access_probe.json")
UA = {"User-Agent": "OpenFARM-SourceIntegrityProbe/1.0 (research verification)"}
S = requests.Session()
S.headers.update(UA)


def safe_url(url: str) -> str:
    x = urlparse(url)
    return f"{x.scheme}://{x.netloc}{x.path}"


def fetch(url: str, *, limit=1024 * 1024 * 2, timeout=25) -> tuple[dict, bytes]:
    d = {"url": safe_url(url)}
    try:
        with S.get(url, timeout=timeout, stream=True, allow_redirects=True) as resp:
            d.update(http_status=resp.status_code, final_url=safe_url(resp.url),
                     content_type=resp.headers.get("content-type", ""),
                     content_length=resp.headers.get("content-length", ""),
                     redirected=len(resp.history))
            if resp.status_code != 200:
                return d, b""
            content = bytearray()
            for chunk in resp.iter_content(128 * 1024):
                content.extend(chunk)
                if len(content) > limit:
                    d["over_size_limit"] = True
                    break
            if not d.get("over_size_limit"):
                d["bytes"] = len(content)
                d["sha256"] = hashlib.sha256(content).hexdigest()
            return d, bytes(content) if not d.get("over_size_limit") else b""
    except requests.RequestException as e:
        d["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return d, b""


def docx_stats(blob: bytes) -> dict:
    try:
        with zipfile.ZipFile(BytesIO(blob)) as z:
            names = z.namelist()
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                return {"valid_docx": False, "members": len(names)}
            main = z.read("word/document.xml")
            return {"valid_docx": True, "members": len(names),
                    "word_document_xml_bytes": len(main),
                    "paragraph_tags": len(re.findall(rb"<w:p(?: |>|\r|\n)", main)),
                    "table_tags": len(re.findall(rb"<w:tbl(?: |>|\r|\n)", main))}
    except Exception as e:
        return {"valid_docx": False, "error": str(e)[:160]}


def api_probe(url: str) -> tuple[dict, dict]:
    metadata, blob = fetch(url, limit=1024 * 1024 * 3)
    parsed = {}
    if blob:
        try:
            parsed = json.loads(blob)
        except ValueError:
            metadata["json_parse_error"] = True
    return metadata, parsed


def main():
    result = {
        "status": "SOURCE_BYTES_OR_METADATA_CHECK_NOT_BENCHMARK_ADMISSION",
        "wong": {}, "farrow": {},
    }
    # DataCite DOI registration and safe fields
    d, j = api_probe("https://api.datacite.org/dois/10.25910/rbk1-p833")
    a = j.get("data", {}).get("attributes", {})
    result["wong"]["datacite"] = {**d, "is_registered": bool(a.get("doi")),
        "doi": a.get("doi"), "title": (a.get("titles") or [{}])[0].get("title"),
        "publisher": a.get("publisher"), "version": a.get("version"),
        "rights": [x.get("rights") for x in a.get("rightsList", [])],
        "destination_url": safe_url(a["url"]) if a.get("url") else None}
    for key, url in [
        ("doi", "https://doi.org/10.25910/rbk1-p833"),
        ("wiley_supp", "https://bvajournals.onlinelibrary.wiley.com/doi/supinfo/10.1002/vetr.6126"),
        ("wiley_docx", "https://bvajournals.onlinelibrary.wiley.com/doi/suppl/10.1002/vetr.6126/supinfo/vetr6126-sup-0001-SuppMat.docx"),
    ]:
        d, blob = fetch(url, limit=1024 * 1024)
        result["wong"][key] = d
        if key == "wiley_docx" and blob:
            result["wong"][key]["docx"] = docx_stats(blob)
    # Figshare official public API / PLOS mirror
    for key, url in [
        ("plos_figshare", "https://api.figshare.com/v2/articles/31090093"),
        ("owner_data_figshare", "https://api.figshare.com/v2/articles/30884855"),
    ]:
        d, j = api_probe(url)
        meta = {"access": d, "article_id": j.get("id"),
                "title": j.get("title"), "doi": j.get("doi"),
                "license": (j.get("license") or {}).get("name") if isinstance(j.get("license"), dict) else None,
                "files": [{"id": f.get("id"), "name": f.get("name"), "size": f.get("size"),
                           "md5": f.get("supplied_md5") or f.get("computed_md5"),
                           "link_only": f.get("is_link_only", False)}
                          for f in j.get("files", [])]}
        result["farrow"][key] = meta
        if key == "plos_figshare":
            for f in j.get("files", []):
                if f.get("name", "").lower().endswith(".docx"):
                    fd, blob = fetch(f["download_url"], limit=1024 * 1024 * 3, timeout=40)
                    result["farrow"]["figshare_docx"] = {**fd, "name": f.get("name")}
                    if blob:
                        result["farrow"]["figshare_docx"]["docx"] = docx_stats(blob)
                    break
    for key,url in [
        ("plos_original", "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0339723.s001&type=supplementary"),
        ("pmc_mirror", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12810856/bin/pone.0339723.s001.docx"),
    ]:
        fd,blob = fetch(url, limit=1024*1024*3)
        result["farrow"][key] = {**fd, "docx": docx_stats(blob) if blob else None}
    OUT.parent.mkdir(exist_ok=True, parents=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for group, data in result.items():
        if isinstance(data, dict):
            print(group)
            for k, row in data.items():
                if isinstance(row, dict):
                    x = row.get("access", row)
                    print(" ", k, "status=", x.get("http_status"),
                          "bytes=", x.get("bytes"), "url=", x.get("final_url"),
                          "valid_docx=", (row.get("docx") or {}).get("valid_docx"),
                          "files=", len(row.get("files", [])),
                          "error=", x.get("error", ""))
    print("PROBE_RESULT", OUT)


if __name__ == '__main__':
    main()
