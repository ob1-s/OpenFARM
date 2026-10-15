"""Find actual direct cow video/rating spreadsheet links (or prove only page copy).

Checks raw site HTML (including iframe, embed, JSON scripts, uploads) without
inventing a download URL or dereferencing any access-controlled resource.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from urllib.parse import urljoin,urlparse
import requests
from bs4 import BeautifulSoup

SITES=("https://animalwelfarehub.org/cows-project/",
       "https://animalwelfarehub.org/new-data-repository/")
KEYWORDS=("cows-project","dairy","lameness","locomotion","video",
          "mp4","xlsx","csv","drive.google","docs.google","sharepoint","mega.nz",
          "dropbox","onedrive","zoho","iframe","download","wordpress")
def inspect(url):
    r=requests.get(url,timeout=35,headers={"User-Agent":"OpenFARM-project-data-link-inspection/0.1"})
    r.raise_for_status()
    html=r.text
    soup=BeautifulSoup(html,"html.parser")
    candidates=[]
    for tag in soup.select("a[href],iframe[src],source[src],video[src],embed[src],script[src],object[data]"):
        raw=tag.get("href") or tag.get("src") or tag.get("data")
        if not raw:continue
        resolved=urljoin(url,raw)
        desc=" ".join(tag.stripped_strings)[:80]
        attrs={k:str(v)[:120] for k,v in tag.attrs.items() if k in ("id","title","class","type")}
        if any(term in (raw+" "+desc).lower() for term in KEYWORDS) or tag.name in ("iframe","video","source","embed","object"):
            candidates.append({"tag":tag.name,"url":resolved,"description":desc,"attributes":attrs})
    # Find non-HTML inline URLs even in embedded wpblocks; no credentials.
    names=re.findall(r'https?[^"\\\s<>]+(?:\.xlsx|\.xls|\.csv|\.mp4|\.zip)[^"\\\s<>]*',html,re.I)
    return {"url":url,"html_bytes":len(r.content),
            "page_title":soup.title.get_text(" ",strip=True) if soup.title else None,
            "filelike_URL_literals":sorted(set(names))[:30],
            "candidate_external_or_media_refs":candidates[:140],
            "number_of_tags_with_links":len(soup.select("a[href],iframe[src],source[src],video[src],embed[src]")),
            "explicit_download_links":sum("download" in str(v).lower() for v in candidates),
            "note":"If no direct source is discoverable, requires author-host support; don't claim access verified."}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    out={}
    for site in SITES:
        try:out[site]=inspect(site)
        except Exception as e:out[site]={"error":f"{type(e).__name__}: {str(e)[:220]}"}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+"\n")
    print("COW_PROJECT_ACCESS:"+json.dumps(out))
