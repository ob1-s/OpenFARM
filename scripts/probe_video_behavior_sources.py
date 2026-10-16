"""Inspect *actual* public Meerkat human video annotations and Big Broiler partial ZIP.

No source media or training data is redistributed. Fail if remote metadata
cannot be verified. This is an access gate; not model benchmarking.
"""
from __future__ import annotations
import json,requests,io,zipfile,hashlib,collections
M="https://meerkat-dataset.github.io/Annotations.zip"
B="https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
out={}
try:
    with requests.get(M,timeout=(20,65),stream=True) as r:
        r.raise_for_status()
        blob=bytearray()
        for ch in r.iter_content(262144):
            blob.extend(ch)
            if len(blob)>380_000_000:raise RuntimeError("Meerkat annotations exceeded 380MB cap")
    z=zipfile.ZipFile(io.BytesIO(blob))
    listing=[{"name":n.filename,"bytes":n.file_size} for n in z.infolist() if not n.is_dir()]
    tiny=[]
    for item in listing:
        if len(tiny)>=8:break
        if item["bytes"]<=50000 and (item["name"].lower().endswith((".json",".txt",".csv"))):
            x=z.read(item["name"])
            tiny.append({"file":item["name"],"start":x[:700].decode("utf-8","replace")})
    samples=[]
    for item in listing[:2]+listing[-3:-1]:
        if not item["name"].endswith(".csv"):continue
        with z.open(item["name"]) as fp:
            samples.append({"file":item["name"],"first_5_lines":[fp.readline().decode("utf-8","replace").strip() for _ in range(5)]})
    out["meerkat"]={"source":M,"zip_bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest(),"file_count":len(listing),"files":listing[:100],"tiny_samples":tiny,"csv_samples":samples}
except Exception as e:out["meerkat"]={"error":f"{type(e).__name__}: {e}"}
try:
    with requests.get(B,headers={"Range":"bytes=0-1023"},timeout=(16,40),stream=True) as r:
        status=r.status_code
        header={k:r.headers.get(k) for k in ("Content-Length","Content-Range","Accept-Ranges","Content-Type")}
        content=r.raw.read(1024,decode_content=False)
        out["big_broiler"]={"range_status":status,"headers":header,"first_bytes":content[:20].hex()}
except Exception as e:out["big_broiler"]={"error":f"{type(e).__name__}: {e}"}
print("BEHAVIOR_SOURCE_PROBE:"+json.dumps(out))
if "error" in out["meerkat"]:raise SystemExit(2)
