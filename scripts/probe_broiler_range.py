"""Source metadata only: inspect a 33GB public ZIP using genuine HTTP byte ranges."""
import json,hashlib
from remotezip import RemoteZip
URL="https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
r={"source":URL}
try:
    with RemoteZip(URL,initial_buffer_size=4*1024*1024) as z:
        infos=z.infolist()
        csv=[v for v in infos if v.filename.lower().endswith(".csv")]
        r["total_files"]=len(infos);r["csv_files"]=len(csv)
        r["annotation_entries"]=[{"file":i.filename,"bytes":i.file_size,"compressed":i.compress_size} for i in csv[:17]]
        ids=sorted([v for v in csv if "annotations/" in v.filename.lower()],key=lambda v:v.filename)
        r["annotation_files"]=len(ids)
        if ids:
            for entry in [ids[0],ids[-1]]:
                if entry.file_size>90_000_000:
                    r.setdefault("samples",[]).append({"file":entry.filename,"skipped":">90MB"})
                    continue
                with z.open(entry) as f:
                    content=f.read(500)
                    r.setdefault("samples",[]).append({"file":entry.filename,"first_bytes":content.decode("utf-8","replace")})
except Exception as e:
    r["error"]=f"{type(e).__name__}: {str(e)[:500]}"
print("BROILER_RANGE_PROBE:"+json.dumps(r))
if "error" in r:raise SystemExit(2)
